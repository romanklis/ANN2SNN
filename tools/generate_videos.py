#!/usr/bin/env python3
"""Generate MP4 videos of the ball balancing on the plate.

For each benchmark controller -- **Random ANN**, **Fly-like ANN**,
**SNN transferred** and **PID controlled** -- this script drives the closed-loop
ball-and-plate simulation in :mod:`sim_engine` and renders an MP4 animation of
the ball on the tilting plate, the reference orbit, the radial error, the plate
tilt command and (for the spiking brain) the spike raster.

The clips are written in a stable order::

    videos/01_random_ann.mp4
    videos/02_flylike_ann.mp4
    videos/03_snn_transferred.mp4
    videos/04_pid.mp4
    videos/all_controllers_sequence.mp4      # the four clips concatenated

Usage
-----
    python3 tools/generate_videos.py                 # distil, then render
    python3 tools/generate_videos.py --no-train      # fast untrained baselines
    python3 tools/generate_videos.py --weights weights.pt
    python3 tools/generate_videos.py --jobs 4        # render the 4 clips in parallel
    python3 tools/generate_videos.py --only pid      # (re)render a single clip
    python3 tools/generate_videos.py --stride 2 --dpi 90

The learned brains are behaviourally distilled from the PID teacher when
``--train`` is set (the default) so they actually balance; ``--no-train`` keeps
the fast untrained baselines (random weights diverge, which is the benchmark's
honest random baseline).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

# --------------------------------------------------------------------------- #
# Repo / package import
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np  # noqa: E402

from sim_engine.config import EMBODIMENT_PRESETS  # noqa: E402

try:  # running as a script (`python3 tools/generate_videos.py`)
    from render import (  # type: ignore
        DEFAULT_COLORS,
        Rollout,
        VideoInfo,
        ffmpeg_available,
        probe_video,
        render_video,
    )
except ImportError:  # running as a module (`python3 -m tools.generate_videos`)
    from tools.render import (  # type: ignore
        DEFAULT_COLORS,
        Rollout,
        VideoInfo,
        ffmpeg_available,
        probe_video,
        render_video,
    )

__all__ = ["main", "SEQUENCE", "RenderJob"]

#: (registry key, human label, one-line description) in task order.
SEQUENCE = [
    ("random_ann", "Random ANN",
     "untrained dense feed-forward ANN (4-1000-2, random weights)"),
    ("flylike_ann", "Fly-like ANN",
     "sparse recurrent connectome ANN (Dale's law, distilled from the PID teacher)"),
    ("snn_transferred", "SNN transferred",
     "lossless micro-stepping IF spiking network, transferred from the fly-like ANN"),
    ("pid", "PID controlled",
     "classical PD controller with acceleration feed-forward"),
]

DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "videos")
DEFAULT_CACHE = os.environ.get("ANN2SNN_VIDEO_CACHE", "/tmp/ann2snn_video_cache")


# --------------------------------------------------------------------------- #
# Rollout helpers
# --------------------------------------------------------------------------- #
@dataclass
class RenderJob:
    """Everything a worker needs to render one clip (picklable)."""

    key: str
    label: str
    description: str
    color: str
    rollout_npz: str
    reference_npz: str
    out_path: str
    fps: int = 30
    dpi: int = 100
    stride: int = 1
    plate_half: float = 0.25
    max_tilt: float = 0.25


def _render_worker(job_dict: dict) -> dict:
    """Module-level worker so ``multiprocessing`` (spawn) can import it."""
    job = RenderJob(**job_dict)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        rollout = Rollout.load(job.rollout_npz)
        ref = np.load(job.reference_npz, allow_pickle=False)
        info = render_video(
            rollout,
            ref_pos=ref["ref_pos"],
            dt=float(ref["dt"]),
            radius=float(ref["radius"]),
            out_path=job.out_path,
            label=job.label,
            description=job.description,
            color=job.color,
            fps=job.fps,
            dpi=job.dpi,
            stride=job.stride,
            plate_half=job.plate_half,
            max_tilt=job.max_tilt,
        )
        info.key = job.key
        d = info.to_dict()
        d["error"] = None
        return d
    except Exception as exc:  # noqa: BLE001 - report, don't kill the pool
        return {
            "key": job.key,
            "label": job.label,
            "error": f"{type(exc).__name__}: {exc}",
        }


def concat_videos(paths, out_path: str):
    """Concatenate MP4s with ffmpeg (re-encode for a seamless join)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in paths:
            fh.write(f"file '{os.path.abspath(p)}'\n")
        list_file = fh.name
    tmp_out = out_path + ".part.mp4"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "veryfast", "-movflags", "+faststart", tmp_out,
    ]
    try:
        subprocess.run(cmd, check=True)
        probe_video(tmp_out)              # verify before publishing
        os.replace(tmp_out, out_path)
    except Exception as exc:  # pragma: no cover
        print(f"  ! ffmpeg concat failed: {exc}", file=sys.stderr)
        try:
            os.unlink(tmp_out)
        except OSError:
            pass
        return None
    finally:
        try:
            os.unlink(list_file)
        except OSError:
            pass
    return out_path


def _write_status(outdir: str, status: dict) -> None:
    tmp = os.path.join(outdir, ".status.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(status, fh, indent=2, default=str)
    os.replace(tmp, os.path.join(outdir, "status.json"))


def _build_engine(args):
    """Construct a :class:`sim_engine.engine.Engine` per the CLI flags."""
    from sim_engine.config import EmbodimentConfig, EngineConfig, TrainingConfig
    from sim_engine.engine import Engine

    weights = args.weights
    if weights and not os.path.isabs(weights):
        weights = os.path.join(REPO_ROOT, weights)

    overrides = dict(
        steps=args.steps,
        radius=args.radius,
        freq=args.freq,
        seed=args.seed,
        epochs=args.epochs,
        train_on_init=bool(args.train and not weights),
        weights_path=weights,
        device="cpu",
    )
    if args.n_neurons:
        overrides["n_neurons"] = args.n_neurons
    if getattr(args, "embodiment", None):
        overrides["embodiment_preset"] = args.embodiment
    if getattr(args, "profile", "clean") == "robust":
        overrides["training"] = TrainingConfig(
            profile="robust",
            epochs=args.epochs,
            embodiment=EmbodimentConfig.from_preset(args.embodiment or "embodied"),
        )
    cfg = EngineConfig(**overrides)
    engine = Engine(cfg, train=(True if (args.train and not weights) else None))

    if args.train and not weights and args.save_weights:
        path = args.save_weights
        if not os.path.isabs(path):
            path = os.path.join(REPO_ROOT, path)
        engine.save_weights(path)
        print(f"      weights saved -> {path}", flush=True)
    return engine


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate ball-and-plate balancing videos for the controllers."
    )
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--steps", type=int, default=250,
                    help="control frames (250 = the canonical 5 s benchmark)")
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-neurons", type=int, default=None)
    ap.add_argument("--no-train", dest="train", action="store_false", default=True,
                    help="skip behavioural distillation (fast, untrained baselines)")
    ap.add_argument("--weights", default=None,
                    help="load a weights bundle instead of distilling")
    ap.add_argument("--save-weights", default="weights.pt",
                    help="where to save distilled weights when training")
    ap.add_argument("--embodiment", default=None, choices=sorted(EMBODIMENT_PRESETS),
                    help="embodied environment preset for the closed loop")
    ap.add_argument("--profile", default="clean", choices=["clean", "robust"],
                    help="distillation profile (robust = train inside the environment)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1,
                    help="render one video frame every STRIDE control frames")
    ap.add_argument("--plate-half", type=float, default=0.25)
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel render processes (default 4)")
    ap.add_argument("--only", default=None,
                    help="comma-separated controller keys to (re)render only")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="reuse cached rollouts instead of re-simulating")
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    if not ffmpeg_available():
        print("error: ffmpeg/ffprobe not found on PATH; cannot encode MP4",
              file=sys.stderr)
        return 2

    import torch

    from sim_engine.config import BenchmarkConfig
    from sim_engine.benchmark import run_closed_loop
    from sim_engine.physics import DT, MAX_TILT
    from sim_engine.reference import orbit_reference
    from sim_engine.serialization import to_jsonable

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    train = bool(args.train)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sequence = SEQUENCE
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        sequence = [s for s in SEQUENCE if s[0] in wanted]
        missing = wanted - {s[0] for s in sequence}
        if missing:
            print(f"error: unknown --only key(s): {sorted(missing)}", file=sys.stderr)
            return 2

    status = {"stage": "starting", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "controllers": [s[0] for s in sequence], "errors": []}
    _write_status(args.outdir, status)

    # -- reference + config -------------------------------------------------- #
    reference = orbit_reference(steps=args.steps, radius=args.radius,
                                freq=args.freq, device="cpu")
    from sim_engine.config import EmbodimentConfig
    from sim_engine.environment import EmbodiedEnv

    embodiment = (EmbodimentConfig.from_preset(args.embodiment)
                  if getattr(args, "embodiment", None) else EmbodimentConfig())
    config = BenchmarkConfig(steps=args.steps, radius=args.radius, freq=args.freq,
                             record_spikes=True, embodiment=embodiment)
    init_state = torch.tensor([-0.05, 0.05, 0.0, 0.0])
    ref_npz = os.path.join(args.cache_dir, "reference.npz")
    np.savez(ref_npz, ref_pos=reference.pos.cpu().numpy(),
             dt=reference.dt, radius=args.radius)

    # -- build (and, if requested, distil) the brains ------------------------ #
    all_keys = [s[0] for s in sequence]
    reuse = args.reuse_cache or bool(args.only)
    ctrl_cache = {k: os.path.join(args.cache_dir, f"{k}.npz") for k in all_keys}
    need_sim = [k for k in all_keys if not (reuse and os.path.exists(ctrl_cache[k]))]

    if need_sim:
        t0 = time.time()
        status["stage"] = "building controllers"
        _write_status(args.outdir, status)
        print(f"[1/4] building controllers  (train={train}, epochs={args.epochs}, "
              f"seed={args.seed})", flush=True)
        engine = _build_engine(args)
        if engine.last_training is not None:
            loss = engine.last_training.get("final_loss")
            print(f"      trained (final loss={loss})", flush=True)
        print(f"      done in {time.time() - t0:.1f}s\n", flush=True)

        status["stage"] = "simulating"
        _write_status(args.outdir, status)
        print("[2/4] running closed loop", flush=True)
        for key, label, _desc in sequence:
            npz = ctrl_cache[key]
            if reuse and os.path.exists(npz):
                continue
            ctrl = engine.build_controller(key)
            env = None
            if not embodiment.is_clean:
                env = EmbodiedEnv(embodiment, dt=reference.dt, max_tilt=MAX_TILT,
                                  init_state=init_state)
            res = run_closed_loop(ctrl, reference=reference, init_state=init_state,
                                  config=config, name=label, env=env)
            Rollout(
                name=key,
                trajectory=np.asarray(res.trajectory),
                tilts=np.asarray(res.tilts),
                tracking_error=np.asarray(res.tracking_error),
                mean_error_cm=float(res.mean_error_cm),
                spikes=(None if res.spikes is None else np.asarray(res.spikes)),
            ).save(npz)
            print(f"      {label:<16s} mean {res.mean_error_cm:8.3f} cm -> "
                  f"{os.path.basename(npz)}", flush=True)
    else:
        print("[1/4] controllers reused from cache\n"
              "[2/4] rollouts reused from cache", flush=True)

    # -- render one clip per controller (in parallel) ------------------------ #
    outdir = args.outdir
    index_of = {k: i for i, (k, _l, _d) in enumerate(SEQUENCE, start=1)}
    jobs = []
    for key, label, desc in sequence:
        jobs.append(RenderJob(
            key=key, label=label, description=desc,
            color=DEFAULT_COLORS.get(key, "#1f77b4"),
            rollout_npz=ctrl_cache[key], reference_npz=ref_npz,
            out_path=os.path.join(outdir, f"{index_of[key]:02d}_{key}.mp4"),
            fps=args.fps, dpi=args.dpi, stride=args.stride,
            plate_half=args.plate_half, max_tilt=MAX_TILT,
        ))

    status["stage"] = "rendering"
    _write_status(outdir, status)
    n_jobs = max(1, min(args.jobs, len(jobs)))
    print(f"[3/4] rendering {len(jobs)} clip(s) with {n_jobs} worker(s)", flush=True)

    t0 = time.time()
    payloads = [asdict(j) for j in jobs]
    if n_jobs == 1:
        results = [_render_worker(p) for p in payloads]
    else:
        try:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=n_jobs) as pool:
                results = pool.map(_render_worker, payloads)
        except Exception as exc:  # noqa: BLE001 - fall back to sequential
            print(f"      ! parallel render failed ({exc}); falling back to serial",
                  file=sys.stderr, flush=True)
            results = [_render_worker(p) for p in payloads]

    videos = []
    meta_by_key = {}
    for r in results:
        if r.get("error"):
            print(f"      ! {r['key']}: {r['error']}", file=sys.stderr, flush=True)
            status["errors"].append({"key": r["key"], "error": r["error"]})
            continue
        info = VideoInfo(**{
            k: r[k] for k in
            ("key", "label", "path", "frames", "duration_s", "size_bytes",
             "width", "height", "mean_error_cm")
        })
        videos.append(info)
        meta_by_key[info.key] = r.get("mean_error_cm")
        print(f"      {info.label:<16s} {os.path.basename(info.path):<26s} "
              f"{info.frames:4d} frames  {info.duration_s:5.2f}s  "
              f"{info.size_bytes / 1e6:5.2f} MB", flush=True)
    print(f"      rendered in {time.time() - t0:.1f}s", flush=True)

    # -- concatenate the clips "in sequence" --------------------------------- #
    combined = None
    if not args.no_combined and len(videos) > 1:
        status["stage"] = "concatenating"
        _write_status(outdir, status)
        print("\n[4/4] concatenating the clips in sequence", flush=True)
        combined_path = os.path.join(outdir, "all_controllers_sequence.mp4")
        combined = concat_videos([v.path for v in videos], combined_path)
        print(f"      -> {os.path.basename(combined_path) if combined else 'FAILED'}",
              flush=True)
    else:
        print("\n[4/4] skipping combined video", flush=True)

    # -- metrics sidecar ----------------------------------------------------- #
    metrics = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "trained": train,
        "epochs": args.epochs if train else 0,
        "benchmark": {
            "steps": args.steps, "dt": DT, "radius": args.radius,
            "freq": args.freq, "duration_s": round(args.steps * DT, 3),
            "init_state": [float(v) for v in init_state],
            "plate_half_m": args.plate_half, "max_tilt_rad": MAX_TILT,
        },
        "environment": {
            "preset": getattr(args, "embodiment", None) or "clean",
            "profile": getattr(args, "profile", "clean"),
        },
        "video": {"fps": args.fps, "dpi": args.dpi, "stride": args.stride,
                  "codec": "H.264 / libx264 / yuv420p"},
        "sequence": [k for k, _l, _d in SEQUENCE],
        "videos": [
            {"key": v.key, "label": v.label, "file": os.path.basename(v.path),
             "frames": v.frames, "duration_s": round(v.duration_s, 3),
             "width": v.width, "height": v.height, "size_bytes": v.size_bytes,
             "mean_error_cm": v.mean_error_cm}
            for v in videos
        ],
        "combined_video": os.path.basename(combined) if combined else None,
        "errors": status["errors"],
        "controllers": CTRL_META,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump(to_jsonable(metrics), fh, indent=2)

    status["stage"] = "done" if not status["errors"] else "done_with_errors"
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["videos"] = [os.path.basename(v.path) for v in videos]
    status["combined_video"] = os.path.basename(combined) if combined else None
    _write_status(outdir, status)

    print("\n" + "=" * 74)
    print(f"BALL-BALANCING VIDEOS  ({args.steps} frames @ {1 / DT:.0f} Hz, "
          f"{args.steps * DT:.1f} s, trained={train})")
    print("=" * 74)
    for v in videos:
        print(f"  {v.label:<16s} {os.path.basename(v.path):<26s} "
              f"{v.frames:4d} frames  {v.duration_s:5.2f}s  "
              f"{v.size_bytes / 1e6:5.2f} MB")
    if combined:
        print(f"  combined -> {os.path.basename(combined)}")
    print("=" * 74)
    print("ALL_DONE")
    return 0 if not status["errors"] else 1


#: static description of every controller, copied into metrics.json
CTRL_META = {
    key: {"label": label, "description": desc}
    for key, label, desc in SEQUENCE
}


if __name__ == "__main__":
    raise SystemExit(main())
