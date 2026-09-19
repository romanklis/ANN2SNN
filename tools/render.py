"""Engine-agnostic MP4 renderer for ball-and-plate rollouts.

This module turns a closed-loop rollout (plant trajectory, applied tilts, radial
tracking error and, for spiking brains, a spike raster) into a polished 4-panel
animation written with matplotlib + ffmpeg:

* **top-down plate** - ball, trail, reference orbit, live reference point and the
  commanded tilt vector, with a status HUD;
* **radial tracking error** vs time;
* **plate tilt command** bars for ``theta_x``/``theta_y``;
* **spike raster** (spiking brains only).

It deliberately knows nothing about :mod:`sim_engine`: callers pass plain numpy
arrays (see :class:`Rollout`), so the same renderer is used by the
``tools/generate_videos.py`` CLI and by the Flask ``/api/export/mp4`` endpoint.

Every clip is rendered to ``*.part.mp4``, validated with ``ffprobe`` (non-zero
duration / ``moov`` present) and only then atomically renamed into place, so a
killed render never leaves a truncated file behind.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional

# matplotlib needs a writable config dir in headless containers
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

__all__ = [
    "Rollout",
    "VideoInfo",
    "probe_video",
    "ffmpeg_available",
    "render_video",
    "render_rollout",
]

#: default per-controller colours (used when the caller passes no colour)
DEFAULT_COLORS = {
    "random_ann": "#7f7f7f",
    "flylike_ann": "#2a9d8f",
    "snn_transferred": "#8338ec",
    "dense_ann": "#e07a5f",
    "pid": "#1f77b4",
}


# --------------------------------------------------------------------------- #
# Data holders
# --------------------------------------------------------------------------- #
@dataclass
class Rollout:
    """The per-controller signals the renderer needs (a numpy-only snapshot)."""

    name: str
    trajectory: np.ndarray            # (T, 4) plant states [x, y, vx, vy]
    tilts: np.ndarray                 # (T, 2) applied actuator commands [rad]
    tracking_error: np.ndarray        # (T,) instantaneous radial error [cm]
    mean_error_cm: float = float("nan")
    spikes: "Optional[np.ndarray]" = None  # (T, N) or None

    # -- (de)serialisation -------------------------------------------------- #
    def save(self, path: str) -> str:
        kw = dict(
            name=np.asarray(self.name),
            trajectory=np.asarray(self.trajectory, dtype=np.float32),
            tilts=np.asarray(self.tilts, dtype=np.float32),
            tracking_error=np.asarray(self.tracking_error, dtype=np.float32),
            mean_error_cm=np.asarray(self.mean_error_cm, dtype=np.float64),
        )
        if self.spikes is not None:
            kw["spikes"] = np.asarray(self.spikes, dtype=np.float32)
        np.savez(path, **kw)
        return path

    @classmethod
    def load(cls, path: str) -> "Rollout":
        d = np.load(path, allow_pickle=False)
        spikes = d["spikes"] if "spikes" in d.files else None
        return cls(
            name=str(d["name"]),
            trajectory=d["trajectory"],
            tilts=d["tilts"],
            tracking_error=d["tracking_error"],
            mean_error_cm=float(d["mean_error_cm"]),
            spikes=spikes,
        )

    @classmethod
    def from_result(cls, result, *, name: Optional[str] = None) -> "Rollout":
        """Build a rollout from a ``sim_engine`` ``TrajectoryResult``-like object."""
        spikes = getattr(result, "spikes", None)
        return cls(
            name=name or getattr(result, "name", "controller"),
            trajectory=np.asarray(result.trajectory, dtype=float),
            tilts=np.asarray(result.tilts, dtype=float),
            tracking_error=np.asarray(result.tracking_error, dtype=float),
            mean_error_cm=float(getattr(result, "mean_error_cm", np.nan)),
            spikes=None if spikes is None else np.asarray(spikes, dtype=float),
        )


@dataclass
class VideoInfo:
    key: str
    label: str
    path: str
    frames: int
    duration_s: float
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    mean_error_cm: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# External tool checks
# --------------------------------------------------------------------------- #
def ffmpeg_available() -> bool:
    """True when both ``ffmpeg`` and ``ffprobe`` are on ``PATH``."""
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def probe_video(path: str) -> dict:
    """Return ``{duration, frames, width, height, codec}`` for a finished MP4.

    Raises ``RuntimeError`` if the file is missing a ``moov`` atom (i.e. it is a
    truncated render).
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe not found on PATH")
    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe rejected {os.path.basename(path)}: {proc.stderr.strip()}"
        )
    data = json.loads(proc.stdout)
    vstream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if vstream is None:
        raise RuntimeError(f"no video stream in {os.path.basename(path)}")
    duration = float(
        data.get("format", {}).get("duration", vstream.get("duration", 0.0)) or 0.0
    )
    if duration <= 0.0:
        raise RuntimeError(
            f"{os.path.basename(path)} reports zero duration (truncated)"
        )
    nb = vstream.get("nb_frames")
    if nb not in (None, "N/A"):
        frames = int(nb)
    else:
        try:
            num, den = vstream.get("r_frame_rate", "30/1").split("/")
            fps = float(num) / max(1.0, float(den))
        except Exception:  # noqa: BLE001
            fps = 30.0
        frames = int(round(duration * fps))
    return {
        "duration": duration,
        "frames": frames,
        "width": int(vstream.get("width", 0)),
        "height": int(vstream.get("height", 0)),
        "codec": vstream.get("codec_name", ""),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_video(
    result: Rollout,
    *,
    ref_pos: np.ndarray,
    dt: float,
    radius: float,
    out_path: str,
    label: str,
    description: str = "",
    color: str = "#1f77b4",
    fps: int = 30,
    dpi: int = 100,
    plate_half: float = 0.25,
    stride: int = 1,
    max_tilt: float = 0.25,
    title: Optional[str] = None,
) -> VideoInfo:
    """Render one closed-loop :class:`Rollout` to an MP4 (atomically)."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH; cannot encode MP4")

    traj = np.asarray(result.trajectory, dtype=float)        # (T, 4)
    tilts = np.asarray(result.tilts, dtype=float)            # (T, 2)
    err = np.asarray(result.tracking_error, dtype=float)     # (T,) cm
    T = len(traj)
    if T == 0:
        raise ValueError("cannot render an empty rollout")
    t_axis = np.arange(T) * dt
    ref_pos = np.asarray(ref_pos, dtype=float)[:T]
    orbit_r = float(radius)

    spikes = getattr(result, "spikes", None)
    spikes = np.asarray(spikes, dtype=float) if spikes is not None else None

    frames = list(range(0, T, max(1, stride)))
    if frames[-1] != T - 1:
        frames.append(T - 1)

    # -- figure / panel layout --------------------------------------------- #
    fig = plt.figure(figsize=(11.2, 6.4))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.82, 0.82],
        left=0.055, right=0.975, top=0.855, bottom=0.075,
        wspace=0.24, hspace=0.62,
    )
    ax_plate = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_tilt = fig.add_subplot(gs[1, 1])
    ax_spk = fig.add_subplot(gs[2, 1])

    fig.suptitle(title or f"{label} - ball balancing on the plate",
                 fontsize=14, fontweight="bold")
    if description:
        fig.text(0.5, 0.905, description, ha="center", va="center",
                 fontsize=9, color="#555")

    # -- plate panel (top-down view) --------------------------------------- #
    L = float(plate_half)
    ax_plate.set_xlim(-L, L)
    ax_plate.set_ylim(-L, L)
    ax_plate.set_aspect("equal", adjustable="box")
    ax_plate.set_xlabel("x [m]")
    ax_plate.set_ylabel("y [m]")

    ax_plate.add_patch(
        Rectangle((-L, -L), 2 * L, 2 * L, facecolor="#f4f7fb",
                  edgecolor="#33404d", lw=1.8, zorder=0)
    )
    for g in np.linspace(-L, L, 5):
        ax_plate.axvline(g, color="#e2e8f0", lw=0.7, zorder=0)
        ax_plate.axhline(g, color="#e2e8f0", lw=0.7, zorder=0)
    ax_plate.axhline(0.0, color="#b6c2cf", lw=0.9, zorder=1)
    ax_plate.axvline(0.0, color="#b6c2cf", lw=0.9, zorder=1)

    th = np.linspace(0.0, 2.0 * np.pi, 256)
    ax_plate.plot(orbit_r * np.cos(th), orbit_r * np.sin(th), "--",
                  color="#8a94a6", lw=1.2, zorder=2, label="reference orbit")

    trail_line, = ax_plate.plot([], [], "-", color=color, lw=2.2, alpha=0.9,
                                zorder=3, label="ball path")
    ref_dot, = ax_plate.plot([], [], "o", color="#e4572e", ms=9, zorder=4,
                             label="reference")
    ball = ax_plate.scatter([], [], s=190, color=color, edgecolor="white",
                            linewidth=1.4, zorder=6, label="ball")
    tilt_arrow, = ax_plate.plot([], [], "-", color="#e4572e", lw=2.4, alpha=0.75,
                                zorder=5, solid_capstyle="round")
    tilt_tip, = ax_plate.plot([], [], "o", color="#c1121f", ms=5, zorder=5)
    status_txt = ax_plate.text(
        0.025, 0.035, "", transform=ax_plate.transAxes, fontsize=9.5,
        family="monospace", va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1", alpha=0.92),
    )
    ax_plate.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # -- radial error panel ------------------------------------------------ #
    err_max = float(np.max(err)) if T else 0.0
    err_top = max(5.0, min(err_max, 60.0)) * 1.12
    x_max = float(t_axis[-1]) if T > 1 else 1.0
    ax_err.set_xlim(0.0, x_max)
    ax_err.set_ylim(0.0, err_top)
    err_line, = ax_err.plot([], [], "-", color=color, lw=1.7)
    err_dot, = ax_err.plot([], [], "o", color=color, ms=5)
    ax_err.axhline(min(float(result.mean_error_cm), err_top), color="#e4572e",
                   ls=":", lw=1.1)
    clip_note = "  (axis clipped at 60 cm)" if err_max > 60.0 else ""
    ax_err.set_title(f"radial tracking error [cm]{clip_note}", fontsize=9.5)
    ax_err.set_xlabel("time [s]", fontsize=8)
    ax_err.tick_params(labelsize=8)
    ax_err.grid(alpha=0.25)

    # -- tilt-command panel ------------------------------------------------ #
    ax_tilt.set_xlim(-max_tilt * 1.15, max_tilt * 1.15)
    ax_tilt.set_ylim(-0.75, 1.75)
    bars = ax_tilt.barh([1, 0], [0.0, 0.0], height=0.5,
                        color=["#4c72b0", "#dd8452"], zorder=3)
    ax_tilt.axvline(0.0, color="#94a3b8", lw=0.9, zorder=2)
    ax_tilt.axvline(max_tilt, color="#cbd5e1", ls=":", lw=0.9, zorder=1)
    ax_tilt.axvline(-max_tilt, color="#cbd5e1", ls=":", lw=0.9, zorder=1)
    ax_tilt.set_yticks([1, 0])
    ax_tilt.set_yticklabels(["$\\theta_x$", "$\\theta_y$"], fontsize=9)
    ax_tilt.set_title("plate tilt command [rad]  (saturation $\\pm 0.25$)",
                      fontsize=9.5)
    ax_tilt.tick_params(labelsize=8)
    tilt_txt = ax_tilt.text(0.5, 0.92, "", transform=ax_tilt.transAxes,
                            ha="center", va="top", fontsize=8.5, family="monospace")

    # -- spike-raster panel ------------------------------------------------ #
    raster_img = None
    raster_vline = None
    spike_data = None
    n_neurons = 0
    if spikes is not None and spikes.ndim == 2 and spikes.shape[1] > 0:
        n_neurons = int(spikes.shape[1])
        n_show = min(n_neurons, 400)
        idx = np.linspace(0, n_neurons - 1, n_show).astype(int)
        spike_data = spikes[:, idx].T   # (n_show, T)
        blank = np.zeros_like(spike_data)
        raster_img = ax_spk.imshow(
            blank, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0,
            interpolation="nearest", extent=[0, T, n_show, 0], zorder=2,
        )
        raster_vline = ax_spk.axvline(0, color="#e4572e", lw=1.0, zorder=3)
        ax_spk.set_title(f"spike raster ({n_show} of {n_neurons} neurons)",
                         fontsize=9.5)
        ax_spk.set_xlim(0, T)
        ax_spk.set_ylim(n_show, 0)
        ax_spk.set_xlabel("frame", fontsize=8)
        ax_spk.set_ylabel("neuron", fontsize=8)
        ax_spk.tick_params(labelsize=8)
    else:
        ax_spk.axis("off")
        ax_spk.text(0.5, 0.55, "non-spiking controller", ha="center", va="center",
                    fontsize=10, color="#64748b", transform=ax_spk.transAxes)
        ax_spk.text(0.5, 0.38, "(no spike raster)", ha="center", va="center",
                    fontsize=9, color="#94a3b8", transform=ax_spk.transAxes)

    arrow_scale = L * 0.45 / max(max_tilt, 1e-9)
    artists = [ref_dot, trail_line, ball, tilt_arrow, tilt_tip, status_txt,
               err_line, err_dot, bars[0], bars[1], tilt_txt]

    def update(i: int):
        k = frames[i]
        # --- plate
        ref_dot.set_data([ref_pos[k, 0]], [ref_pos[k, 1]])
        trail_line.set_data(traj[: k + 1, 0], traj[: k + 1, 1])
        bx, by = float(traj[k, 0]), float(traj[k, 1])
        inside = (abs(bx) <= L) and (abs(by) <= L)
        ball.set_offsets([[float(np.clip(bx, -L, L)), float(np.clip(by, -L, L))]])
        ball.set_color(color if inside else "#d62728")

        tx, ty = float(tilts[k, 0]), float(tilts[k, 1])
        ax_, ay_ = -tx * arrow_scale, -ty * arrow_scale
        tilt_arrow.set_data([0.0, ax_], [0.0, ay_])
        tilt_tip.set_data([ax_], [ay_])

        status_txt.set_text(
            f"t = {t_axis[k]:5.2f} s   frame {k:3d}/{T}\n"
            f"radial error = {err[k]:6.2f} cm\n"
            f"status: {'ON PLATE' if inside else 'BALL LEFT PLATE'}"
        )
        status_txt.set_color("#0f7a3d" if inside else "#c62828")

        # --- radial error
        err_line.set_data(t_axis[: k + 1], err[: k + 1])
        err_dot.set_data([t_axis[k]], [min(float(err[k]), err_top)])

        # --- tilt bars
        bars[0].set_width(float(tilts[k, 0]))
        bars[1].set_width(float(tilts[k, 1]))
        tilt_txt.set_text(
            f"$\\theta_x$={tilts[k, 0]:+.3f}  $\\theta_y$={tilts[k, 1]:+.3f}"
        )

        # --- spike raster
        if raster_img is not None:
            masked = np.zeros_like(spike_data)
            masked[:, : k + 1] = spike_data[:, : k + 1]
            raster_img.set_data(masked)
            raster_vline.set_xdata([k, k])

        return artists

    if raster_img is not None:
        artists = artists + [raster_img, raster_vline]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000.0 / max(1, fps),
        blit=True, cache_frame_data=False,
    )
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=[
            # force even dimensions (yuv420p requirement) regardless of dpi
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast",
            "-movflags", "+faststart",
        ],
    )

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".part.mp4"
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    try:
        anim.save(tmp_path, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)

    meta = probe_video(tmp_path)
    os.replace(tmp_path, out_path)

    return VideoInfo(
        key="", label=label, path=out_path, frames=int(meta["frames"]),
        duration_s=float(meta["duration"]),
        size_bytes=os.path.getsize(out_path),
        width=int(meta["width"]), height=int(meta["height"]),
        mean_error_cm=float(result.mean_error_cm),
    )


def render_rollout(
    result: Rollout,
    reference_pos,
    *,
    dt: float,
    radius: float,
    out_path: str,
    label: Optional[str] = None,
    **kwargs,
) -> VideoInfo:
    """Convenience wrapper: render a :class:`Rollout` given the orbit positions."""
    label = label or result.name
    color = kwargs.pop("color", None) or DEFAULT_COLORS.get(result.name, "#1f77b4")
    return render_video(
        result,
        ref_pos=np.asarray(reference_pos, dtype=float),
        dt=dt,
        radius=radius,
        out_path=out_path,
        label=label,
        color=color,
        **kwargs,
    )
