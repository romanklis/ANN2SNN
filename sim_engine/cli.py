"""Command-line entrypoint: ``python -m sim_engine <command>``.

Commands
--------
``benchmark``  run the closed-loop orbit benchmark for one or more controllers
``train``      behavioral distillation of the PD teacher into the ANNs
``session``    drive an interactive session and print/emit the trajectory
``describe``   dump the engine configuration + controller catalogue as JSON
``list``       list the available controllers

Examples
--------
::

    python -m sim_engine list
    python -m sim_engine benchmark --controllers pid,random_ann,flylike_ann,snn_transferred \\
        --steps 250 --out results.json
    python -m sim_engine train --epochs 100 --save weights.pt
    python -m sim_engine session --controller snn_transferred --frames 20
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .api import build_engine, config_from_dict, list_controllers
from .engine import Engine
from .physics import DT
from .registry import CANONICAL_CONTROLLERS


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--steps", type=int, default=250, help="control frames in the orbit")
    p.add_argument("--radius", type=float, default=0.15, help="orbit radius [m]")
    p.add_argument("--freq", type=float, default=0.5, help="orbit frequency [Hz]")
    p.add_argument("--neurons", type=int, default=1000, help="hidden neurons per brain")
    p.add_argument("--micro-steps", type=int, default=10, help="SNN micro-steps per frame")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--epochs", type=int, default=100, help="distillation epochs (with --train)")
    p.add_argument("--device", default="cpu", help="torch device (cpu/cuda/auto)")
    p.add_argument("--weights", default=None, help="path to a trained .pt weights bundle")
    p.add_argument("--train", action="store_true", help="distil weights before running")


def _engine_from_args(args) -> Engine:
    cfg = {
        "steps": args.steps,
        "radius": args.radius,
        "freq": args.freq,
        "micro_steps": args.micro_steps,
        "n_neurons": args.neurons,
        "seed": args.seed,
        "device": args.device,
        "weights_path": args.weights,
        "train_on_init": bool(getattr(args, "train", False)),
    }
    return build_engine(cfg, train=bool(getattr(args, "train", False)))


def cmd_list(args) -> int:
    print(json.dumps(list_controllers(), indent=2))
    return 0


def cmd_describe(args) -> int:
    engine = _engine_from_args(args)
    print(json.dumps(engine.describe(), indent=2))
    return 0


def cmd_benchmark(args) -> int:
    engine = _engine_from_args(args)
    names = args.controllers.split(",") if args.controllers else list(CANONICAL_CONTROLLERS)
    report = engine.run_benchmark(names, include_trace=not args.no_trace, as_dict=False)

    # JSON to stdout (or file)
    payload = report.to_dict(include_trace=not args.no_trace)
    text = json.dumps(payload)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")
    elif args.quiet:
        pass
    else:
        print(text[:0])  # nothing; summary below

    if not args.quiet:
        print("=" * 66)
        print("CLOSED-LOOP TRACKING BENCHMARK (Mean Radial Error)")
        print("=" * 66)
        for rank, name in enumerate(report.ranking, start=1):
            res = report.results[name]
            print(f"{rank}. {name:<20s} mean {res.mean_error_cm:6.3f} cm | "
                  f"rms {res.rms_error_cm:6.3f} cm | max {res.max_error_cm:6.3f} cm")
        print("=" * 66)
    return 0


def cmd_train(args) -> int:
    from . import training as training_mod
    from .config import NetworkConfig, TrainingConfig

    net = NetworkConfig(n_neurons=args.neurons)
    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        log_every=args.log_every,
        seed=args.seed,
    )
    result = training_mod.distill(net, cfg, log_fn=lambda w, e, l: print(
        f"[train:{w}] epoch {e:03d} | loss {l:.6f}"
    ))
    print(f"final dense loss      : {result['final_loss']['dense']:.6f}")
    print(f"final connectome loss : {result['final_loss']['connectome']:.6f}")

    if args.save:
        training_mod.save_weights(
            args.save,
            dense=result["dense"],
            connectome=result["connectome"],
            training=result["config"],
        )
        print(f"saved weights -> {args.save}")
    return 0


def cmd_session(args) -> int:
    engine = _engine_from_args(args)
    sess = engine.session(args.controller, steps=args.steps, radius=args.radius, freq=args.freq)
    print(json.dumps(sess.describe(), indent=2))

    frames = args.frames
    for _ in range(frames):
        obs = sess.step()
        if args.verbose:
            print(
                f"t={obs['t']:5.2f}s step={obs['step']:3d} "
                f"p=({obs['state'][0]:+.4f},{obs['state'][1]:+.4f}) "
                f"tgt=({obs['target'][0]:+.4f},{obs['target'][1]:+.4f}) "
                f"tilt=({obs['tilt'][0]:+.4f},{obs['tilt'][1]:+.4f}) "
                f"err={obs['error_cm']:.3f}cm"
            )
    traj = sess.trajectory()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(traj, fh, indent=2)
        print(f"wrote {args.out}")
    else:
        print(f"steps={traj['steps']} mean_error_cm={traj['mean_error_cm']:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sim_engine",
        description="Ball-and-plate simulation engine (ANN2SNN benchmark).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="list available controllers")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("describe", help="dump engine config + controller catalogue")
    _add_engine_args(sp)
    sp.set_defaults(func=cmd_describe)

    sp = sub.add_parser("benchmark", help="run the closed-loop orbit benchmark")
    _add_engine_args(sp)
    sp.add_argument("--controllers", default=None, help="comma-separated controller names")
    sp.add_argument("--out", default=None, help="write full JSON report here")
    sp.add_argument("--no-trace", action="store_true", help="omit per-step traces from JSON")
    sp.add_argument("--quiet", action="store_true", help="suppress the summary table")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("train", help="behavioral distillation of the PD teacher")
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--lr", type=float, default=0.008)
    sp.add_argument("--neurons", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--log-every", type=int, default=25)
    sp.add_argument("--save", default=None, help="write a .pt weights bundle here")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("session", help="step an interactive session")
    _add_engine_args(sp)
    sp.add_argument("--controller", default="pid")
    sp.add_argument("--frames", type=int, default=10)
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--out", default=None, help="write the session trajectory JSON here")
    sp.set_defaults(func=cmd_session)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
