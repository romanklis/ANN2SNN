#!/usr/bin/env python3
"""End-to-end example: run the four-brain closed-loop benchmark.

    python examples/run_benchmark.py                 # uses untrained brains
    python examples/run_benchmark.py --train         # distils first (~1 min CPU)
    python examples/run_benchmark.py --plot out.png  # also save a figure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_engine.api import build_engine
from sim_engine.registry import CANONICAL_CONTROLLERS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", default=",".join(CANONICAL_CONTROLLERS[:4]))
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--train", action="store_true", help="distil the ANNs first")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--json", default=None, help="write full report JSON here")
    ap.add_argument("--plot", default=None, help="write a PNG figure here")
    args = ap.parse_args()

    names = [n.strip() for n in args.controllers.split(",") if n.strip()]
    engine = build_engine(
        {"steps": args.steps, "epochs": args.epochs, "train_on_init": args.train},
        train=args.train,
    )

    t0 = time.time()
    report = engine.run_benchmark(names, include_trace=True, as_dict=False)
    dt = time.time() - t0

    print("=" * 70)
    print(f"CLOSED-LOOP ORBIT BENCHMARK  ({args.steps} frames, {engine.device})")
    print("=" * 70)
    for rank, name in enumerate(report.ranking, start=1):
        r = report.results[name]
        print(
            f"{rank}. {name:<18s} mean {r.mean_error_cm:7.3f} cm | "
            f"rms {r.rms_error_cm:7.3f} cm | max {r.max_error_cm:7.3f} cm"
        )
    print("=" * 70)
    print(f"wall time: {dt:.2f}s")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report.to_dict(include_trace=True), fh, indent=2)
        print(f"wrote {args.json}")

    if args.plot:
        _plot(report, args.plot)
        print(f"wrote {args.plot}")
    return 0


def _plot(report, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ref = report.reference.pos.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(ref[:, 0], ref[:, 1], "k--", lw=2, label="Target orbit")
    for name, res in report.results.items():
        axes[0].plot(res.trajectory[:, 0], res.trajectory[:, 1], lw=1.8,
                     label=f"{name} ({res.mean_error_cm:.2f} cm)")
    axes[0].set_title("Ball trajectory on the tilting plate")
    axes[0].set_xlabel("X position (m)")
    axes[0].set_ylabel("Y position (m)")
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    t = np.arange(len(report.results[report.ranking[0]].tracking_error)) * report.reference.dt
    for name, res in report.results.items():
        axes[1].plot(t, res.tracking_error, lw=1.6, label=f"{name} ({res.mean_error_cm:.2f} cm)")
    axes[1].set_title("Radial tracking error vs time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (cm)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    raise SystemExit(main())
