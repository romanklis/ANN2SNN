"""Robustness sweeps: tracking error as a function of environmental difficulty.

Each cell is one closed-loop evaluation with a single environmental parameter
varied (a preset, sensor noise, sensor delay or impulse magnitude), so the
result is a robustness curve for every controller under a **shared** disturbance
realisation per cell.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .benchmark import evaluate
from .config import EMBODIMENT_PRESETS, BenchmarkConfig, EmbodimentConfig

__all__ = ["sweep", "build_env_config", "DEFAULT_AXIS_POINTS", "multi_seed"]

#: Default sweep grids (bounded so a sweep stays fast and cache-friendly).
DEFAULT_AXIS_POINTS: Dict[str, list] = {
    "preset": list(EMBODIMENT_PRESETS),
    "noise": [0.0, 0.002, 0.005, 0.01, 0.02],
    "delay": [0, 1, 2, 3, 5, 8],
    "impulse": [0.0, 0.05, 0.1, 0.2, 0.4],
}

#: Hard caps so an API caller cannot ask for an unbounded sweep.
MAX_POINTS = 12
MAX_CONTROLLERS = 5


def build_env_config(axis: str, point, base: Optional[EmbodimentConfig] = None) -> EmbodimentConfig:
    """Embodiment config for one sweep cell."""
    base = base or EmbodimentConfig()
    if axis == "preset":
        return EmbodimentConfig.from_preset(str(point))
    cfg = EmbodimentConfig(preset=axis, seed=base.seed)
    if axis == "noise":
        cfg.sensor_noise_pos = float(point)
    elif axis == "delay":
        d = int(point)
        cfg.sensor_delay = d
        cfg.actuator_delay = max(0, d - 1)
    elif axis == "impulse":
        cfg.impulse_interval = 60
        cfg.impulse_std = float(point)
    else:
        raise ValueError(
            f"unknown sweep axis {axis!r}; expected one of {sorted(DEFAULT_AXIS_POINTS)}"
        )
    return cfg


def sweep(
    build_controllers: Callable[[], Dict[str, object]],
    *,
    config: Optional[BenchmarkConfig] = None,
    axis: str = "preset",
    points: Optional[Sequence] = None,
    seed: Optional[int] = None,
) -> dict:
    """Evaluate ``build_controllers()`` across one environmental axis."""
    base = config or BenchmarkConfig()
    if axis not in DEFAULT_AXIS_POINTS:
        raise ValueError(f"unknown sweep axis {axis!r}")
    if points is None:
        points = DEFAULT_AXIS_POINTS[axis]
    points = list(points)[:MAX_POINTS]

    controllers = build_controllers()
    if len(controllers) > MAX_CONTROLLERS:
        raise ValueError(f"sweep supports at most {MAX_CONTROLLERS} controllers")

    if seed is not None:
        base = dataclasses.replace(base, embodiment=dataclasses.replace(
            base.embodiment, seed=seed
        ))

    cells: List[dict] = []
    for point in points:
        embodiment = build_env_config(axis, point, base.embodiment)
        cfg = dataclasses.replace(base, embodiment=embodiment)
        report = evaluate(controllers, config=cfg)
        per_controller = {
            name: {
                "mean_error_cm": res.mean_error_cm,
                "rms_error_cm": res.rms_error_cm,
                "max_error_cm": res.max_error_cm,
                "on_plate_pct": res.on_plate_pct,
                "impulse_count": res.impulse_count,
            }
            for name, res in report.results.items()
        }
        cells.append({
            "point": point,
            "embodiment": embodiment.to_dict(),
            "per_controller": per_controller,
        })

    return {
        "axis": axis,
        "points": points,
        "steps": base.steps,
        "radius": base.radius,
        "freq": base.freq,
        "controllers": list(controllers),
        "cells": cells,
    }


def multi_seed(
    build_controllers: Callable[[int], Dict[str, object]],
    *,
    seeds: Sequence[int] = (42, 1, 2, 3, 4),
    config: Optional[BenchmarkConfig] = None,
) -> dict:
    """Evaluate every controller across seeds and return mean ± std.

    ``build_controllers(seed)`` must return a fresh controller dict for that seed
    (different initialisations / disturbance realisations).  This is the evidence
    behind claims like "the ANN→SNN transfer is essentially lossless" — a point
    estimate from one seed is not enough.
    """
    base = config or BenchmarkConfig()
    values: Dict[str, List[float]] = {}
    for seed in seeds:
        report = evaluate(build_controllers(int(seed)), config=base)
        for name, res in report.results.items():
            values.setdefault(name, []).append(float(res.mean_error_cm))

    summary = {
        name: {
            "mean_error_cm": float(np.mean(v)),
            "std_error_cm": float(np.std(v)),
            "min_error_cm": float(np.min(v)),
            "max_error_cm": float(np.max(v)),
            "n": len(v),
            "values": v,
        }
        for name, v in values.items()
    }
    return {"seeds": [int(s) for s in seeds], "summary": summary}
