"""Closed-loop trajectory benchmark.

Drives any :class:`~sim_engine.controllers.base.BaseController` through the
circular-orbit tracking task and reports the same *mean radial tracking error*
(in centimetres) that the original prototype printed, plus per-step traces that
a frontend can animate.

The loop is deliberately tiny and controller-agnostic::

    for k in range(steps):
        record(state)
        tilt = controller.act(state, reference.at(k))
        state = step_physics(state, tilt)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import BenchmarkConfig
from .controllers.base import BaseController
from .physics import DT, step_physics
from .reference import Reference, orbit_reference
from .serialization import to_jsonable

__all__ = [
    "TrajectoryResult",
    "BenchmarkReport",
    "run_closed_loop",
    "mean_radial_error_cm",
    "evaluate",
    "evaluate_many",
]


@dataclass
class TrajectoryResult:
    """Per-controller closed-loop trace + metrics."""

    name: str
    trajectory: np.ndarray            # (T, 4) plant states
    tilts: np.ndarray                 # (T, 2) applied actuator commands
    tracking_error: np.ndarray        # (T,) instantaneous radial error [m]
    mean_error_cm: float
    final_error_cm: float
    rms_error_cm: float
    max_error_cm: float
    settling_step: Optional[int] = None
    spikes: Optional[np.ndarray] = None   # (T, N)
    meta: dict = field(default_factory=dict)

    def to_dict(self, include_trace: bool = True) -> dict:
        d = {
            "name": self.name,
            "metrics": {
                "mean_error_cm": self.mean_error_cm,
                "rms_error_cm": self.rms_error_cm,
                "max_error_cm": self.max_error_cm,
                "final_error_cm": self.final_error_cm,
                "settling_step": self.settling_step,
            },
            "meta": dict(self.meta),
        }
        if include_trace:
            d["trajectory"] = self.trajectory.tolist()
            d["tilts"] = self.tilts.tolist()
            d["tracking_error"] = self.tracking_error.tolist()
        if self.spikes is not None:
            d["spikes"] = self.spikes.tolist()
        return to_jsonable(d)


@dataclass
class BenchmarkReport:
    """Aggregate report over several controllers on one reference trajectory."""

    reference: Reference
    results: Dict[str, TrajectoryResult]
    init_state: tuple
    config: dict

    @property
    def ranking(self) -> List[str]:
        """Controller names ordered from best (lowest mean error) to worst."""
        return sorted(self.results, key=lambda n: self.results[n].mean_error_cm)

    def to_dict(self, include_trace: bool = True) -> dict:
        return to_jsonable(
            {
                "reference": {
                    "kind": "orbit",
                    "steps": len(self.reference),
                    "dt": self.reference.dt,
                    "radius": self.reference.radius,
                    "freq": self.reference.freq,
                    "pos": self.reference.pos.tolist(),
                    "vel": self.reference.vel.tolist(),
                    "acc": self.reference.acc.tolist(),
                },
                "init_state": list(self.init_state),
                "config": self.config,
                "ranking": self.ranking,
                "results": {
                    name: res.to_dict(include_trace=include_trace)
                    for name, res in self.results.items()
                },
            }
        )


def mean_radial_error_cm(
    trajectory: np.ndarray,
    reference: Reference,
) -> np.ndarray:
    """Instantaneous radial tracking error ``|p - p_ref|`` in centimetres."""
    T = min(len(trajectory), len(reference))
    ref = reference.pos.cpu().numpy()[:T]
    return np.sqrt(np.sum((trajectory[:T, :2] - ref) ** 2, axis=1)) * 100.0


def run_closed_loop(
    controller: BaseController,
    reference: Optional[Reference] = None,
    init_state: Optional[torch.Tensor] = None,
    config: Optional[BenchmarkConfig] = None,
    *,
    record_spikes: Optional[bool] = None,
    name: Optional[str] = None,
) -> TrajectoryResult:
    """Roll the plant + controller forward over the whole reference trajectory."""
    config = config or BenchmarkConfig()
    if reference is None:
        reference = orbit_reference(
            steps=config.steps, radius=config.radius, freq=config.freq
        )
    steps = min(config.steps, len(reference))
    if record_spikes is None:
        record_spikes = config.record_spikes

    device = getattr(controller, "device", torch.device("cpu"))
    if init_state is None:
        init_state = torch.tensor([-0.05, 0.05, 0.0, 0.0], device=device)
    state = torch.as_tensor(init_state, dtype=torch.float32, device=device).clone()

    controller.reset()

    traj: List[np.ndarray] = []
    tilts: List[np.ndarray] = []
    spikes: List[np.ndarray] = []
    wants_spikes = bool(record_spikes and getattr(controller, "spiking", False))

    with torch.no_grad():
        for k in range(steps):
            traj.append(state.detach().cpu().numpy().copy())
            ref_k = reference.at(k)
            u = controller.act(state, ref_k)
            tilts.append(u.detach().cpu().numpy().copy())
            if wants_spikes:
                spk = controller.last_spikes()
                if spk is not None:
                    spikes.append(spk.detach().cpu().numpy().copy())
            state = step_physics(state, u, dt=reference.dt)

    trajectory = np.asarray(traj)
    tilt_arr = np.asarray(tilts)
    err = mean_radial_error_cm(trajectory, reference)

    # Settling: first index after which the error never again exceeds 2x its
    # steady-state (last-decile) median.
    steady = float(np.median(err[int(0.9 * len(err)):])) if len(err) else 0.0
    tol = max(2.0 * steady, 0.5)  # cm
    settling = None
    for i in range(len(err)):
        if np.all(err[i:] <= tol):
            settling = i
            break

    return TrajectoryResult(
        name=name or controller.name,
        trajectory=trajectory,
        tilts=tilt_arr,
        tracking_error=err,
        mean_error_cm=float(np.mean(err)) if len(err) else float("nan"),
        rms_error_cm=float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        max_error_cm=float(np.max(err)) if len(err) else float("nan"),
        final_error_cm=float(err[-1]) if len(err) else float("nan"),
        settling_step=settling,
        spikes=np.asarray(spikes) if spikes else None,
        meta={"controller": controller.describe()},
    )


def evaluate(
    controllers: Dict[str, BaseController],
    reference: Optional[Reference] = None,
    init_state: Optional[torch.Tensor] = None,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkReport:
    """Evaluate several named controllers on a **shared** reference trajectory."""
    config = config or BenchmarkConfig()
    if reference is None:
        reference = orbit_reference(
            steps=config.steps,
            radius=config.radius,
            freq=config.freq,
            device="cpu",
        )
    if init_state is None:
        init_state = torch.tensor([-0.05, 0.05, 0.0, 0.0])

    results: Dict[str, TrajectoryResult] = {}
    for name, ctrl in controllers.items():
        results[name] = run_closed_loop(
            ctrl, reference=reference, init_state=init_state, config=config, name=name
        )

    return BenchmarkReport(
        reference=reference,
        results=results,
        init_state=tuple(float(v) for v in init_state),
        config=config.to_dict(),
    )


def evaluate_many(*args, **kwargs) -> BenchmarkReport:  # pragma: no cover - alias
    """Alias for :func:`evaluate`, kept for call-site readability."""
    return evaluate(*args, **kwargs)
