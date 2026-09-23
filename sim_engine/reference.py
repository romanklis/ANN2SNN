"""Reference trajectories for the closed-loop benchmark.

Two families live here:

* **2-D** — the canonical circular *orbit* the ball must follow:
  ``x_ref(t) = R cos(2 pi f t)``, ``y_ref(t) = R sin(2 pi f t)`` with analytic
  first and second derivatives (velocity and acceleration feed-forward);
* **3-D** — the drone tasks: a *lissajous* flight path and a *setpoint* (hover
  with a step), again with analytic derivatives.

:class:`Reference` stores ``pos``/``vel``/``acc`` as ``(T, D)`` tensors, so the
same container serves both families; the legacy 2-D accessors (``x``, ``y``,
``vx`` …) are kept for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch

from .physics import DT

__all__ = [
    "RefPoint",
    "Reference",
    "orbit_reference",
    "lissajous_reference",
    "setpoint_reference",
    "ReferenceTrajectory",
]


@dataclass
class RefPoint:
    """A single reference sample: position, velocity and acceleration."""

    pos: torch.Tensor
    vel: torch.Tensor
    acc: torch.Tensor

    def to_dict(self):
        return {
            "pos": [float(v) for v in self.pos],
            "vel": [float(v) for v in self.vel],
            "acc": [float(v) for v in self.acc],
        }


@dataclass
class Reference:
    """A whole reference trajectory (``(T, D)`` position/velocity/acceleration)."""

    pos: torch.Tensor = field(default_factory=lambda: torch.empty(0, 2))
    vel: torch.Tensor = field(default_factory=lambda: torch.empty(0, 2))
    acc: torch.Tensor = field(default_factory=lambda: torch.empty(0, 2))
    dt: float = DT
    radius: Optional[float] = None
    freq: Optional[float] = None
    meta: dict = field(default_factory=dict)

    # -- dimensions --------------------------------------------------------- #
    @property
    def pos_dim(self) -> int:
        return int(self.pos.shape[1]) if self.pos.ndim == 2 else 0

    def __len__(self) -> int:
        return int(self.pos.shape[0])

    # -- sampling ----------------------------------------------------------- #
    def at(self, k: int) -> RefPoint:
        """Reference sample at index ``k`` (clamped to the trajectory)."""
        k = max(0, min(int(k), len(self) - 1))
        return RefPoint(pos=self.pos[k], vel=self.vel[k], acc=self.acc[k])

    def time_vector(self) -> torch.Tensor:
        return torch.arange(len(self), dtype=self.pos.dtype, device=self.pos.device) * self.dt

    # -- 2-D compatibility accessors ---------------------------------------- #
    @property
    def x(self) -> torch.Tensor:
        return self.pos[:, 0]

    @property
    def y(self) -> torch.Tensor:
        return self.pos[:, 1]

    @property
    def vx(self) -> torch.Tensor:
        return self.vel[:, 0]

    @property
    def vy(self) -> torch.Tensor:
        return self.vel[:, 1]

    @property
    def ax(self) -> torch.Tensor:
        return self.acc[:, 0]

    @property
    def ay(self) -> torch.Tensor:
        return self.acc[:, 1]

    @property
    def z(self) -> torch.Tensor:
        return self.pos[:, 2]

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "kind": self.meta.get("kind", "orbit"),
            "steps": len(self),
            "pos_dim": self.pos_dim,
            "dt": self.dt,
            "radius": self.radius,
            "freq": self.freq,
            "pos": self.pos.tolist(),
            "vel": self.vel.tolist(),
            "acc": self.acc.tolist(),
            "meta": dict(self.meta),
        }


def _as_reference(pos, vel, acc, dt, radius, freq, meta, device, dtype) -> Reference:
    to_t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
    return Reference(
        pos=to_t(pos), vel=to_t(vel), acc=to_t(acc),
        dt=dt, radius=radius, freq=freq, meta=meta,
    )


def orbit_reference(
    steps: int = 250,
    radius: float = 0.15,
    freq: float = 0.5,
    dt: float = DT,
    device="cpu",
    dtype=torch.float32,
) -> Reference:
    """Build the 2-D circular-orbit reference trajectory (the ball task)."""
    t_vec = np.linspace(0.0, (steps - 1) * dt, steps)
    w = 2.0 * np.pi * freq

    x = radius * np.cos(w * t_vec)
    y = radius * np.sin(w * t_vec)
    vx = -radius * w * np.sin(w * t_vec)
    vy = radius * w * np.cos(w * t_vec)
    ax = -radius * (w ** 2) * np.cos(w * t_vec)
    ay = -radius * (w ** 2) * np.sin(w * t_vec)

    pos = np.stack([x, y], axis=1)
    vel = np.stack([vx, vy], axis=1)
    acc = np.stack([ax, ay], axis=1)
    return _as_reference(
        pos, vel, acc, dt, radius, freq,
        {"kind": "orbit", "shape": "circle", "steps": steps}, device, dtype,
    )


def lissajous_reference(
    steps: int = 500,
    amplitude: float = 0.6,
    freq: float = 0.25,
    z_amplitude: float = 0.35,
    z_ratio: float = 2.0,
    dt: float = DT,
    device="cpu",
    dtype=torch.float32,
) -> Reference:
    """3-D lissajous flight path for the drone (analytic derivatives).

    ``x``/``y`` trace an ellipse at ``freq`` and ``z`` oscillates at
    ``z_ratio * freq`` with half the amplitude, giving a smooth closed 3-D path.
    ``radius`` is reported as the horizontal amplitude so the UI can frame it.
    """
    t_vec = np.linspace(0.0, (steps - 1) * dt, steps)
    w = 2.0 * np.pi * freq
    wz = 2.0 * np.pi * freq * z_ratio

    x = amplitude * np.cos(w * t_vec)
    y = amplitude * np.sin(w * t_vec)
    z = z_amplitude * 0.5 * (1.0 - np.cos(wz * t_vec))       # starts at rest

    vx = -amplitude * w * np.sin(w * t_vec)
    vy = amplitude * w * np.cos(w * t_vec)
    vz = z_amplitude * 0.5 * wz * np.sin(wz * t_vec)

    ax = -amplitude * w ** 2 * np.cos(w * t_vec)
    ay = -amplitude * w ** 2 * np.sin(w * t_vec)
    az = z_amplitude * 0.5 * wz ** 2 * np.cos(wz * t_vec)

    pos = np.stack([x, y, z], axis=1)
    vel = np.stack([vx, vy, vz], axis=1)
    acc = np.stack([ax, ay, az], axis=1)
    return _as_reference(
        pos, vel, acc, dt, amplitude, freq,
        {"kind": "lissajous", "shape": "3d", "steps": steps,
         "z_amplitude": z_amplitude, "z_ratio": z_ratio},
        device, dtype,
    )


def setpoint_reference(
    steps: int = 500,
    setpoint: Tuple[float, float, float] = (0.5, -0.3, 0.6),
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.6,
    dt: float = DT,
    device="cpu",
    dtype=torch.float32,
) -> Reference:
    """3-D hover task: a smooth (minimum-jerk) move to a fixed setpoint."""
    t_vec = np.linspace(0.0, (steps - 1) * dt, steps)
    T = max(t_vec[-1], 1e-6)
    s = np.clip(t_vec / T, 0.0, 1.0)
    # minimum-jerk profile 10s^3 - 15s^4 + 6s^5 and its derivatives
    ramp = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    ramp_v = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    ramp_a = (60 * s - 180 * s ** 2 + 120 * s ** 3) / (T ** 2)

    pos = np.stack([
        start[i] + (setpoint[i] - start[i]) * ramp for i in range(3)
    ], axis=1)
    vel = np.stack([
        (setpoint[i] - start[i]) * ramp_v for i in range(3)
    ], axis=1)
    acc = np.stack([
        (setpoint[i] - start[i]) * ramp_a for i in range(3)
    ], axis=1)
    return _as_reference(
        pos, vel, acc, dt, radius, None,
        {"kind": "setpoint", "shape": "3d", "steps": steps,
         "setpoint": list(setpoint)},
        device, dtype,
    )


# Human-friendly alias used by the CLI / API.
ReferenceTrajectory = Reference
