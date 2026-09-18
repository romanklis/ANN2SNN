"""Reference trajectories for the closed-loop benchmark.

The canonical task is a circular *orbit* the ball must follow:

    x_ref(t) = R cos(2 pi f t),   y_ref(t) = R sin(2 pi f t)

together with its analytic first and second derivatives (velocity and
acceleration feed-forward).  These are exactly the signals used by the
original prototype's "dynamic orbit tracking" benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch

from .physics import DT

__all__ = ["RefPoint", "Reference", "orbit_reference", "ReferenceTrajectory"]


@dataclass
class RefPoint:
    """A single reference sample: position, velocity and acceleration."""

    pos: torch.Tensor
    vel: torch.Tensor
    acc: torch.Tensor

    def to_dict(self):
        return {
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "vel": [float(self.vel[0]), float(self.vel[1])],
            "acc": [float(self.acc[0]), float(self.acc[1])],
        }


@dataclass
class Reference:
    """A whole reference trajectory (position/velocity/acceleration arrays)."""

    x: torch.Tensor
    y: torch.Tensor
    vx: torch.Tensor
    vy: torch.Tensor
    ax: torch.Tensor
    ay: torch.Tensor
    dt: float = DT
    radius: float = 0.15
    freq: float = 0.5
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def pos(self) -> torch.Tensor:
        """``(T, 2)`` stacked position array."""
        return torch.stack([self.x, self.y], dim=1)

    @property
    def vel(self) -> torch.Tensor:
        """``(T, 2)`` stacked velocity array."""
        return torch.stack([self.vx, self.vy], dim=1)

    @property
    def acc(self) -> torch.Tensor:
        """``(T, 2)`` stacked acceleration array."""
        return torch.stack([self.ax, self.ay], dim=1)

    def at(self, k: int) -> RefPoint:
        """Reference sample at index ``k`` (clamped to the trajectory)."""
        k = max(0, min(int(k), len(self) - 1))
        return RefPoint(
            pos=torch.stack([self.x[k], self.y[k]]),
            vel=torch.stack([self.vx[k], self.vy[k]]),
            acc=torch.stack([self.ax[k], self.ay[k]]),
        )

    def time_vector(self) -> torch.Tensor:
        return torch.arange(len(self), dtype=self.x.dtype, device=self.x.device) * self.dt

    def to_dict(self) -> dict:
        return {
            "kind": "orbit",
            "steps": len(self),
            "dt": self.dt,
            "radius": self.radius,
            "freq": self.freq,
            "pos": self.pos.tolist(),
            "vel": self.vel.tolist(),
            "acc": self.acc.tolist(),
            "meta": dict(self.meta),
        }


def orbit_reference(
    steps: int = 250,
    radius: float = 0.15,
    freq: float = 0.5,
    dt: float = DT,
    device="cpu",
    dtype=torch.float32,
) -> Reference:
    """Build the circular-orbit reference trajectory.

    Parameters
    ----------
    steps:
        Number of control frames ``T``.
    radius:
        Orbit radius ``R`` [m].
    freq:
        Orbit frequency ``f`` [Hz].
    dt:
        Timestep [s] (must match the physics timestep for the benchmark).
    """
    t_vec = np.linspace(0.0, (steps - 1) * dt, steps)
    w = 2.0 * np.pi * freq

    x = radius * np.cos(w * t_vec)
    y = radius * np.sin(w * t_vec)
    vx = -radius * w * np.sin(w * t_vec)
    vy = radius * w * np.cos(w * t_vec)
    ax = -radius * (w ** 2) * np.cos(w * t_vec)
    ay = -radius * (w ** 2) * np.sin(w * t_vec)

    to_t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
    return Reference(
        x=to_t(x), y=to_t(y),
        vx=to_t(vx), vy=to_t(vy),
        ax=to_t(ax), ay=to_t(ay),
        dt=dt, radius=radius, freq=freq,
        meta={"shape": "circle", "steps": steps},
    )


# Human-friendly alias used by the CLI / API.
ReferenceTrajectory = Reference
