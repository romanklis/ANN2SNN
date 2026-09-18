"""Controller interface shared by every brain in the benchmark.

All controllers implement the same minimal contract::

    ctrl.reset()
    tilt = ctrl.act(state, ref_point)     # -> (2,) tensor, saturated

so the closed-loop benchmark loop in :mod:`sim_engine.benchmark` is completely
controller-agnostic and a *new* brain can be dropped in without touching the
simulation code.

Controllers that keep internal state (recurrent/SNN brains) expose ``reset``;
feed-forward controllers may implement it as a no-op.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..physics import N_IN, N_OUT, clamp_action

__all__ = ["BaseController", "ControllerError", "error_vector"]


class ControllerError(RuntimeError):
    """Raised when a controller is used incorrectly."""


def error_vector(state: torch.Tensor, ref) -> torch.Tensor:
    """Tracking error ``[ex, ey, evx, evy] = state - reference``.

    ``ref`` is a :class:`sim_engine.reference.RefPoint` (or any object exposing
    ``pos``/``vel`` attributes).
    """
    return torch.stack(
        [
            state[0] - ref.pos[0],
            state[1] - ref.pos[1],
            state[2] - ref.vel[0],
            state[3] - ref.vel[1],
        ]
    )


class BaseController:
    """Abstract base class for every ball-and-plate brain."""

    #: short machine name used by the registry / CLI / API
    name: str = "base"
    #: one-line human description surfaced to the frontend
    description: str = ""
    #: True when the controller carries hidden state across timesteps
    recurrent: bool = False
    #: True when the controller produces a spike raster
    spiking: bool = False
    #: True when the controller needs the reference acceleration feed-forward
    uses_feedforward: bool = False

    def __init__(
        self,
        n_in: int = N_IN,
        n_out: int = N_OUT,
        device="cpu",
        dt: float | None = None,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.device = torch.device(device)
        if dt is not None:
            self.dt = dt
        else:
            from ..physics import DT

            self.dt = DT

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Reset any internal (recurrent) state. No-op for feed-forward nets."""

    # -- control ------------------------------------------------------------ #
    def raw_act(self, state: torch.Tensor, ref) -> torch.Tensor:
        """Unsaturated actuator command; subclasses must implement this."""
        raise NotImplementedError

    def act(self, state: torch.Tensor, ref) -> torch.Tensor:
        """Return the saturating actuator command ``[theta_x, theta_y]``."""
        return clamp_action(self.raw_act(state, ref))

    # -- optional introspection -------------------------------------------- #
    def last_spikes(self) -> torch.Tensor | None:
        """Most recent spike vector, or ``None`` for non-spiking controllers."""
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "recurrent": self.recurrent,
            "spiking": self.spiking,
            "uses_feedforward": self.uses_feedforward,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "dt": self.dt,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
