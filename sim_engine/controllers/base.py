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

from ..physics import C_CONST, DT, N_IN, N_OUT, clamp_action

__all__ = ["BaseController", "ControllerError", "error_vector", "ReferenceAccelEstimator"]


class ControllerError(RuntimeError):
    """Raised when a controller is used incorrectly."""


def error_vector(state: torch.Tensor, ref) -> torch.Tensor:
    """Tracking error ``[ex, ey, evx, evy] = state - reference``.

    ``ref`` is a :class:`sim_engine.reference.RefPoint` (or any object exposing
    ``pos``/``vel`` attributes).  This is the sign the plant needs: to push a ball
    that sits at ``x > r`` back, the plate must tilt positively, i.e. the command
    is a positive function of ``x - r``.
    """
    return torch.stack(
        [
            state[0] - ref.pos[0],
            state[1] - ref.pos[1],
            state[2] - ref.vel[0],
            state[3] - ref.vel[1],
        ]
    )


class ReferenceAccelEstimator:
    """Reference acceleration reconstructed from the Kalman estimate.

    The controller receives the estimated state ``x̂`` and the reference point, so
    the reference itself is recoverable as ``r̂ = x̂ − e`` with
    ``e = error_vector(x̂, r)``.  A three-point second difference of ``r̂`` gives the
    reference acceleration ``â_ref`` (equal to ``ref.acc`` for a smooth reference),
    and the feed-forward command follows as ``u_ff = −â_ref / C`` with the known
    rolling gain ``C`` (ball mass).  No ground-truth state or oracle acceleration
    is used anywhere.
    """

    def __init__(self, dt: float = DT, c_const: float = C_CONST) -> None:
        self.dt = float(dt)
        self.c_const = float(c_const)
        self._hist: list = []

    def reset(self) -> None:
        self._hist = []

    def update(self, state: torch.Tensor, ref) -> torch.Tensor:
        """Feed-forward command ``u_ff`` (2,) for the current ``(x̂, ref)``."""
        err = error_vector(state, ref)
        r_hat = state - err                     # = ref, reconstructed from x̂
        self._hist.append(r_hat)
        if len(self._hist) > 3:
            self._hist.pop(0)
        if len(self._hist) == 3:
            # second difference of the reference *position* (2-D)
            a_hat = (
                self._hist[2][:2] - 2.0 * self._hist[1][:2] + self._hist[0][:2]
            ) / (self.dt ** 2)
        else:
            a_hat = torch.zeros(2, dtype=r_hat.dtype, device=r_hat.device)
        return -a_hat / self.c_const


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
        self._ff = ReferenceAccelEstimator(self.dt)

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Reset any internal (recurrent) state. No-op for feed-forward nets."""
        self._ff.reset()

    # -- policy input ------------------------------------------------------- #
    def policy_input(self, state: torch.Tensor, ref) -> torch.Tensor:
        """Policy input ``[ex, ey, evx, evy, uff_x, uff_y]``.

        ``state`` is the Kalman estimate ``x̂`` (never the true state); the
        feed-forward term is reconstructed from it (see
        :class:`ReferenceAccelEstimator`).
        """
        err = error_vector(state, ref)
        u_ff = self._ff.update(state, ref)
        return torch.cat([err, u_ff])

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
