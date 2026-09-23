"""Controller interface shared by every brain in the benchmark.

All controllers implement the same minimal contract::

    ctrl.reset()
    u = ctrl.act(estimate, ref_point)     # -> (D,) tensor, saturated

so the closed-loop benchmark loop in :mod:`sim_engine.benchmark` is completely
controller-agnostic and a *new* brain can be dropped in without touching the
simulation code.  ``D`` is the example's control dimension (2 for the plate,
3 for the drone).

Controllers that keep internal state (recurrent/SNN brains) expose ``reset``;
feed-forward controllers may implement it as a no-op.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..physics import C_CONST, DT, MAX_TILT, N_IN, N_OUT

__all__ = ["BaseController", "ControllerError", "error_vector", "ReferenceAccelEstimator"]


class ControllerError(RuntimeError):
    """Raised when a controller is used incorrectly."""


def error_vector(state: torch.Tensor, ref) -> torch.Tensor:
    """Tracking error ``[p_error, v_error] = state − reference`` (dimension ``2D``).

    ``ref`` is a :class:`sim_engine.reference.RefPoint` (or any object exposing
    ``pos``/``vel``).  This is the sign the plants need: to push a body that sits
    at ``x > r`` back, the command is a positive function of ``x − r``.
    """
    d = int(ref.pos.shape[0])
    return torch.cat([state[:d] - ref.pos, state[d:2 * d] - ref.vel])


class ReferenceAccelEstimator:
    """Reference acceleration reconstructed from the Kalman estimate.

    The controller receives the estimated state ``x̂`` and the reference point, so
    the reference itself is recoverable as ``r̂ = x̂ − e`` with
    ``e = error_vector(x̂, r)``.  A three-point second difference of ``r̂`` gives the
    reference acceleration ``â_ref`` (equal to ``ref.acc`` for a smooth reference),
    and the feed-forward command follows as ``u_ff = −â_ref / gain`` with the
    known control gain ``gain`` (the rolling constant ``C`` for the plate, ``1``
    for the drone).  No ground-truth state or oracle acceleration is used.
    """

    def __init__(self, dt: float = DT, plant_gain: float = -C_CONST, pos_dim: int = 2) -> None:
        self.dt = float(dt)
        #: *signed* control -> acceleration gain: −C for the plate (a = −C·θ),
        #: +1 for the drone (a = u).  Feed-forward is ``u_ff = â_ref / gain``.
        self.gain = float(plant_gain)
        self.pos_dim = int(pos_dim)
        self._hist: list = []

    def reset(self) -> None:
        self._hist = []

    def update(self, state: torch.Tensor, ref) -> torch.Tensor:
        """Feed-forward command ``u_ff`` ``(D,)`` for the current ``(x̂, ref)``."""
        d = self.pos_dim
        err = error_vector(state, ref)
        r_hat = state - err                     # = ref, reconstructed from x̂
        self._hist.append(r_hat)
        if len(self._hist) > 3:
            self._hist.pop(0)
        if len(self._hist) == 3:
            # second difference of the reference *position* (D-dim)
            a_hat = (
                self._hist[2][:d] - 2.0 * self._hist[1][:d] + self._hist[0][:d]
            ) / (self.dt ** 2)
        else:
            a_hat = torch.zeros(d, dtype=r_hat.dtype, device=r_hat.device)
        return a_hat / self.gain


class BaseController:
    """Abstract base class for every brain (plate or drone)."""

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
        *,
        action_limit: float = MAX_TILT,
        plant_gain: float = -C_CONST,
        pos_dim: int | None = None,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.device = torch.device(device)
        self.action_limit = float(action_limit)
        #: *signed* control -> acceleration gain (see ReferenceAccelEstimator)
        self.plant_gain = float(plant_gain)
        # control dimension: n_out == pos_dim for every current example
        self.pos_dim = int(pos_dim) if pos_dim is not None else int(n_out)
        self.dt = dt if dt is not None else DT
        self._ff = ReferenceAccelEstimator(self.dt, self.plant_gain, self.pos_dim)

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Reset any internal (recurrent) state. No-op for feed-forward nets."""
        self._ff.reset()

    # -- policy input ------------------------------------------------------- #
    def policy_input(self, state: torch.Tensor, ref) -> torch.Tensor:
        """Policy input ``[error (2D), u_ff (D)]``.

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
        """Return the saturated actuator command ``(D,)``."""
        return torch.clamp(self.raw_act(state, ref), -self.action_limit, self.action_limit)

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
            "pos_dim": self.pos_dim,
            "action_limit": self.action_limit,
            "plant_gain": self.plant_gain,
            "dt": self.dt,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
