"""Classical PD controller with acceleration feed-forward (any dimension).

Implements the analytic baseline from the prototype, generalised to ``D`` axes::

    u_i = kp * e_i + kd * ė_i + a_ref_i / G_signed

where ``G_signed`` is the *signed* control→acceleration gain (``−C`` for the
plate, ``+1`` for the drone), and the gains come from a target closed-loop
natural frequency and damping ratio::

    kp = omega_n^2 / |G|
    kd = 2 * zeta * omega_n / |G|

For the plate this is exactly ``theta = kp*e + kd*ė − a_ref/C``.
"""

from __future__ import annotations

import torch

from ..physics import C_CONST, MAX_TILT
from .base import BaseController, error_vector

__all__ = ["ClassicalPDController"]


class ClassicalPDController(BaseController):
    """Analytic PD + feed-forward controller (the benchmark reference / teacher)."""

    name = "pid"
    description = "Classical PD with acceleration feed-forward"
    recurrent = False
    spiking = False
    uses_feedforward = True

    def __init__(
        self,
        omega_n: float = 3.5,
        zeta: float = 0.85,
        c_const: float = C_CONST,
        max_tilt: float = MAX_TILT,
        use_feedforward: bool = True,
        device="cpu",
        pos_dim: int = 2,
        n_out: int | None = None,
        plant_gain: float | None = None,
        action_limit: float | None = None,
    ) -> None:
        gain = float(plant_gain) if plant_gain is not None else -abs(float(c_const))
        limit = float(action_limit) if action_limit is not None else float(max_tilt)
        dim = int(n_out) if n_out is not None else int(pos_dim)
        super().__init__(
            n_in=3 * dim, n_out=dim, device=device,
            action_limit=limit, plant_gain=gain, pos_dim=dim,
        )
        self.omega_n = omega_n
        self.zeta = zeta
        self.c_const = abs(gain)
        mag = abs(gain)
        self.kp = (omega_n ** 2) / mag
        self.kd = (2.0 * zeta * omega_n) / mag
        self.use_feedforward = bool(use_feedforward)
        self.uses_feedforward = self.use_feedforward

    def raw_act(self, state: torch.Tensor, ref) -> torch.Tensor:
        d = self.pos_dim
        err = error_vector(state, ref)
        # Desired acceleration: corrective feedback (+ reference feed-forward).
        w2 = self.omega_n ** 2
        z2w = 2.0 * self.zeta * self.omega_n
        accel = -(w2 * err[:d] + z2w * err[d:2 * d])
        if self.use_feedforward:
            accel = accel + ref.acc[:d]
        # The command is the acceleration divided by the *signed* plant gain.
        return accel / self.plant_gain

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            {
                "gains": {"kp": self.kp, "kd": self.kd},
                "omega_n": self.omega_n,
                "zeta": self.zeta,
            }
        )
        return d
