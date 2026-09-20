"""Classical PD controller with acceleration feed-forward.

Implements the analytic baseline from the prototype::

    theta_x = kp * ex + kd * evx - ax_ref / C
    theta_y = kp * ey + kd * evy - ay_ref / C

where the gains are derived from a target closed-loop natural frequency
``omega_n`` and damping ratio ``zeta``::

    kp = omega_n^2 / C
    kd = 2 * zeta * omega_n / C
"""

from __future__ import annotations

import torch

from ..physics import C_CONST, MAX_TILT
from .base import BaseController, error_vector

__all__ = ["ClassicalPDController"]


class ClassicalPDController(BaseController):
    """Analytic PD + feed-forward controller (the benchmark reference)."""

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
    ) -> None:
        super().__init__(device=device)
        self.omega_n = omega_n
        self.zeta = zeta
        self.c_const = c_const
        self.max_tilt = max_tilt
        self.kp = (omega_n ** 2) / c_const
        self.kd = (2.0 * zeta * omega_n) / c_const
        self.use_feedforward = bool(use_feedforward)
        self.uses_feedforward = self.use_feedforward

    def raw_act(self, state: torch.Tensor, ref) -> torch.Tensor:
        err = error_vector(state, ref)
        # The feed-forward term cancels the reference acceleration so the PD
        # action only has to correct the *residual* tracking error.
        theta_x = self.kp * err[0] + self.kd * err[2]
        theta_y = self.kp * err[1] + self.kd * err[3]
        if self.use_feedforward:
            theta_x = theta_x - ref.acc[0] / self.c_const
            theta_y = theta_y - ref.acc[1] / self.c_const
        return torch.stack([theta_x, theta_y])

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
