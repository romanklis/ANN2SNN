"""State estimation: a linear Kalman filter for the ball-and-plate plant.

The controller never sees the true state. The camera measures **position only**,

    y_k = [x_k, y_k] + v_k,          v_k ~ N(0, R),

and the filter reconstructs the full state

    x̂ = [x, y, vx, vy]

from the measurement stream and the commanded tilts. This is the canonical
separation between the true state ``x``, the observation ``y`` and the estimate
``x̂``: the policy acts on ``e = r − x̂``, not on ``r − x``.

Nominal model (forward Euler, matching :func:`sim_engine.physics.step_physics`):

    v_{k+1} = v_k − C·θ_k·dt
    p_{k+1} = p_k + v_{k+1}·dt

so ``A``/``B`` below are exact for the ideal (undamped, unit-gain, undelayed)
plant. Damping, actuator gain/delay and other body changes are deliberately
*unmodelled* — the filter is suboptimal there, which is the honest robustness
signal.

Sensor delay is handled properly: a measurement ``y_k`` refers to ``x_{k−d}``, so
the filter keeps a short history of priors, applies the correction at the
measurement's time index, then re-propagates forward to the current step.
"""

from __future__ import annotations

from typing import Optional

import torch

from .physics import C_CONST, DT, MAX_TILT

__all__ = ["KalmanFilter"]


class KalmanFilter:
    """Linear KF for ``[x, y, vx, vy]`` from position-only measurements."""

    name = "kalman"
    description = "Linear Kalman filter (position-only camera measurement)"

    def __init__(
        self,
        dt: float = DT,
        *,
        c_const: float = C_CONST,
        max_tilt: float = MAX_TILT,
        process_noise: float = 0.1,
        meas_noise: float = 5e-3,
        delay: int = 0,
        init_pos_var: float = 1e-2,
        init_vel_var: float = 1e0,
        device="cpu",
        dtype=torch.float64,
    ) -> None:
        self.dt = float(dt)
        self.c_const = float(c_const)
        self.max_tilt = float(max_tilt)
        self.process_noise = float(process_noise)
        self.meas_noise = float(meas_noise)
        self.delay = max(0, int(delay))
        self.device = torch.device(device)
        self.dtype = dtype

        dt, C = self.dt, self.c_const
        # v' = v - C*theta*dt ; p' = p + v'*dt
        self.A = torch.tensor(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=dtype, device=self.device,
        )
        self.B = torch.tensor(
            [[-C * dt * dt, 0],
             [0, -C * dt * dt],
             [-C * dt, 0],
             [0, -C * dt]],
            dtype=dtype, device=self.device,
        )
        self.H = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=dtype, device=self.device,
        )
        # Discrete white-noise acceleration model; per-axis [position, velocity]
        # block for x -> indices (0, 2) and y -> indices (1, 3).
        q = self.process_noise ** 2
        qb = torch.tensor(
            [[dt ** 4 / 4.0, dt ** 3 / 2.0],
             [dt ** 3 / 2.0, dt ** 2.0]],
            dtype=dtype, device=self.device,
        ) * q
        self.Q = torch.zeros(4, 4, dtype=dtype, device=self.device)
        for pos_i, vel_i in ((0, 2), (1, 3)):
            self.Q[pos_i, pos_i] = qb[0, 0]
            self.Q[pos_i, vel_i] = qb[0, 1]
            self.Q[vel_i, pos_i] = qb[1, 0]
            self.Q[vel_i, vel_i] = qb[1, 1]

        self.R = torch.eye(2, dtype=dtype, device=self.device) * (self.meas_noise ** 2)
        self.P0 = torch.diag(torch.tensor(
            [init_pos_var, init_pos_var, init_vel_var, init_vel_var],
            dtype=dtype, device=self.device,
        ))
        self.I4 = torch.eye(4, dtype=dtype, device=self.device)

        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, y0: Optional[torch.Tensor] = None) -> None:
        self.x = torch.zeros(4, dtype=self.dtype, device=self.device)
        self.P = self.P0.clone()
        self._started = False
        self._step = 0
        # per-step priors and the commands that produced them (for delay)
        self._px: list = []
        self._pP: list = []
        self._cmds: list = []
        if y0 is not None:
            self._seed_with(y0)

    def _seed_with(self, y: torch.Tensor) -> None:
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(2)
        self.x = torch.zeros(4, dtype=self.dtype, device=self.device)
        self.x[0], self.x[1] = y[0], y[1]
        self.P = self.P0.clone()

    # ----------------------------------------------------------------- update
    def update(self, y: torch.Tensor, u_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fold in the (possibly delayed) measurement ``y`` and return ``x̂``."""
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(2)
        if not self._started:
            self._seed_with(y)
            self._started = True
            self._px.append(self.x.clone())
            self._pP.append(self.P.clone())
            self._cmds.append(torch.zeros(2, dtype=self.dtype, device=self.device))
            return self.x.clone()

        u = torch.zeros(2, dtype=self.dtype, device=self.device)
        if u_prev is not None:
            u = torch.as_tensor(u_prev, dtype=self.dtype, device=self.device).reshape(2)

        # predict current step
        x_pred = self.A @ self.x + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q
        self._step += 1
        self._px.append(x_pred.clone())
        self._pP.append(P_pred.clone())
        self._cmds.append(u.clone())

        src = self._step - self.delay          # index the measurement refers to
        if src < 0:
            self.x, self.P = x_pred, P_pred
            return self.x.clone()

        # correct the prior at `src`, then re-propagate forward to now
        self._px[src], self._pP[src] = self._correct(self._px[src], self._pP[src], y)
        for j in range(src + 1, self._step + 1):
            xj = self.A @ self._px[j - 1] + self.B @ self._cmds[j]
            Pj = self.A @ self._pP[j - 1] @ self.A.T + self.Q
            self._px[j], self._pP[j] = xj, Pj
        self.x, self.P = self._px[self._step], self._pP[self._step]
        return self.x.clone()

    def _correct(self, x_pred: torch.Tensor, P_pred: torch.Tensor, y: torch.Tensor):
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ torch.linalg.inv(S)
        x = x_pred + K @ (y - self.H @ x_pred)
        P = (self.I4 - K @ self.H) @ P_pred
        return x, P

    # -------------------------------------------------------------- reporting
    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "dt": self.dt,
            "process_noise": self.process_noise,
            "meas_noise": self.meas_noise,
            "delay": self.delay,
            "measurement": "position-only [x, y]",
        }
