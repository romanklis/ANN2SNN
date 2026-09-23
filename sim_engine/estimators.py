"""State estimation: a linear Kalman filter for the double-integrator plants.

The controller never sees the true state. The camera measures **position only**,

    y_k = p_k + v_k,          v_k ~ N(0, R),      p in R^D

and the filter reconstructs the full state ``x̂ = [p, ṗ]`` from the measurement
stream and the commanded actions. This is the canonical separation between the
true state ``x``, the observation ``y`` and the estimate ``x̂``: the policy acts
on ``e = x̂ − r``, never on ``r − x``.

Nominal model (forward Euler, matching the plant step for a control gain ``G``):

    v_{k+1} = v_k + G·u_k·dt
    p_{k+1} = p_k + v_{k+1}·dt

``A``/``B``/``H`` are built for any position dimension ``D`` (``D = 2`` for the
ball-and-plate, ``D = 3`` for the drone). Damping, actuator gain/delay and other
body changes are deliberately *unmodelled* — the filter is suboptimal there,
which is the honest robustness signal.

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
    """Linear KF for ``[p, ṗ]`` from position-only measurements in ``D`` dims."""

    name = "kalman"
    description = "Linear Kalman filter (position-only camera measurement)"

    def __init__(
        self,
        dt: float = DT,
        *,
        pos_dim: int = 2,
        gain: float = -C_CONST,
        control_limit: float = MAX_TILT,
        process_noise: float = 0.1,
        meas_noise: float = 5e-3,
        delay: int = 0,
        init_pos_var: float = 1e-2,
        init_vel_var: float = 1e0,
        device="cpu",
        dtype=torch.float64,
    ) -> None:
        self.dt = float(dt)
        self.pos_dim = int(pos_dim)
        #: *signed* control -> acceleration gain: −C for the plate (a = −C·θ),
        #: +1 for the drone (a = u).
        self.gain = float(gain)
        self.control_limit = float(control_limit)
        self.process_noise = float(process_noise)
        self.meas_noise = float(meas_noise)
        self.delay = max(0, int(delay))
        self.device = torch.device(device)
        self.dtype = dtype

        D, dt, G = self.pos_dim, self.dt, self.gain
        n = 2 * D
        eye, zero = torch.eye(D, dtype=dtype, device=self.device), torch.zeros(
            D, D, dtype=dtype, device=self.device
        )
        # v' = v + G*u*dt ; p' = p + v'*dt
        self.A = torch.cat([
            torch.cat([eye, dt * eye], dim=1),
            torch.cat([zero, eye], dim=1),
        ], dim=0)
        self.B = torch.cat([G * dt * dt * eye, G * dt * eye], dim=0)   # (2D, D)
        self.H = torch.cat([eye, zero], dim=1)                          # (D, 2D)

        # Discrete white-noise acceleration model, one block per axis.
        q = self.process_noise ** 2
        qb = torch.tensor(
            [[dt ** 4 / 4.0, dt ** 3 / 2.0],
             [dt ** 3 / 2.0, dt ** 2.0]],
            dtype=dtype, device=self.device,
        ) * q
        self.Q = torch.zeros(n, n, dtype=dtype, device=self.device)
        for i in range(D):
            pos_i, vel_i = i, i + D
            self.Q[pos_i, pos_i], self.Q[pos_i, vel_i] = qb[0, 0], qb[0, 1]
            self.Q[vel_i, pos_i], self.Q[vel_i, vel_i] = qb[1, 0], qb[1, 1]

        self.R = torch.eye(D, dtype=dtype, device=self.device) * (self.meas_noise ** 2)
        self.P0 = torch.diag(torch.cat([
            torch.full((D,), init_pos_var, dtype=dtype, device=self.device),
            torch.full((D,), init_vel_var, dtype=dtype, device=self.device),
        ]))
        self.I = torch.eye(n, dtype=dtype, device=self.device)

        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, y0: Optional[torch.Tensor] = None) -> None:
        n = 2 * self.pos_dim
        self.x = torch.zeros(n, dtype=self.dtype, device=self.device)
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
        D = self.pos_dim
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(D)
        self.x = torch.zeros(2 * D, dtype=self.dtype, device=self.device)
        self.x[:D] = y
        self.P = self.P0.clone()

    # ----------------------------------------------------------------- update
    def update(self, y: torch.Tensor, u_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fold in the (possibly delayed) measurement ``y`` and return ``x̂``."""
        D = self.pos_dim
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(D)
        if not self._started:
            self._seed_with(y)
            self._started = True
            self._px.append(self.x.clone())
            self._pP.append(self.P.clone())
            self._cmds.append(torch.zeros(D, dtype=self.dtype, device=self.device))
            return self.x.clone()

        u = torch.zeros(D, dtype=self.dtype, device=self.device)
        if u_prev is not None:
            u = torch.as_tensor(u_prev, dtype=self.dtype, device=self.device).reshape(D)

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
        P = (self.I - K @ self.H) @ P_pred
        return x, P

    # -------------------------------------------------------------- reporting
    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "pos_dim": self.pos_dim,
            "dt": self.dt,
            "gain": self.gain,
            "process_noise": self.process_noise,
            "meas_noise": self.meas_noise,
            "delay": self.delay,
            "measurement": f"position-only [{', '.join('xyz'[:self.pos_dim])}]",
        }
