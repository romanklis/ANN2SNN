"""State estimation: a linear Kalman filter for the double-integrator plants.

The controller never sees the true state.  Each example declares *what its
sensors measure* (see :mod:`sim_engine.sensors`):

* a **position camera** ``y = p + v`` (the ball, and the plain drone), and
* a **multi-channel suite** (the GPS-denied drone): an accelerometer used as the
  filter's *prediction input*, plus altitude, optical-flow velocity and
  geometry-gated checkpoint fixes.

Whichever suite is active, the filter reconstructs ``x̂ = [p, ṗ]`` from the
samples and the applied inputs.  The canonical separation between the true state
``x``, the observations ``y`` and the estimate ``x̂`` holds: the policy acts on
``e = x̂ − r``, never on ``r − x``.

Nominal model (forward Euler, matching the plant step for a control gain ``G``):

    v_{k+1} = v_k + G·u_k·dt          (or + a_meas·dt with an accelerometer)
    p_{k+1} = p_k + v_{k+1}·dt

``A``/``H`` are built for any position dimension ``D``.  Damping, actuator
gain/delay and other body changes are deliberately *unmodelled* — the filter is
suboptimal there, which is the honest robustness signal.  The one exception is
the accelerometer: because it measures the acceleration the body *actually*
achieved, feeding it into the prediction is what makes an actuator gain/bias an
estimation problem rather than a pure penalty.

Sensor delay is handled properly and **per channel**: a measurement ``y_k``
refers to ``x_{k−d_c}`` for its channel's latency ``d_c``, so the filter keeps a
window of estimates, applies every correction at its own time index (oldest
first) and re-propagates forward to the current step.  A late-arriving fix is
therefore folded in at the right index even when a faster channel has already
been processed beyond it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .physics import C_CONST, DT, MAX_TILT
from .sensors import (
    FIX,
    NOISE_FLOOR,
    ResolvedChannel,
    SensorModel,
    SensorReadings,
    selection_matrix,
)

__all__ = ["KalmanFilter"]


def _legacy_sensor(pos_dim: int, sigma: float, delay: int) -> SensorModel:
    """The historical single position-camera suite (all position axes)."""
    return SensorModel(
        channels=(
            ResolvedChannel(
                kind=FIX,
                axes=tuple(range(int(pos_dim))),
                sigma=float(sigma),
                latency=int(delay),
                description="position-only camera",
            ),
        ),
        imu=None,
        anchors=(),
        launch=(),
        pos_dim=int(pos_dim),
        description="position-only camera",
    )


class KalmanFilter:
    """Linear KF for ``[p, ṗ]`` from one or more sensor channels.

    Passing ``sensor=...`` uses that suite (multi-channel).  Without it the
    historical constructor arguments ``meas_noise`` / ``delay`` describe a single
    absolute-position channel, so existing callers keep their exact behaviour.
    """

    name = "kalman"

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
        sensor: Optional[SensorModel] = None,
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
        self.sensor = sensor or _legacy_sensor(self.pos_dim, self.meas_noise, self.delay)
        # A single absolute-position camera is the historical configuration, even
        # when the example declares it explicitly: it must report identically.
        self._legacy = bool(
            self.sensor.imu is None
            and len(self.sensor.channels) == 1
            and self.sensor.camera_channel is not None
        )

        self.description = (
            "Linear Kalman filter (position-only camera measurement)"
            if self._legacy
            else "Linear Kalman filter (multi-channel onboard sensing)"
        )

        D, dt = self.pos_dim, self.dt
        n = 2 * D
        eye, zero = torch.eye(D, dtype=dtype, device=self.device), torch.zeros(
            D, D, dtype=dtype, device=self.device
        )
        # v' = v + G*u*dt ; p' = p + v'*dt
        self.A = torch.cat([
            torch.cat([eye, dt * eye], dim=1),
            torch.cat([zero, eye], dim=1),
        ], dim=0)
        #: commanded control -> state (gain-scaled), the historical input model.
        self.B_cmd = torch.cat([self.gain * dt * dt * eye, self.gain * dt * eye], dim=0)
        #: measured acceleration -> state (unit gain): the IMU prediction input.
        self.B_acc = torch.cat([dt * dt * eye, dt * eye], dim=0)

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

        # Per-channel measurement models: H selects the observed state axes, R is
        # the noise floor (kept invertible even for a perfectly clean sensor).
        self._H: Dict[str, torch.Tensor] = {}
        self._R: Dict[str, torch.Tensor] = {}
        self._latency: Dict[str, int] = {}
        self._gated: Dict[str, bool] = {}
        for ch in self.sensor.channels:
            self._register(ch)

        self.reset()

    # ------------------------------------------------------------------ setup
    def _register(self, channel: ResolvedChannel) -> None:
        n = 2 * self.pos_dim
        if channel.kind == FIX and any(int(a) >= self.pos_dim for a in channel.axes):
            raise ValueError("position_fix channels must observe position axes only")
        H = torch.tensor(
            selection_matrix(channel.axes, n), dtype=self.dtype, device=self.device
        )
        sigma = max(float(channel.sigma), NOISE_FLOOR)
        R = torch.eye(channel.dim, dtype=self.dtype, device=self.device) * (sigma ** 2)
        self._H[channel.kind] = H
        self._R[channel.kind] = R
        self._latency[channel.kind] = int(channel.latency)
        self._gated[channel.kind] = channel.gate_range is not None

    def reset(self, y0: Optional[torch.Tensor] = None) -> None:
        """Clear the estimate.  Seeded from ``y0``, else the launch point, else
        deferred until the first absolute-position measurement."""
        n = 2 * self.pos_dim
        self.x = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.P = self.P0.clone()
        self._k = -1                       # newest estimate index (−1 = none)
        self._x: List[torch.Tensor] = []
        self._P: List[torch.Tensor] = []
        self._bu: List[torch.Tensor] = []
        #: every measurement ever folded in, keyed by the index it refers to.
        #: Kept (not consumed) so a late-arriving sample can trigger a replay that
        #: re-applies all corrections after its index.
        self._meas: Dict[int, list] = {}
        #: per-index estimator internals for the extended dashboard, kept alongside
        #: the measurement log: ``_innov[j][kind] = z − H·x̂`` (recorded at the index
        #: the sample refers to) and ``_cov_diag[j] = diag(P_j)`` (posterior).
        self._innov: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cov_diag: Dict[int, torch.Tensor] = {}
        self._last_innovation = torch.zeros(0, dtype=self.dtype, device=self.device)
        self._started = False
        #: True right after a *launch-point* seed: index 0 is the pre-loop state,
        #: so the first ``update`` must fold samples into index 0 instead of
        #: predicting index 1.  Keeps the filter index == the control frame for
        #: both seeding paths (the deferred camera seed already consumes frame 0).
        self._pending_launch = False
        self._seed_x = self.x.clone()
        self._seed_P = self.P.clone()
        if y0 is not None:
            self._seed(y0)
        elif self.sensor.launch:
            self._seed(torch.tensor(self.sensor.launch, dtype=self.dtype, device=self.device))
            self._pending_launch = True

    def _seed(self, z: torch.Tensor) -> None:
        """Anchor the estimate at index 0 from a launch point or first fix."""
        D = self.pos_dim
        z = torch.as_tensor(z, dtype=self.dtype, device=self.device).reshape(-1)
        self.x = torch.zeros(2 * D, dtype=self.dtype, device=self.device)
        self.x[:D] = z[:D]
        if z.numel() >= 2 * D:
            self.x[D:2 * D] = z[D:2 * D]
        self.P = self.P0.clone()
        self._k = 0
        self._x = [self.x.clone()]
        self._P = [self.P.clone()]
        self._bu = [torch.zeros(2 * D, dtype=self.dtype, device=self.device)]
        self._seed_x = self.x.clone()
        self._seed_P = self.P.clone()
        self._cov_diag[0] = torch.diagonal(self.P).clone()
        self._started = True

    # ----------------------------------------------------------------- update
    def update(
        self,
        y: Optional[torch.Tensor] = None,
        u_prev: Optional[torch.Tensor] = None,
        *,
        readings: Optional[SensorReadings] = None,
        control: Optional[torch.Tensor] = None,
        acceleration: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fold in one frame of sensing and return the estimate ``x̂``.

        Legacy call form: ``update(y, u_prev)`` — one absolute-position sample and
        the previously commanded control.  Rich form: ``update(readings=...,
        control=..., acceleration=...)`` where ``acceleration`` is the measured
        body acceleration used for the prediction step.
        """
        if readings is None:
            readings = SensorReadings()
            if y is not None:
                kind = self.sensor.camera_channel
                key = kind.kind if kind is not None else FIX
                readings.channels[key] = torch.as_tensor(
                    y, dtype=self.dtype, device=self.device
                ).reshape(-1)
            if control is None:
                control = u_prev

        if acceleration is not None:
            acceleration = torch.as_tensor(
                acceleration, dtype=self.dtype, device=self.device
            ).reshape(-1)
        if control is not None:
            control = torch.as_tensor(
                control, dtype=self.dtype, device=self.device
            ).reshape(-1)

        # ---- no estimate yet: wait for something that anchors the position
        if not self._started:
            cam = self.sensor.camera_channel
            if cam is None or cam.kind not in readings.channels:
                return self.x.clone()
            self._seed(readings.channels[cam.kind])
            return self.x.clone()

        if self._pending_launch:
            # Index 0 is the launch point (the state at frame 0), so this frame's
            # samples correct index 0 rather than predicting a new one.
            self._pending_launch = False
            if self._log_samples(readings, 0):
                self._replay(0, 0)
            self.x, self.P = self._x[self._k], self._P[self._k]
            return self.x.clone()

        D = self.pos_dim
        k = self._k + 1
        bu = self._predict_input(control, acceleration)
        x_pred = self.A @ self._x[self._k] + bu
        P_pred = self.A @ self._P[self._k] @ self.A.T + self.Q
        self._k = k
        self._x.append(x_pred)
        self._P.append(P_pred)
        self._bu.append(bu)
        # Record the propagated covariance for this index; a replay below
        # overwrites it with the posterior if a correction lands here, so every
        # index always has a value even when no sample arrives (high latency).
        self._cov_diag[k] = torch.diagonal(P_pred).clone()

        # ---- replay from the oldest *new* index: this re-applies every stored
        #      correction after it, so a late fix cannot discard earlier ones.
        new_srcs = self._log_samples(readings, k)
        if new_srcs:
            self._replay(min(new_srcs), k)

        self.x, self.P = self._x[self._k], self._P[self._k]
        return self.x.clone()

    def _log_samples(self, readings: SensorReadings, k: int) -> List[int]:
        """Log every sample at the index it refers to; return the new indices.

        The measurement log is never consumed, so a late-arriving sample can
        trigger a replay that re-applies every correction after its index.
        """
        new_srcs: List[int] = []
        for kind, value in readings.channels.items():
            if kind not in self._H:
                continue
            src = int(k) - self._latency[kind]
            if src < 0:
                continue
            anchor = readings.anchors.get(kind)
            if anchor is not None:
                anchor = torch.as_tensor(
                    anchor, dtype=self.dtype, device=self.device
                ).reshape(-1)
            self._meas.setdefault(src, []).append((kind, value, anchor))
            new_srcs.append(src)
        return new_srcs

    def _predict_input(
        self, control: Optional[torch.Tensor], acceleration: Optional[torch.Tensor]
    ) -> torch.Tensor:
        D = self.pos_dim
        if acceleration is not None and acceleration.numel() == D:
            return self.B_acc @ acceleration
        if control is not None and control.numel() == D:
            return self.B_cmd @ control
        return torch.zeros(2 * D, dtype=self.dtype, device=self.device)

    def _replay(self, j0: int, k: int) -> None:
        """Recompute indices ``j0..k``, re-applying every stored correction."""
        j0 = max(0, int(j0))
        if j0 == 0:
            x, P = self._seed_x.clone(), self._seed_P.clone()
        else:
            x, P = self._x[j0 - 1], self._P[j0 - 1]
        for j in range(j0, k + 1):
            if j > 0:
                x = self.A @ x + self._bu[j]
                P = self.A @ P @ self.A.T + self.Q
            for kind, value, anchor in self._meas.get(j, ()):
                x, P = self._apply(kind, value, anchor, x, P)
                self._innov.setdefault(j, {})[kind] = self._last_innovation
            self._x[j], self._P[j] = x, P
            self._cov_diag[j] = torch.diagonal(P).clone()

    def _apply(
        self,
        kind: str,
        value: torch.Tensor,
        anchor: Optional[torch.Tensor],
        x: torch.Tensor,
        P: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linear correction for one sample (adds the anchor for a relative fix)."""
        H, R = self._H[kind], self._R[kind]
        z = torch.as_tensor(value, dtype=self.dtype, device=self.device).reshape(-1)
        if anchor is not None:
            z = z + anchor
        S = H @ P @ H.T + R
        K = P @ H.T @ torch.linalg.inv(S)
        innovation = z - H @ x
        #: last innovation ``z − H·x̂`` seen by :meth:`_apply` (for the trace).
        self._last_innovation = innovation.clone()
        x_new = x + K @ innovation
        P_new = (self.I - K @ H) @ P
        return x_new, P_new

    # ------------------------------------------------------------- internals
    def innovation(self, index: int, kind: str) -> Optional[torch.Tensor]:
        """Innovation ``z − H·x̂`` folded in at ``index`` for ``kind``, if any.

        Recorded against the index the sample *refers to* (not the frame it was
        applied in), so a channel with latency reports its residual for the state
        it actually measured.
        """
        rec = self._innov.get(int(index))
        return None if rec is None else rec.get(kind)

    def covariance_diag(self, index: int) -> Optional[torch.Tensor]:
        """Posterior ``diag(P)`` at ``index``, if the filter has reached it."""
        return self._cov_diag.get(int(index))

    @property
    def fix_frames(self) -> List[int]:
        """Indices where a geometry-gated checkpoint fix was folded in."""
        return sorted(
            j for j, items in self._meas.items()
            if any(k == FIX and self._gated.get(k, False) for k, _, _ in items)
        )

    # -------------------------------------------------------------- reporting
    def describe(self) -> dict:
        d = {
            "name": self.name,
            "description": self.description,
            "pos_dim": self.pos_dim,
            "dt": self.dt,
            "gain": self.gain,
            "process_noise": self.process_noise,
            "meas_noise": self.meas_noise,
            "delay": self.delay,
            "seeded_from_launch": bool(self.sensor.launch),
            "multi_channel": not self._legacy,
        }
        if self._legacy:
            d["measurement"] = f"position-only [{', '.join('xyz'[:self.pos_dim])}]"
        else:
            d["measurement"] = ", ".join(
                f"{ch.kind}[{ch.dim}]@{ch.latency}f" for ch in self.sensor.channels
            )
            d["sensor"] = self.sensor.to_dict()
        d["fix_frames"] = self.fix_frames
        d["measurement_counts"] = {
            kind: sum(1 for items in self._meas.values() for k, _, _ in items if k == kind)
            for kind in sorted(self._H)
        }
        return d
