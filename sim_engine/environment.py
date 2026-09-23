"""Embodied environment layer: sensing, actuation, body and perturbations.

The original benchmark handed controllers the exact plant state.  This module
inserts a realistic *body/environment* between the plant and the controller:

* **sensing** — a multi-channel suite (see :mod:`sim_engine.sensors`), each
  channel with its own Gaussian noise and delay,
* **actuation** — command delay, gain and bias,
* **body** — velocity damping and a scaled rolling gain ``C``,
* **perturbations** — continuous process noise and scheduled/random impulses.

It is deliberately deterministic: :meth:`EmbodiedEnv.reset` re-seeds a local RNG,
so every controller evaluated with the same seed faces the **same disturbance
realisation** (noise values, impulse timing and magnitudes).

The channel suite is supplied by the example.  Without one, a single
absolute-position camera is used — the historical behaviour, unchanged.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import EMBODIMENT_PRESETS, EmbodimentConfig
from .physics import C_CONST, DT, MAX_TILT, step_physics, step_point_mass
from .sensors import (
    FIX,
    ResolvedChannel,
    SensorModel,
    SensorReadings,
    camera_channel,
)

__all__ = ["EmbodiedEnv", "embodiment_specs"]


def _ball_step(state, command, *, dt, limit, gain, damping, disturbance):
    """Plate plant wrapper: signed ``gain`` → positive rolling constant."""
    return step_physics(
        state, command, dt=dt, max_tilt=limit, c_const=abs(gain),
        damping=damping, disturbance=disturbance,
    )


def _drone_step(state, command, *, dt, limit, gain, damping, disturbance):
    """Point-mass plant wrapper (the control gain is 1 by construction)."""
    return step_point_mass(
        state, command, dt=dt, max_accel=limit,
        damping=damping, disturbance=disturbance,
    )


def legacy_sensor_model(config: EmbodimentConfig, pos_dim: int) -> SensorModel:
    """The historical single absolute-position camera, resolved from *config*."""
    ch = camera_channel(pos_dim)
    return SensorModel(
        channels=(
            ResolvedChannel(
                kind=FIX,
                axes=ch.axes,
                sigma=float(config.sensor_noise_pos),
                latency=int(config.sensor_delay),
                description="camera y = p",
            ),
        ),
        imu=None,
        anchors=(),
        launch=(),
        pos_dim=int(pos_dim),
        description="position-only camera",
    )


def embodiment_specs() -> dict:
    """All named presets with their fully resolved configuration (for API/UI)."""
    return {
        name: EmbodimentConfig.from_preset(name).to_dict()
        for name in EMBODIMENT_PRESETS
    }


class EmbodiedEnv:
    """Observation + actuation + body model wrapped around the plant step."""

    def __init__(
        self,
        config: Optional[EmbodimentConfig] = None,
        *,
        dt: float = DT,
        max_tilt: float = MAX_TILT,
        device="cpu",
        dtype=torch.float32,
        init_state: Tuple[float, ...] = (-0.05, 0.05, 0.0, 0.0),
        pos_dim: int = 2,
        plant_gain: float = -C_CONST,
        step_fn=None,
        sensor: Optional[SensorModel] = None,
    ) -> None:
        self.config = config or EmbodimentConfig()
        self.dt = dt
        self.pos_dim = int(pos_dim)
        self.control_limit = float(max_tilt)
        self.max_tilt = self.control_limit        # legacy alias
        self.plant_gain = float(plant_gain)       # signed control -> accel gain
        self.step_fn = step_fn or _ball_step
        self.device = torch.device(device)
        self.dtype = dtype
        self.init_state = torch.as_tensor(init_state, dtype=dtype, device=self.device)
        self.sensor = sensor or legacy_sensor_model(self.config, self.pos_dim)

        self._rng = np.random.default_rng(self.config.seed)
        self._actuator_buf: deque = deque(maxlen=1)
        self._buffers: Dict[str, deque] = {}
        self._imu_sigma = 0.0
        self._last_accel = torch.zeros(self.pos_dim, dtype=dtype, device=self.device)
        self.sensor_delay = 0
        self.actuator_delay = 0
        self.c_scale = self.config.c_scale
        self.damping = self.config.damping
        self.process_noise = self.config.process_noise
        self.impulse_interval = self.config.impulse_interval
        self.impulse_std = self.config.impulse_std
        self.impulse_count = 0
        self.fix_count = 0
        self._k = 0
        self.last_disturbance = torch.zeros(self.pos_dim, dtype=dtype, device=self.device)
        self.last_kick = torch.zeros(self.pos_dim, dtype=dtype, device=self.device)
        self.last_impulse = False
        self.last_fix_anchor: Optional[torch.Tensor] = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the RNG, sample randomisation and clear the delay buffers."""
        cfg = self.config
        if seed is None:
            seed = cfg.seed
        self._rng = np.random.default_rng(seed)

        self.sensor_delay = int(cfg.sensor_delay)
        self.actuator_delay = int(cfg.actuator_delay)
        self.c_scale = float(cfg.c_scale)
        self.damping = float(cfg.damping)
        self.process_noise = float(cfg.process_noise)
        self.impulse_interval = int(cfg.impulse_interval)
        self.impulse_std = float(cfg.impulse_std)

        if cfg.randomize:
            lo, hi = cfg.c_scale_range
            self.c_scale = float(self._rng.uniform(lo, hi))
            lo, hi = cfg.damping_range
            self.damping = float(self._rng.uniform(lo, hi))

        # Per-channel delay lines, prefilled with the initial sample so the first
        # frames are not an artificial delayed transient.  A gated channel starts
        # empty (no fix has happened yet).
        d = self.pos_dim
        self._buffers = {}
        for ch in self.sensor.channels:
            n = int(ch.latency) + 1
            if ch.gate_range is not None:
                self._buffers[ch.kind] = deque([None] * n, maxlen=n)
            else:
                init = (self.init_state[list(ch.axes)].clone(), None)
                self._buffers[ch.kind] = deque([init] * n, maxlen=n)
        self._imu_sigma = float(self.sensor.imu.sigma) if self.sensor.has_imu else 0.0

        zero = torch.zeros(self.pos_dim, dtype=self.dtype, device=self.device)
        self._actuator_buf = deque(
            [zero.clone() for _ in range(self.actuator_delay + 1)],
            maxlen=self.actuator_delay + 1,
        )
        self._last_accel = zero.clone()
        self.impulse_count = 0
        self.fix_count = 0
        self._fix_active: set = set()
        self._k = 0
        self.last_disturbance = zero.clone()
        self.last_kick = zero.clone()
        self.last_impulse = False
        self.last_fix_new = False
        self.last_fix_anchor = None

    # --------------------------------------------------------------- sensing
    def sense(self, state: torch.Tensor) -> SensorReadings:
        """Sample every channel once for this frame.

        The accelerometer reports the acceleration the body *achieved* over the
        interval that just elapsed, so it is available as the filter's prediction
        input for the step into this frame (no additional latency).  Every other
        channel goes through its own delay line.
        """
        cfg = self.config
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        out = SensorReadings()
        self.last_fix_new = False

        if self.sensor.has_imu:
            imu = self._last_accel.clone()
            if self._imu_sigma:
                imu = imu + torch.as_tensor(
                    self._rng.normal(0.0, self._imu_sigma, size=self.pos_dim),
                    dtype=self.dtype, device=self.device,
                )
            out.imu = imu

        for ch in self.sensor.channels:
            current = self._sample(ch, state)
            buf = self._buffers[ch.kind]
            buf.append(current)
            delayed = buf[0]
            if delayed is None:
                continue
            value, anchor = delayed
            out.channels[ch.kind] = value.clone()
            if anchor is not None:
                out.anchors[ch.kind] = anchor.clone()
        return out

    def _sample(self, ch: ResolvedChannel, state: torch.Tensor):
        """One raw sample ``(value, anchor)`` for a channel, or ``None``."""
        cfg = self.config
        value = state[list(ch.axes)].clone()
        anchor = None
        if ch.gate_range is not None:
            chosen = self._visible_anchor(ch, state)
            if chosen is None:
                self._fix_active.discard(ch.kind)
                return None
            _, c = chosen
            anchor = c[list(ch.axes)].clone()
            value = value - anchor
            if ch.kind not in self._fix_active:
                # a new acquisition ("entered the checkpoint's visible range")
                self._fix_active.add(ch.kind)
                self.fix_count += 1
                self.last_fix_new = True
            self.last_fix_anchor = anchor
        if ch.sigma:
            value = value + torch.as_tensor(
                self._rng.normal(0.0, ch.sigma, size=ch.dim),
                dtype=self.dtype, device=self.device,
            )
        return value, anchor

    def _visible_anchor(self, ch: ResolvedChannel, state: torch.Tensor):
        """Nearest anchor inside the geometry gate, as ``(index, position)``."""
        p = state.detach()
        if ch.gate_min_z is not None and float(p[2]) < float(ch.gate_min_z):
            return None
        best = None
        for i, c in enumerate(self.sensor.anchors):
            c_t = torch.as_tensor(c, dtype=self.dtype, device=self.device)
            dist = float(torch.linalg.vector_norm(p[:len(c)] - c_t))
            if dist <= float(ch.gate_range) and (best is None or dist < best[0]):
                best = (dist, i, c_t)
        if best is None:
            return None
        return best[1], best[2]

    def measure(self, state: torch.Tensor) -> torch.Tensor:
        """Legacy accessor: the absolute-position camera sample of this frame.

        Performs one :meth:`sense` (so the delay lines advance exactly once) and
        returns the camera channel.  Examples without an absolute-position camera
        (e.g. the GPS-denied drone) raise: use :meth:`sense` there.
        """
        cam = self.sensor.camera_channel
        if cam is None:
            raise RuntimeError(
                "this example has no absolute-position camera channel; use sense()"
            )
        return self.sense(state).channels[cam.kind]

    # -------------------------------------------------------------- actuation
    def actuate(self, command: torch.Tensor) -> torch.Tensor:
        """Apply actuator gain/bias, clamp, and return the delayed command."""
        cfg = self.config
        cmd = command.detach().to(self.dtype).clone()
        if cfg.actuator_gain != 1.0:
            cmd = cmd * float(cfg.actuator_gain)
        if cfg.actuator_bias:
            cmd = cmd + float(cfg.actuator_bias)
        cmd = torch.clamp(cmd, -self.control_limit, self.control_limit)
        self._actuator_buf.append(cmd)
        return self._actuator_buf[0]

    # ------------------------------------------------------------- disturbance
    def _disturbance(self, k: int):
        """Return ``(accel, kick, impulse)``.

        ``accel`` is a continuous acceleration disturbance [m/s^2] (process noise);
        ``kick`` is a one-off **velocity** impulse [m/s] (``impulse_std``), applied
        directly to the body's velocity.
        """
        cfg = self.config
        accel = torch.zeros(self.pos_dim, dtype=self.dtype, device=self.device)
        kick = torch.zeros(self.pos_dim, dtype=self.dtype, device=self.device)
        impulse = False
        if self.process_noise:
            accel = accel + torch.as_tensor(
                self._rng.normal(0.0, self.process_noise, size=self.pos_dim),
                dtype=self.dtype, device=self.device,
            )
        if self.impulse_std:
            if self.impulse_interval and k > 0 and k % self.impulse_interval == 0:
                impulse = True
            elif cfg.impulse_prob and self._rng.random() < cfg.impulse_prob:
                impulse = True
            if impulse:
                kick = kick + torch.as_tensor(
                    self._rng.normal(0.0, self.impulse_std, size=self.pos_dim),
                    dtype=self.dtype, device=self.device,
                )
        return accel, kick, impulse

    # -------------------------------------------------------------------- step
    def step(self, state: torch.Tensor, command: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """Integrate one timestep with the body model and the current disturbance."""
        step_k = self._k if k is None else int(k)
        accel, kick, impulse = self._disturbance(step_k)
        # `last_disturbance` reports the equivalent acceleration (kick/dt) so the
        # disturbance RMS reflects both channels.
        self.last_kick = kick
        self.last_disturbance = accel + kick / self.dt
        self.last_impulse = impulse
        if impulse:
            self.impulse_count += 1
        d = self.pos_dim
        v_before = state[d:2 * d].clone()
        if float(kick.abs().sum()) > 0.0:
            state = state.clone()
            state[d:2 * d] = state[d:2 * d] + kick
        nxt = self.step_fn(
            state,
            command,
            dt=self.dt,
            limit=self.control_limit,
            gain=self.plant_gain * self.c_scale,
            damping=self.damping,
            disturbance=accel,
        )
        # The accelerometer would read the velocity change actually achieved,
        # including damping, actuator effects and velocity kicks.
        self._last_accel = (nxt[d:2 * d] - v_before) / self.dt
        self._k = step_k + 1
        return nxt

    # -------------------------------------------------------------- reporting
    def describe(self) -> dict:
        cfg = self.config
        return {
            "preset": cfg.preset,
            "enable": bool(cfg.enable),
            "clean": cfg.is_clean,
            "pos_dim": self.pos_dim,
            "sensor_noise_pos": cfg.sensor_noise_pos,
            "sensor_noise_scale": cfg.sensor_noise_scale,
            "sensor_delay": self.sensor_delay,
            "sensor": self.sensor.to_dict(),
            "fix_count": self.fix_count,
            "actuator_delay": self.actuator_delay,
            "actuator_gain": cfg.actuator_gain,
            "actuator_bias": cfg.actuator_bias,
            "damping": self.damping,
            "c_scale": self.c_scale,
            "process_noise": self.process_noise,
            "impulse_interval": self.impulse_interval,
            "impulse_std": self.impulse_std,
            "randomize": bool(cfg.randomize),
            "seed": cfg.seed,
            "impulse_count": self.impulse_count,
        }

    to_dict = describe
