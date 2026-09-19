"""Embodied environment layer: sensing, actuation, body and perturbations.

The original benchmark handed controllers the exact plant state.  This module
inserts a realistic *body/environment* between the plant and the controller:

* **sensing** — Gaussian sensor noise and a fixed observation delay,
* **actuation** — command delay, gain and bias,
* **body** — velocity damping and a scaled rolling gain ``C``,
* **perturbations** — continuous process noise and scheduled/random impulses.

It is deliberately deterministic: :meth:`EmbodiedEnv.reset` re-seeds a local RNG,
so every controller evaluated with the same seed faces the **same disturbance
realisation** (noise values, impulse timing and magnitudes).
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch

from .config import EMBODIMENT_PRESETS, EmbodimentConfig
from .physics import C_CONST, DT, MAX_TILT, step_physics

__all__ = ["EmbodiedEnv", "embodiment_specs"]


def embodiment_specs() -> dict:
    """All named presets with their fully resolved configuration (for API/UI)."""
    return {
        name: EmbodimentConfig.from_preset(name).to_dict()
        for name in EMBODIMENT_PRESETS
    }


class EmbodiedEnv:
    """Observation + actuation + body model wrapped around :func:`step_physics`."""

    def __init__(
        self,
        config: Optional[EmbodimentConfig] = None,
        *,
        dt: float = DT,
        max_tilt: float = MAX_TILT,
        device="cpu",
        dtype=torch.float32,
        init_state: Tuple[float, float, float, float] = (-0.05, 0.05, 0.0, 0.0),
    ) -> None:
        self.config = config or EmbodimentConfig()
        self.dt = dt
        self.max_tilt = max_tilt
        self.device = torch.device(device)
        self.dtype = dtype
        self.init_state = torch.as_tensor(init_state, dtype=dtype, device=self.device)

        self._rng = np.random.default_rng(self.config.seed)
        self._sensor_buf: deque = deque(maxlen=1)
        self._actuator_buf: deque = deque(maxlen=1)
        self.sensor_delay = 0
        self.actuator_delay = 0
        self.c_scale = self.config.c_scale
        self.damping = self.config.damping
        self.process_noise = self.config.process_noise
        self.impulse_interval = self.config.impulse_interval
        self.impulse_std = self.config.impulse_std
        self.impulse_count = 0
        self._k = 0
        self.last_disturbance = torch.zeros(2, dtype=dtype, device=self.device)
        self.last_impulse = False
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

        # Prefill both buffers with the initial state / zero command so the first
        # frames are not an artificial delayed transient.
        self._sensor_buf = deque(
            [self.init_state.clone() for _ in range(self.sensor_delay + 1)],
            maxlen=self.sensor_delay + 1,
        )
        zero = torch.zeros(2, dtype=self.dtype, device=self.device)
        self._actuator_buf = deque(
            [zero.clone() for _ in range(self.actuator_delay + 1)],
            maxlen=self.actuator_delay + 1,
        )
        self.impulse_count = 0
        self._k = 0
        self.last_disturbance = zero.clone()
        self.last_impulse = False

    # --------------------------------------------------------------- sensing
    def observe(self, state: torch.Tensor) -> torch.Tensor:
        """Return the (noisy, delayed) state the controller is allowed to see."""
        cfg = self.config
        obs = state.detach().to(self.dtype).clone()
        if cfg.sensor_noise_pos or cfg.sensor_noise_vel:
            std = np.array([
                cfg.sensor_noise_pos, cfg.sensor_noise_pos,
                cfg.sensor_noise_vel, cfg.sensor_noise_vel,
            ])
            noise = self._rng.normal(0.0, std)
            obs = obs + torch.as_tensor(noise, dtype=self.dtype, device=self.device)
        self._sensor_buf.append(obs)
        return self._sensor_buf[0]

    # -------------------------------------------------------------- actuation
    def actuate(self, command: torch.Tensor) -> torch.Tensor:
        """Apply actuator gain/bias, clamp, and return the delayed command."""
        cfg = self.config
        cmd = command.detach().to(self.dtype).clone()
        if cfg.actuator_gain != 1.0:
            cmd = cmd * float(cfg.actuator_gain)
        if cfg.actuator_bias:
            cmd = cmd + float(cfg.actuator_bias)
        cmd = torch.clamp(cmd, -self.max_tilt, self.max_tilt)
        self._actuator_buf.append(cmd)
        return self._actuator_buf[0]

    # ------------------------------------------------------------- disturbance
    def _disturbance(self, k: int):
        cfg = self.config
        d = torch.zeros(2, dtype=self.dtype, device=self.device)
        impulse = False
        if self.process_noise:
            d = d + torch.as_tensor(
                self._rng.normal(0.0, self.process_noise, size=2),
                dtype=self.dtype, device=self.device,
            )
        if self.impulse_std:
            if self.impulse_interval and k > 0 and k % self.impulse_interval == 0:
                impulse = True
            elif cfg.impulse_prob and self._rng.random() < cfg.impulse_prob:
                impulse = True
            if impulse:
                d = d + torch.as_tensor(
                    self._rng.normal(0.0, self.impulse_std, size=2),
                    dtype=self.dtype, device=self.device,
                )
        return d, impulse

    # -------------------------------------------------------------------- step
    def step(self, state: torch.Tensor, command: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """Integrate one timestep with the body model and the current disturbance."""
        step_k = self._k if k is None else int(k)
        d, impulse = self._disturbance(step_k)
        self.last_disturbance = d
        self.last_impulse = impulse
        if impulse:
            self.impulse_count += 1
        nxt = step_physics(
            state,
            command,
            dt=self.dt,
            max_tilt=self.max_tilt,
            c_const=C_CONST * self.c_scale,
            damping=self.damping,
            disturbance=d,
        )
        self._k = step_k + 1
        return nxt

    # -------------------------------------------------------------- reporting
    def describe(self) -> dict:
        cfg = self.config
        return {
            "preset": cfg.preset,
            "enable": bool(cfg.enable),
            "clean": cfg.is_clean,
            "sensor_noise_pos": cfg.sensor_noise_pos,
            "sensor_noise_vel": cfg.sensor_noise_vel,
            "sensor_delay": self.sensor_delay,
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
