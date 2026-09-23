"""Swappable **examples**: the plant + task + estimator model + metric + labels.

An example is everything that is specific to *what is being controlled*:

* the plant step (plate tilt → rolling ball; thrust vector → point-mass drone),
* the closed-loop task (a 2-D circle; a 3-D lissajous flight path or a hover),
* the nominal model the Kalman filter uses, and
* the metric, success bounds, labels/units and renderer hint used by the UI.

Everything else — the controller contract, closed-loop distillation, the
ANN→SNN transfer, sessions, the API — is example-agnostic.  Adding a new example
is one module plus :func:`register_example`.

Two built-ins ship today:

===========  ================================  ==================  ==================
name         plant                            command             task
===========  ================================  ==================  ==================
``ball``     ball rolling on a tilting plate  ``[θx, θy]`` rad    circular orbit
``drone``    3-D point mass                   ``[ax, ay, az]``    lissajous / hover
===========  ================================  ==================  ==================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .estimators import KalmanFilter
from .environment import EmbodiedEnv
from .physics import (
    C_CONST,
    DRONE_HALF,
    DT,
    MAX_THRUST,
    MAX_TILT,
    PLATE_HALF,
)
from .reference import Reference, lissajous_reference, orbit_reference, setpoint_reference

__all__ = [
    "ExampleSpec",
    "ExamplePlant",
    "register_example",
    "get_example",
    "list_examples",
    "example_names",
    "DEFAULT_EXAMPLE",
]

DEFAULT_EXAMPLE = "ball"


class ExamplePlant:
    """Stateful plant for interactive sessions, driven by an :class:`ExampleSpec`."""

    def __init__(self, spec: "ExampleSpec", *, dt: float = DT, device="cpu",
                 damping: float = 0.0, c_scale: float = 1.0) -> None:
        self.spec = spec
        self.dt = float(dt)
        self.device = torch.device(device)
        self.damping = float(damping)
        self.c_scale = float(c_scale)
        self.init_state = torch.as_tensor(spec.init_state, dtype=torch.float32, device=self.device)
        self.state = self.init_state.clone()
        self.t = 0

    def reset(self) -> torch.Tensor:
        self.state = self.init_state.clone()
        self.t = 0
        return self.state.clone()

    def step(self, command, disturbance=None) -> torch.Tensor:
        cmd = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        self.state = self.spec.step_fn(
            self.state, cmd,
            dt=self.dt, limit=self.spec.control_limit,
            gain=self.spec.plant_gain * self.c_scale,
            damping=self.damping, disturbance=disturbance,
        )
        self.t += 1
        return self.state.clone()

    @property
    def position(self) -> torch.Tensor:
        return self.state[:self.spec.pos_dim]


@dataclass(frozen=True)
class ExampleSpec:
    """Everything that distinguishes one controlled example from another."""

    name: str
    label: str
    pos_dim: int
    control_limit: float
    plant_gain: float          # signed control -> acceleration gain
    init_state: Tuple[float, ...]
    step_fn: Callable
    reference_fn: Callable
    bounds_low: Tuple[float, ...]
    bounds_high: Tuple[float, ...]
    units: Dict[str, str]
    labels: Dict[str, str]
    renderer: str
    defaults: Dict[str, Any] = field(default_factory=dict)

    # -- derived dimensions ------------------------------------------------- #
    @property
    def control_dim(self) -> int:
        return self.pos_dim

    @property
    def measurement_dim(self) -> int:
        return self.pos_dim

    @property
    def state_dim(self) -> int:
        return 2 * self.pos_dim

    @property
    def n_in(self) -> int:
        """Policy input width ``3 * pos_dim`` (error 2D + feed-forward D)."""
        return 3 * self.pos_dim

    @property
    def n_out(self) -> int:
        return self.pos_dim

    # -- task --------------------------------------------------------------- #
    def reference(self, steps: int = 500, dt: float = DT, device="cpu", **params) -> Reference:
        """Build this example's reference trajectory (``radius``/``freq`` honoured)."""
        merged = dict(self.defaults)
        merged.update({k: v for k, v in params.items() if v is not None})
        return self.reference_fn(steps=int(steps), dt=dt, device=device, **merged)

    def tracking_error_cm(self, trajectory: np.ndarray, reference: Reference) -> np.ndarray:
        """Per-frame tracking error ``‖p − r‖`` in centimetres."""
        d = self.pos_dim
        ref_pos = np.asarray(reference.pos)[:len(trajectory), :d]
        return np.linalg.norm(np.asarray(trajectory)[:, :d] - ref_pos, axis=1) * 100.0

    def in_bounds(self, trajectory: np.ndarray) -> np.ndarray:
        """Boolean mask: is the body still inside the allowed workspace?"""
        d = self.pos_dim
        pos = np.abs(np.asarray(trajectory)[:, :d])
        hi = np.asarray(self.bounds_high[:d])
        return np.all(pos <= hi, axis=1)

    # -- runtime factories -------------------------------------------------- #
    def make_env(self, embodiment, *, dt: float = DT, device="cpu") -> EmbodiedEnv:
        return EmbodiedEnv(
            embodiment, dt=dt, max_tilt=self.control_limit, device=device,
            init_state=self.init_state, pos_dim=self.pos_dim,
            plant_gain=self.plant_gain, step_fn=self.step_fn,
        )

    def make_plant(self, *, dt: float = DT, device="cpu",
                   damping: float = 0.0, c_scale: float = 1.0) -> ExamplePlant:
        """Stateful plant for interactive sessions (no sensing/actuator model)."""
        return ExamplePlant(self, dt=dt, device=device, damping=damping, c_scale=c_scale)

    def make_estimator(self, embodiment, *, dt: float = DT, device="cpu") -> KalmanFilter:
        return KalmanFilter(
            dt=dt,
            pos_dim=self.pos_dim,
            gain=self.plant_gain,
            control_limit=self.control_limit,
            process_noise=embodiment.estimate_process_noise,
            meas_noise=max(float(embodiment.sensor_noise_pos), 1e-4),
            delay=int(embodiment.sensor_delay),
            init_pos_var=embodiment.estimate_init_pos_var,
            init_vel_var=embodiment.estimate_init_vel_var,
            device=device,
        )

    # -- reporting ---------------------------------------------------------- #
    def describe(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "pos_dim": self.pos_dim,
            "control_dim": self.control_dim,
            "measurement_dim": self.measurement_dim,
            "state_dim": self.state_dim,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "control_limit": self.control_limit,
            "plant_gain": self.plant_gain,
            "init_state": list(self.init_state),
            "bounds_low": list(self.bounds_low),
            "bounds_high": list(self.bounds_high),
            "units": dict(self.units),
            "labels": dict(self.labels),
            "renderer": self.renderer,
            "defaults": dict(self.defaults),
        }


# --------------------------------------------------------------------------- #
# Built-in examples
# --------------------------------------------------------------------------- #
def _ball_step(state, command, *, dt, limit, gain, damping, disturbance):
    from .physics import step_physics

    return step_physics(state, command, dt=dt, max_tilt=limit,
                        c_const=abs(gain), damping=damping, disturbance=disturbance)


def _drone_step(state, command, *, dt, limit, gain, damping, disturbance):
    from .physics import step_point_mass

    return step_point_mass(state, command, dt=dt, max_accel=limit,
                           damping=damping, disturbance=disturbance)


def _ball_reference(*, steps, dt, device, radius=0.15, freq=0.5, **_) -> Reference:
    return orbit_reference(steps=steps, radius=radius, freq=freq, dt=dt, device=device)


def _drone_reference(*, steps, dt, device, radius=0.6, freq=0.25,
                     z_amplitude=0.35, z_ratio=2.0, **_):
    return lissajous_reference(steps=steps, amplitude=radius, freq=freq,
                               z_amplitude=z_amplitude, z_ratio=z_ratio,
                               dt=dt, device=device)


def _drone_hover_reference(*, steps, dt, device, setpoint=(0.5, -0.3, 0.6), **_):
    return setpoint_reference(steps=steps, setpoint=setpoint, dt=dt, device=device)


BALL = ExampleSpec(
    name="ball",
    label="Balancing ball",
    pos_dim=2,
    control_limit=MAX_TILT,
    plant_gain=-C_CONST,
    init_state=(-0.05, 0.05, 0.0, 0.0),
    step_fn=_ball_step,
    reference_fn=_ball_reference,
    bounds_low=(-PLATE_HALF, -PLATE_HALF),
    bounds_high=(PLATE_HALF, PLATE_HALF),
    units={"command": "rad", "error": "cm", "position": "m", "velocity": "m/s"},
    labels={"plant": "BALL + PLATE", "command": "PLATE TILT",
            "error": "RADIAL ERROR", "success": "ON PLATE"},
    renderer="plate",
    defaults={"radius": 0.15, "freq": 0.5},
)

DRONE = ExampleSpec(
    name="drone",
    label="Hovering drone",
    pos_dim=3,
    control_limit=MAX_THRUST,
    plant_gain=1.0,
    init_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    step_fn=_drone_step,
    reference_fn=_drone_reference,
    bounds_low=(-DRONE_HALF, -DRONE_HALF, -DRONE_HALF),
    bounds_high=(DRONE_HALF, DRONE_HALF, DRONE_HALF),
    units={"command": "m/s²", "error": "cm", "position": "m", "velocity": "m/s"},
    labels={"plant": "3-D DRONE", "command": "THRUST",
            "error": "TRACKING ERROR", "success": "IN CORRIDOR"},
    renderer="quad",
    defaults={"radius": 0.6, "freq": 0.25},
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, ExampleSpec] = {}


def register_example(spec: ExampleSpec) -> ExampleSpec:
    """Register (or replace) an example by name."""
    _REGISTRY[spec.name] = spec
    return spec


def get_example(name: Optional[str] = None) -> ExampleSpec:
    """Look up an example, defaulting to :data:`DEFAULT_EXAMPLE`."""
    key = (name or DEFAULT_EXAMPLE).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown example {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def example_names() -> List[str]:
    return list(_REGISTRY)


def list_examples() -> List[dict]:
    """Catalogue for the CLI / API / UI."""
    return [spec.describe() for spec in _REGISTRY.values()]


register_example(BALL)
register_example(DRONE)
