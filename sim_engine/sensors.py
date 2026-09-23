"""Onboard sensor models: *what* each example's estimator actually measures.

Until now every example had exactly one sensor: a position camera
``y = p + v``.  That is fine for the ball on a plate, and unrealistic for a
drone: absolute position is generally unavailable indoors, and the quantities a
real vehicle *can* measure are different in kind.

This module makes the sensor suite a first-class, example-owned object:

* :class:`SensorChannel` — one channel (kind, observed state axes, nominal noise,
  latency, optional geometry gate);
* :class:`SensorSpec` — an ordered suite plus the world geometry it refers to
  (checkpoint anchors, launch point);
* :class:`ResolvedChannel` / :class:`SensorModel` — the same suite with the
  *effective* noise and latency after the embodiment config has been applied
  (``sensor_noise_scale`` multiplies nominal noise, ``sensor_delay`` adds frames);
* :class:`SensorReadings` — one frame of samples handed to the state estimator.

Four kinds are supported:

===============  ===============================================================
kind             measurement
===============  ===============================================================
``imu``          the body acceleration achieved over the last step.  This is a
                 function of the *input* (and disturbances), not of the state,
                 so it is used as the filter's **prediction input** rather than
                 as a measurement update.
``altitude``     absolute height along one axis (barometer / rangefinder).
``flow_velocity`` horizontal velocity (optical flow x altitude, the standard
                 downward-camera model).
``position_fix`` relative position ``p - c`` to a known anchor ("checkpoint"),
                 only when the geometry gate is satisfied.  Modelling the fix as
                 *relative position* (range **and** bearing) rather than a bare
                 range keeps the filter linear; a range-only fix would need an
                 EKF and is deliberately out of scope.
===============  ===============================================================

The suite is intentionally torch-free: it is pure data/geometry, so the CLI and
tests can inspect it without importing the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "IMU",
    "ALTITUDE",
    "FLOW",
    "FIX",
    "CHANNEL_KINDS",
    "SensorChannel",
    "SensorSpec",
    "ResolvedChannel",
    "SensorModel",
    "selection_matrix",
    "camera_channel",
    "imu_channel",
    "altitude_channel",
    "flow_channel",
    "checkpoint_channel",
]

IMU = "imu"
ALTITUDE = "altitude"
FLOW = "flow_velocity"
FIX = "position_fix"

CHANNEL_KINDS: Tuple[str, ...] = (IMU, ALTITUDE, FLOW, FIX)

#: Measurement-noise floor the filter assumes, so ``R`` stays invertible for a
#: perfectly clean sensor.  Matches the historical ``max(sigma, 1e-4)``.
NOISE_FLOOR: float = 1e-4


def selection_matrix(axes: Sequence[int], total_dim: int) -> List[List[float]]:
    """``(len(axes), total_dim)`` 0/1 row-selection matrix (as nested lists)."""
    rows: List[List[float]] = []
    for a in axes:
        if not (0 <= int(a) < int(total_dim)):
            raise ValueError(f"axis {a!r} outside 0..{int(total_dim) - 1}")
        row = [0.0] * int(total_dim)
        row[int(a)] = 1.0
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Nominal suite (example-owned)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensorChannel:
    """One sensing channel in an example's nominal suite.

    ``axes`` are **absolute state indices** (``0..pos_dim-1`` for position,
    ``pos_dim..2*pos_dim-1`` for velocity), so the same descriptor works for a
    2-D plate and a 3-D drone.
    """

    kind: str
    axes: Tuple[int, ...]
    base_noise: float = 0.0
    latency: int = 0
    #: True for the historical single position camera: its noise/latency come
    #: from ``EmbodimentConfig.sensor_noise_pos`` / ``sensor_delay`` instead of
    #: ``base_noise`` / ``latency``.  Keeps the ball's numbers bit-for-bit.
    noise_from_config: bool = False
    #: ``position_fix`` only: only produce a sample when ``‖p - c‖ <= gate_range``
    #: and (optionally) ``p[2] >= gate_min_z``.
    gate_range: Optional[float] = None
    gate_min_z: Optional[float] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CHANNEL_KINDS:
            raise ValueError(
                f"unknown sensor kind {self.kind!r}; expected one of {list(CHANNEL_KINDS)}"
            )
        if not self.axes:
            raise ValueError(f"{self.kind}: needs at least one observed axis")
        if len(set(int(a) for a in self.axes)) != len(self.axes):
            raise ValueError(f"{self.kind}: duplicate axes {self.axes!r}")
        if any(int(a) < 0 for a in self.axes):
            raise ValueError(f"{self.kind}: negative axis in {self.axes!r}")
        if float(self.base_noise) < 0.0:
            raise ValueError(f"{self.kind}: base_noise must be >= 0")
        if int(self.latency) < 0:
            raise ValueError(f"{self.kind}: latency must be >= 0")
        if self.kind == IMU and (self.gate_range is not None or self.gate_min_z is not None):
            raise ValueError("imu cannot be geometry-gated")
        if self.kind == FIX and self.gate_range is not None and float(self.gate_range) <= 0.0:
            raise ValueError("position_fix: gate_range must be > 0")

    @property
    def dim(self) -> int:
        return len(self.axes)

    @property
    def is_prediction(self) -> bool:
        """True when the channel feeds the filter's prediction, not an update."""
        return self.kind == IMU

    @property
    def gated(self) -> bool:
        return self.gate_range is not None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "axes": [int(a) for a in self.axes],
            "dim": self.dim,
            "base_noise": float(self.base_noise),
            "latency": int(self.latency),
            "noise_from_config": bool(self.noise_from_config),
            "gated": self.gated,
            "gate_range": None if self.gate_range is None else float(self.gate_range),
            "gate_min_z": None if self.gate_min_z is None else float(self.gate_min_z),
            "description": self.description,
        }


@dataclass(frozen=True)
class SensorSpec:
    """An example's complete nominal sensor suite plus its world geometry."""

    channels: Tuple[SensorChannel, ...]
    anchors: Tuple[Tuple[float, ...], ...] = ()
    launch: Tuple[float, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("a sensor spec needs at least one channel")
        kinds = [c.kind for c in self.channels]
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"one channel per kind; got {kinds!r}")
        if len([c for c in self.channels if c.is_prediction]) > 1:
            raise ValueError("at most one prediction channel (imu)")
        if any(c.kind == FIX and c.gate_range is not None for c in self.channels) \
                and not self.anchors:
            raise ValueError("gated position_fix channels need at least one anchor")
        dims = {len(a) for a in self.anchors}
        if len(dims) > 1:
            raise ValueError("all anchors must share a dimension")

    # -- lookup ------------------------------------------------------------- #
    @property
    def prediction_channel(self) -> Optional[SensorChannel]:
        for c in self.channels:
            if c.is_prediction:
                return c
        return None

    @property
    def correction_channels(self) -> Tuple[SensorChannel, ...]:
        return tuple(c for c in self.channels if not c.is_prediction)

    @property
    def fix_channel(self) -> Optional[SensorChannel]:
        for c in self.channels:
            if c.kind == FIX:
                return c
        return None

    # -- validation / resolution -------------------------------------------- #
    def validate(self, pos_dim: int) -> None:
        """Check axis indices against the example's ``pos_dim``."""
        total = 2 * int(pos_dim)
        for c in self.channels:
            for a in c.axes:
                if not (0 <= int(a) < total):
                    raise ValueError(
                        f"{c.kind}: axis {a} outside 0..{total - 1} for pos_dim={pos_dim}"
                    )
        for c in self.channels:
            if c.kind in (ALTITUDE, FLOW) and any(int(a) >= total for a in c.axes):
                raise ValueError(f"{c.kind}: axis out of range")
        # observability: something must tie the estimate to the world
        observes_position = any(
            (not c.is_prediction) and any(int(a) < int(pos_dim) for a in c.axes)
            for c in self.channels
        )
        if not observes_position and not self.launch:
            raise ValueError(
                "sensor observes no position axis and has no launch point: "
                "position would be unobservable"
            )

    def resolve(self, embodiment: Any = None, pos_dim: int = 2) -> "SensorModel":
        """Apply the embodiment's degradation and return a runtime model."""
        pos_dim = int(pos_dim)
        self.validate(pos_dim)
        if embodiment is None:
            noise_pos, delay, scale = 0.0, 0, 1.0
        else:
            noise_pos = float(getattr(embodiment, "sensor_noise_pos", 0.0) or 0.0)
            delay = int(getattr(embodiment, "sensor_delay", 0) or 0)
            scale = float(getattr(embodiment, "sensor_noise_scale", 1.0) or 0.0)

        channels: List[ResolvedChannel] = []
        imu: Optional[ResolvedChannel] = None
        for c in self.channels:
            if c.noise_from_config:
                sigma = noise_pos
            else:
                sigma = float(c.base_noise) * scale
            rc = ResolvedChannel(
                kind=c.kind,
                axes=tuple(int(a) for a in c.axes),
                sigma=sigma,
                latency=int(c.latency) + delay,
                gate_range=c.gate_range,
                gate_min_z=c.gate_min_z,
                description=c.description,
            )
            if c.is_prediction:
                imu = rc
            else:
                channels.append(rc)

        return SensorModel(
            channels=tuple(channels),
            imu=imu,
            anchors=tuple(tuple(float(v) for v in a) for a in self.anchors),
            launch=tuple(float(v) for v in self.launch),
            pos_dim=pos_dim,
            description=self.description,
        )

    def to_dict(self, pos_dim: Optional[int] = None) -> dict:
        if pos_dim is not None:
            self.validate(int(pos_dim))
        return {
            "description": self.description,
            "pos_dim": None if pos_dim is None else int(pos_dim),
            "launch": [float(v) for v in self.launch],
            "anchors": [[float(v) for v in a] for a in self.anchors],
            "channels": [c.to_dict() for c in self.channels],
            "measurement_dim": sum(c.dim for c in self.correction_channels),
            "prediction_channel": (
                None if self.prediction_channel is None else self.prediction_channel.kind
            ),
        }


# --------------------------------------------------------------------------- #
# Resolved suite (runtime)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedChannel:
    """A channel with the embodiment's noise and latency already applied."""

    kind: str
    axes: Tuple[int, ...]
    sigma: float
    latency: int
    gate_range: Optional[float] = None
    gate_min_z: Optional[float] = None
    description: str = ""

    @property
    def dim(self) -> int:
        return len(self.axes)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "axes": [int(a) for a in self.axes],
            "dim": self.dim,
            "sigma": float(self.sigma),
            "latency": int(self.latency),
            "gate_range": None if self.gate_range is None else float(self.gate_range),
            "gate_min_z": None if self.gate_min_z is None else float(self.gate_min_z),
            "description": self.description,
        }


@dataclass(frozen=True)
class SensorModel:
    """The runtime sensor suite: channels + geometry, ready for env and filter."""

    channels: Tuple[ResolvedChannel, ...]
    imu: Optional[ResolvedChannel] = None
    anchors: Tuple[Tuple[float, ...], ...] = ()
    launch: Tuple[float, ...] = ()
    pos_dim: int = 2
    description: str = ""

    @property
    def has_imu(self) -> bool:
        return self.imu is not None

    @property
    def measurement_dim(self) -> int:
        return sum(c.dim for c in self.channels)

    @property
    def max_latency(self) -> int:
        lats = [c.latency for c in self.channels]
        if self.imu is not None:
            lats.append(self.imu.latency)
        return max(lats) if lats else 0

    def channel(self, kind: str) -> Optional[ResolvedChannel]:
        for c in self.channels:
            if c.kind == kind:
                return c
        return None

    @property
    def fix_channel(self) -> Optional[ResolvedChannel]:
        return self.channel(FIX)

    @property
    def camera_channel(self) -> Optional[ResolvedChannel]:
        """The historical absolute-position camera: ungated, all position axes."""
        for c in self.channels:
            if c.kind == FIX and c.gate_range is None:
                if set(int(a) for a in c.axes) == set(range(self.pos_dim)):
                    return c
        return None

    def is_position_axis(self, axis: int) -> bool:
        return int(axis) < self.pos_dim

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "pos_dim": self.pos_dim,
            "measurement_dim": self.measurement_dim,
            "max_latency": self.max_latency,
            "has_imu": self.has_imu,
            "anchors": [[float(v) for v in a] for a in self.anchors],
            "launch": [float(v) for v in self.launch],
            "channels": [c.to_dict() for c in self.channels],
            "imu": None if self.imu is None else self.imu.to_dict(),
        }


@dataclass
class SensorReadings:
    """One frame of sensor output.

    ``imu`` is the measured body acceleration (prediction input).  ``channels``
    holds the correction samples keyed by kind; a missing key means "no sample
    this frame" (e.g. the checkpoint gate was not satisfied).  ``anchors`` holds
    the anchor a relative channel was measured against, so the estimator can turn
    ``p - c`` into an absolute-position correction.
    """

    imu: Any = None
    channels: Dict[str, Any] = field(default_factory=dict)
    anchors: Dict[str, Any] = field(default_factory=dict)
    fix_anchor_index: Optional[int] = None

    def has(self, kind: str) -> bool:
        return kind in self.channels

    @property
    def kinds(self) -> Tuple[str, ...]:
        return tuple(self.channels)

    def position(self, kind: str = FIX):
        """The raw sample of *kind* (used by the legacy ``measure`` shim)."""
        if kind not in self.channels:
            raise KeyError(f"no {kind!r} sample in this frame")
        return self.channels[kind]


# --------------------------------------------------------------------------- #
# Factories (keep example definitions terse and readable)
# --------------------------------------------------------------------------- #
def camera_channel(pos_dim: int, description: str = "camera y = p") -> SensorChannel:
    """The historical absolute-position camera (all position axes).

    Noise and latency come from the embodiment config, not from ``base_noise``,
    so ``noisy`` / ``delayed`` presets keep behaving exactly as before.
    """
    return SensorChannel(
        kind=FIX,
        axes=tuple(range(int(pos_dim))),
        base_noise=0.0,
        latency=0,
        noise_from_config=True,
        description=description,
    )


def imu_channel(pos_dim: int, *, base_noise: float, latency: int = 1,
                description: str = "IMU a = u_eff + d") -> SensorChannel:
    return SensorChannel(
        kind=IMU,
        axes=tuple(range(int(pos_dim))),
        base_noise=base_noise,
        latency=latency,
        description=description,
    )


def altitude_channel(axis: int, *, base_noise: float, latency: int = 0,
                     description: str = "barometer z") -> SensorChannel:
    return SensorChannel(
        kind=ALTITUDE, axes=(int(axis),), base_noise=base_noise,
        latency=latency, description=description,
    )


def flow_channel(axes: Sequence[int], *, base_noise: float, latency: int = 0,
                 description: str = "optical flow v") -> SensorChannel:
    return SensorChannel(
        kind=FLOW, axes=tuple(int(a) for a in axes), base_noise=base_noise,
        latency=latency, description=description,
    )


def checkpoint_channel(axes: Sequence[int], *, base_noise: float, latency: int = 0,
                       gate_range: Optional[float] = None,
                       gate_min_z: Optional[float] = None,
                       description: str = "checkpoint fix p - c") -> SensorChannel:
    return SensorChannel(
        kind=FIX, axes=tuple(int(a) for a in axes), base_noise=base_noise,
        latency=latency, gate_range=gate_range, gate_min_z=gate_min_z,
        description=description,
    )
