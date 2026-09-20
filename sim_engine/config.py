"""Engine configuration objects.

Everything a caller can tune lives in a dataclass here so the CLI, the Python
API and the web backend all speak the same vocabulary.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .physics import (
    DT,
    MAX_TILT,
    N_IN,
    N_NEURONS,
    N_OUT,
    SYNAPSES_PER_NEURON,
)

__all__ = [
    "PlantConfig",
    "NetworkConfig",
    "EmbodimentConfig",
    "EMBODIMENT_PRESETS",
    "BenchmarkConfig",
    "EngineConfig",
]


def _default_init_state() -> Tuple[float, float, float, float]:
    return (-0.05, 0.05, 0.0, 0.0)


@dataclass
class PlantConfig:
    """Ball-and-plate plant parameters."""

    dt: float = DT
    max_tilt: float = MAX_TILT
    init_state: Tuple[float, float, float, float] = field(
        default_factory=_default_init_state
    )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class NetworkConfig:
    """Shared brain-size parameters (kept identical across controllers)."""

    n_in: int = N_IN
    n_out: int = N_OUT
    n_neurons: int = N_NEURONS
    synapses_per_neuron: int = SYNAPSES_PER_NEURON
    inhibitory_fraction: float = 0.20
    excitatory_weight: float = 1.0
    inhibitory_weight: float = -4.0
    micro_steps: int = 10

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class EmbodimentConfig:
    """Sensor / actuator / body / disturbance model of the embodied plant.

    The defaults are the original *clean* benchmark: an ideal sensor, an ideal
    actuator and the frictionless rolling ball.  Turning any field on makes the
    body and environment part of the problem rather than a convenient wrapper.
    """

    #: named preset this was built from (informational)
    preset: str = "clean"
    #: master switch; False forces the clean loop
    enable: bool = True

    # -- sensing ------------------------------------------------------------ #
    sensor_noise_pos: float = 0.0     # Gaussian sigma on measured position [m]
    sensor_delay: int = 0             # observation delay [control frames]

    # -- estimator (always a Kalman filter; the controller never sees x) ---- #
    estimator: str = "kalman"
    estimate_process_noise: float = 0.1   # filter process noise sigma [m/s^2]
    estimate_init_pos_var: float = 0.01
    estimate_init_vel_var: float = 1.0

    # -- actuation ---------------------------------------------------------- #
    actuator_delay: int = 0           # command delay [control frames]
    actuator_gain: float = 1.0        # command scale
    actuator_bias: float = 0.0        # command offset [rad]

    # -- body dynamics ------------------------------------------------------ #
    damping: float = 0.0              # velocity damping b [1/s]  (a -= b*v)
    c_scale: float = 1.0              # effective body gain C = C_CONST * c_scale

    # -- perturbations ------------------------------------------------------ #
    process_noise: float = 0.0        # continuous acceleration jitter sigma [m/s^2]
    impulse_interval: int = 0         # deterministic impulse every N frames (0 = off)
    impulse_std: float = 0.0          # impulse magnitude sigma [m/s]
    impulse_prob: float = 0.0         # per-frame random impulse probability

    # -- domain randomisation ---------------------------------------------- #
    randomize: bool = False
    c_scale_range: Tuple[float, float] = (0.7, 1.3)
    damping_range: Tuple[float, float] = (0.0, 0.8)

    seed: int = 0

    @property
    def is_clean(self) -> bool:
        """True when this configuration is exactly the original clean benchmark."""
        if not self.enable:
            return True
        return (
            self.sensor_noise_pos == 0.0
            and self.sensor_delay == 0
            and self.actuator_delay == 0
            and self.actuator_gain == 1.0
            and self.actuator_bias == 0.0
            and self.damping == 0.0
            and self.c_scale == 1.0
            and self.process_noise == 0.0
            and self.impulse_interval == 0
            and self.impulse_std == 0.0
            and self.impulse_prob == 0.0
            and not self.randomize
        )

    @classmethod
    def from_preset(cls, name: str) -> "EmbodimentConfig":
        if name not in EMBODIMENT_PRESETS:
            raise ValueError(
                f"unknown embodiment preset {name!r}; "
                f"expected one of {sorted(EMBODIMENT_PRESETS)}"
            )
        return cls(preset=name, **EMBODIMENT_PRESETS[name])

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


#: Named embodiment presets.  ``clean`` is the original benchmark; ``embodied``
#: is the default story setting (mild noise + delay + impulses + a lossy body).
EMBODIMENT_PRESETS: Dict[str, dict] = {
    "clean": {},
    "noisy": {"sensor_noise_pos": 0.005},
    "delayed": {"sensor_delay": 3, "actuator_delay": 2},
    "perturbed": {"impulse_interval": 60, "impulse_std": 0.15},
    "heavy": {"damping": 0.6, "c_scale": 0.8},
    "embodied": {
        "sensor_noise_pos": 0.004,
        "sensor_delay": 2,
        "actuator_delay": 1,
        "impulse_interval": 80,
        "impulse_std": 0.12,
        "damping": 0.3,
        "c_scale": 0.9,
    },
    "randomized": {"randomize": True},
}


@dataclass
class BenchmarkConfig:
    """Closed-loop trajectory-evaluation parameters."""

    steps: int = 250
    radius: float = 0.15
    freq: float = 0.5
    record_spikes: bool = True
    embodiment: EmbodimentConfig = field(default_factory=EmbodimentConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TrainingConfig:
    """Behavioral-distillation hyper-parameters."""

    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.008
    error_scale: float = 0.4
    connectome_unroll: int = 3
    log_every: int = 25
    device: str = "cpu"
    seed: int = 42
    #: "clean" = the original i.i.d. error sampling; "robust" = distill the PD
    #: teacher *inside* an embodied environment (noise + delay + perturbations).
    profile: str = "clean"
    #: embodiment used for the robust profile (None -> preset "embodied")
    embodiment: Optional[EmbodimentConfig] = None
    #: robust data generation
    episodes: int = 4
    episode_steps: int = 250
    noise_augment: float = 0.0
    #: extra i.i.d. error samples added to the closed-loop dataset so the student
    #: also learns the large-error corrections the near-perfect teacher never visits
    coverage_samples: int = 2048

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class EngineConfig:
    """Top-level configuration for :class:`sim_engine.engine.Engine`.

    Both nested groups and flat convenience keywords are accepted, so these are
    equivalent::

        EngineConfig(benchmark=BenchmarkConfig(steps=400), network=NetworkConfig(n_neurons=64))
        EngineConfig(steps=400, n_neurons=64)
    """

    plant: PlantConfig = field(default_factory=PlantConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "cpu"
    #: If True the engine distills the dense + connectome controllers once at
    #: construction and reuses those weights for the "dense"/"connectome"/"snn"
    #: entries of the registry.  "random_ann" always stays untrained.
    train_on_init: bool = False
    default_controllers: Sequence[str] = (
        "pid",
        "random_ann",
        "flylike_ann",
        "snn_transferred",
    )
    #: Optional path to a ``.pt`` weights bundle produced by ``cli train``.
    weights_path: Optional[str] = None

    # -- flat convenience overrides (merged into the nested groups) ---------- #
    steps: Optional[int] = None
    radius: Optional[float] = None
    freq: Optional[float] = None
    record_spikes: Optional[bool] = None
    micro_steps: Optional[int] = None
    n_neurons: Optional[int] = None
    synapses_per_neuron: Optional[int] = None
    init_state: Optional[Tuple[float, float, float, float]] = None
    dt: Optional[float] = None
    max_tilt: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None

    # -- embodiment convenience overrides ----------------------------------- #
    #: full embodiment config (takes precedence over the flat fields below)
    embodiment: Optional[EmbodimentConfig] = None
    embodiment_preset: Optional[str] = None
    sensor_noise_pos: Optional[float] = None
    sensor_delay: Optional[int] = None
    actuator_delay: Optional[int] = None
    disturbance_std: Optional[float] = None   # -> process_noise
    damping: Optional[float] = None
    c_scale: Optional[float] = None

    def __post_init__(self) -> None:
        b, n, p, t = self.benchmark, self.network, self.plant, self.training
        if self.steps is not None:
            b.steps = int(self.steps)
        if self.radius is not None:
            b.radius = float(self.radius)
        if self.freq is not None:
            b.freq = float(self.freq)
        if self.record_spikes is not None:
            b.record_spikes = bool(self.record_spikes)
        if self.micro_steps is not None:
            n.micro_steps = int(self.micro_steps)
        if self.n_neurons is not None:
            n.n_neurons = int(self.n_neurons)
        if self.synapses_per_neuron is not None:
            n.synapses_per_neuron = int(self.synapses_per_neuron)
        if self.init_state is not None:
            p.init_state = tuple(float(v) for v in self.init_state)
        if self.dt is not None:
            p.dt = float(self.dt)
        if self.max_tilt is not None:
            p.max_tilt = float(self.max_tilt)
        if self.epochs is not None:
            t.epochs = int(self.epochs)
        if self.batch_size is not None:
            t.batch_size = int(self.batch_size)
        if self.lr is not None:
            t.lr = float(self.lr)

        # Embodiment: an explicit config wins, otherwise a preset + flat fields.
        flat = (self.sensor_noise_pos, self.sensor_delay,
                self.actuator_delay, self.disturbance_std, self.damping, self.c_scale)
        if self.embodiment is not None:
            b.embodiment = self.embodiment
        elif self.embodiment_preset is not None or any(v is not None for v in flat):
            cfg = EmbodimentConfig.from_preset(self.embodiment_preset or "clean")
            if self.sensor_noise_pos is not None:
                cfg.sensor_noise_pos = float(self.sensor_noise_pos)
            if self.sensor_delay is not None:
                cfg.sensor_delay = int(self.sensor_delay)
            if self.actuator_delay is not None:
                cfg.actuator_delay = int(self.actuator_delay)
            if self.disturbance_std is not None:
                cfg.process_noise = float(self.disturbance_std)
            if self.damping is not None:
                cfg.damping = float(self.damping)
            if self.c_scale is not None:
                cfg.c_scale = float(self.c_scale)
            b.embodiment = cfg

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["default_controllers"] = list(self.default_controllers)
        return d
