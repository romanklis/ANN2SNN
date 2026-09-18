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

__all__ = ["PlantConfig", "NetworkConfig", "EngineConfig", "BenchmarkConfig"]


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
class BenchmarkConfig:
    """Closed-loop trajectory-evaluation parameters."""

    steps: int = 250
    radius: float = 0.15
    freq: float = 0.5
    record_spikes: bool = True

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

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["default_controllers"] = list(self.default_controllers)
        return d
