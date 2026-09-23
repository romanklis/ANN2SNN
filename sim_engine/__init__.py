"""``sim_engine`` — ball-and-plate simulation engine for the ANN2SNN benchmark.

Refactor of the original monolithic prototype into importable modules:

===============  =================================================================
:mod:`physics`   ``step_physics`` + all physical constants, ``BallPlatePlant``
:mod:`controllers`  ``ClassicalPDController``, ``DenseNNController``,
                 ``ConnectomeANNController``, ``LosslessConnectomeSNN``
:mod:`training`  behavioral distillation of the PD teacher into the ANNs
:mod:`benchmark` closed-loop orbit evaluation + metrics
:mod:`engine`    ``Engine`` / ``SimulationSession`` high-level orchestration
:mod:`api`       reusable, JSON-in/JSON-out API for a web backend
:mod:`cli`       ``python -m sim_engine ...`` command-line entrypoint
===============  =================================================================

Quick start
-----------
>>> from sim_engine import Engine
>>> engine = Engine()                       # doctest: +SKIP
>>> report = engine.run_benchmark(["pid", "random_ann", "flylike_ann", "snn_transferred"])
>>> report["ranking"]                       # doctest: +SKIP
['pid', 'snn_transferred', 'flylike_ann', 'random_ann']
"""

from __future__ import annotations

from . import physics, reference, serialization
from .benchmark import BenchmarkReport, TrajectoryResult, evaluate, run_closed_loop
from .config import (
    DEFAULT_EXAMPLE,
    DEFAULT_STEPS,
    EMBODIMENT_PRESETS,
    BenchmarkConfig,
    EmbodimentConfig,
    EngineConfig,
    NetworkConfig,
    PlantConfig,
    TrainingConfig,
)
from .environment import EmbodiedEnv, embodiment_specs
from .examples import (
    DEFAULT_EXAMPLE as _DEFAULT_EXAMPLE,
    ExamplePlant,
    ExampleSpec,
    example_names,
    get_example,
    list_examples,
    register_example,
)
from .controllers import (
    BaseController,
    ClassicalPDController,
    ConnectomeANNController,
    ConnectomeTopology,
    DenseNNController,
    LosslessConnectomeSNN,
)
from .engine import Engine, SimulationSession
from .physics import (
    C_CONST,
    DRONE_HALF,
    DT,
    GRAVITY,
    MAX_THRUST,
    MAX_TILT,
    N_IN,
    N_NEURONS,
    N_OUT,
    PLATE_HALF,
    STATE_DIM,
    SYNAPSES_PER_NEURON,
    TOTAL_SYNAPSES,
    BallPlatePlant,
    PointMass3D,
    step_physics,
    step_point_mass,
)
from .reference import (
    RefPoint,
    Reference,
    lissajous_reference,
    orbit_reference,
    setpoint_reference,
)
from .registry import CANONICAL_CONTROLLERS, ControllerRegistry
from .serialization import to_jsonable

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # physics
    "step_physics", "BallPlatePlant", "step_point_mass", "PointMass3D",
    "GRAVITY", "C_CONST", "DT", "MAX_TILT", "PLATE_HALF",
    "MAX_THRUST", "DRONE_HALF", "STATE_DIM",
    "N_IN", "N_OUT", "N_NEURONS", "SYNAPSES_PER_NEURON", "TOTAL_SYNAPSES",
    # controllers
    "BaseController", "ClassicalPDController", "DenseNNController",
    "ConnectomeANNController", "ConnectomeTopology", "LosslessConnectomeSNN",
    # config
    "PlantConfig", "NetworkConfig", "BenchmarkConfig", "TrainingConfig", "EngineConfig",
    "EmbodimentConfig", "EMBODIMENT_PRESETS", "DEFAULT_STEPS", "DEFAULT_EXAMPLE",
    # environment
    "EmbodiedEnv", "embodiment_specs",
    # examples
    "ExampleSpec", "ExamplePlant", "get_example", "list_examples",
    "example_names", "register_example",
    # reference
    "RefPoint", "Reference", "orbit_reference", "lissajous_reference",
    "setpoint_reference",
    # orchestration
    "Engine", "SimulationSession", "ControllerRegistry", "CANONICAL_CONTROLLERS",
    # benchmark
    "run_closed_loop", "evaluate", "TrajectoryResult", "BenchmarkReport",
    # utils
    "to_jsonable",
    "physics", "reference", "serialization",
]
