"""Reusable, transport-agnostic API for driving the simulation engine.

This module is the seam the ``build-backend-api`` step will import.  Every
function takes plain dicts/lists in and returns JSON-serializable dicts out, so
a Flask/FastAPI layer only has to do routing and validation::

    from sim_engine.api import EngineService

    service = EngineService()                 # optionally trained
    service.benchmark(controllers=["pid", "snn_transferred"])
    sid = service.new_session("random_ann")
    service.step(sid)

Nothing in here imports a web framework.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

from .config import (
    BenchmarkConfig,
    EngineConfig,
    NetworkConfig,
    PlantConfig,
    TrainingConfig,
)
from .engine import Engine, SimulationSession
from .registry import ALIASES, CANONICAL_CONTROLLERS, LABELS, normalize_name
from .serialization import to_jsonable

__all__ = [
    "config_from_dict",
    "build_engine",
    "EngineService",
    "list_controllers",
    "run_benchmark",
    "default_config",
]


def config_from_dict(data: Optional[dict] = None) -> EngineConfig:
    """Build an :class:`EngineConfig` from a (possibly partial) dict."""
    data = data or {}
    cfg = EngineConfig()

    if "plant" in data and data["plant"]:
        cfg.plant = PlantConfig(**{**dataclasses.asdict(cfg.plant), **data["plant"]})
    if "network" in data and data["network"]:
        cfg.network = NetworkConfig(**{**dataclasses.asdict(cfg.network), **data["network"]})
    if "benchmark" in data and data["benchmark"]:
        cfg.benchmark = BenchmarkConfig(
            **{**dataclasses.asdict(cfg.benchmark), **data["benchmark"]}
        )
    if "training" in data and data["training"]:
        cfg.training = TrainingConfig(**{**dataclasses.asdict(cfg.training), **data["training"]})

    for key in ("seed", "device", "train_on_init", "weights_path"):
        if key in data and data[key] is not None:
            setattr(cfg, key, data[key])
    if "default_controllers" in data and data["default_controllers"]:
        cfg.default_controllers = tuple(data["default_controllers"])

    # Flat convenience keys, e.g. {"steps": 250, "radius": 0.15, "freq": 0.5}.
    for flat in ("steps", "radius", "freq", "record_spikes"):
        if flat in data and data[flat] is not None:
            setattr(cfg.benchmark, flat, data[flat])
    if "init_state" in data and data["init_state"] is not None:
        cfg.plant.init_state = tuple(float(v) for v in data["init_state"])
    if "micro_steps" in data and data["micro_steps"] is not None:
        cfg.network.micro_steps = int(data["micro_steps"])
    if "n_neurons" in data and data["n_neurons"] is not None:
        cfg.network.n_neurons = int(data["n_neurons"])
    if "epochs" in data and data["epochs"] is not None:
        cfg.training.epochs = int(data["epochs"])

    return cfg


def default_config() -> dict:
    """The full default engine configuration, JSON-ready."""
    return to_jsonable(EngineConfig().to_dict())


def build_engine(config: Optional[dict] = None, *, train: Optional[bool] = None) -> Engine:
    """Convenience constructor: ``build_engine({"steps": 250, "train_on_init": True})``."""
    return Engine(config_from_dict(config), train=train)


def list_controllers() -> List[dict]:
    """Catalogue of the brains available to the frontend."""
    return [
        {
            "name": name,
            "label": LABELS.get(name, name),
            "aliases": sorted(k for k, v in ALIASES.items() if v == name),
        }
        for name in CANONICAL_CONTROLLERS
    ]


def run_benchmark(
    config: Optional[dict] = None,
    controllers: Optional[Sequence[str]] = None,
    *,
    include_trace: bool = True,
    train: Optional[bool] = None,
) -> dict:
    """One-shot benchmark: build an engine and evaluate the requested brains."""
    engine = build_engine(config, train=train)
    return engine.run_benchmark(controllers, include_trace=include_trace)


class EngineService:
    """Stateful facade with in-memory sessions — the backend's main entry point."""

    def __init__(self, config: Optional[dict] = None, *, train: Optional[bool] = None) -> None:
        self.engine = build_engine(config, train=train)

    # -- catalogue ---------------------------------------------------------- #
    def controllers(self) -> List[dict]:
        return self.engine.describe()["controllers"]

    def describe(self) -> dict:
        return self.engine.describe()

    def config(self) -> dict:
        return to_jsonable(self.engine.config.to_dict())

    # -- batch -------------------------------------------------------------- #
    def benchmark(
        self,
        controllers: Optional[Sequence[str]] = None,
        *,
        include_trace: bool = True,
    ) -> dict:
        return self.engine.run_benchmark(controllers, include_trace=include_trace)

    def run(self, controller: str, *, include_trace: bool = True) -> dict:
        return to_jsonable(self.engine.run(controller).to_dict(include_trace=include_trace))

    # -- interactive sessions ---------------------------------------------- #
    def new_session(self, controller: str = "pid", **kwargs) -> dict:
        session_id = self.engine.new_session(controller, **kwargs)
        return self.engine.get_session(session_id).describe()

    def session_info(self, session_id: str) -> dict:
        return self.engine.get_session(session_id).describe()

    def step(self, session_id: str, action: Optional[Sequence[float]] = None, n: int = 1) -> dict:
        sess = self.engine.get_session(session_id)
        return to_jsonable(sess.step_n(n, action=action))

    def reset(self, session_id: str) -> dict:
        return to_jsonable(self.engine.get_session(session_id).reset())

    def trajectory(self, session_id: str) -> dict:
        return self.engine.get_session(session_id).trajectory()

    def set_controller(self, session_id: str, controller: str) -> dict:
        return self.engine.get_session(session_id).set_controller(controller)

    def close_session(self, session_id: str) -> None:
        self.engine.drop_session(session_id)

    # -- weights ------------------------------------------------------------ #
    def save_weights(self, path: str) -> str:
        return self.engine.save_weights(path)
