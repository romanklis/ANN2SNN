"""High-level engine: the single object a backend needs to hold.

``Engine`` owns the plant configuration, the controller registry and (optionally)
the distilled weights.  It exposes three levels of API:

1. **Batch** — :meth:`Engine.run` / :meth:`Engine.run_benchmark` return fully
   JSON-serializable results for the whole orbit.
2. **Interactive** — :meth:`Engine.session` returns a :class:`SimulationSession`
   that can be stepped one frame at a time (what the web demo uses).
3. **Introspection** — :meth:`Engine.describe` advertises the available brains.

Nothing here imports Flask/HTTP: the engine is transport-agnostic and gets wired
to a web framework by the ``build-backend-api`` step.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import training as training_mod
from .benchmark import BenchmarkReport, TrajectoryResult, evaluate, run_closed_loop
from .config import (
    BenchmarkConfig,
    EngineConfig,
    NetworkConfig,
    PlantConfig,
    TrainingConfig,
)
from .controllers import BaseController
from .physics import BallPlatePlant, step_physics
from .reference import Reference, orbit_reference
from .registry import ControllerRegistry, LABELS
from .serialization import to_jsonable

__all__ = ["Engine", "SimulationSession", "EngineError"]


class EngineError(RuntimeError):
    """Raised for invalid engine usage (e.g. unknown controller)."""


def _resolve_device(preferred: str = "auto") -> str:
    if preferred in ("auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preferred


class Engine:
    """Owns configuration + controllers and drives the benchmark."""

    def __init__(self, config: Optional[EngineConfig] = None, *, train: Optional[bool] = None) -> None:
        self.config = config or EngineConfig()
        self.device = _resolve_device(self.config.device)
        self.config.device = self.device
        self.config.training.device = self.device

        self.reference_cache: Dict[tuple, Reference] = {}

        dense = connectome = None
        trained = False

        if self.config.weights_path:
            loaded = training_mod.load_weights(self.config.weights_path, device=self.device)
            dense = loaded.get("dense")
            connectome = loaded.get("connectome")
            trained = dense is not None and connectome is not None

        do_train = self.config.train_on_init if train is None else train
        if do_train:
            distilled = training_mod.distill(
                self.config.network, self.config.training, log_fn=self._log
            )
            dense = distilled["dense"]
            connectome = distilled["connectome"]
            self.last_training = {
                "history": distilled["history"],
                "final_loss": distilled["final_loss"],
                "config": distilled["config"],
            }
            trained = True
        else:
            self.last_training = None

        self.registry = ControllerRegistry(
            self.config.network,
            device=self.device,
            seed=self.config.seed,
            dense=dense,
            connectome=connectome,
            micro_steps=self.config.network.micro_steps,
            trained=trained,
        )

        self._sessions: Dict[str, "SimulationSession"] = {}
        self._lock = threading.Lock()

    # -- logging ------------------------------------------------------------ #
    def _log(self, which: str, epoch: int, loss: float) -> None:  # pragma: no cover
        print(f"[train:{which}] epoch {epoch:03d} | loss {loss:.6f}")

    # -- reference ---------------------------------------------------------- #
    def reference(
        self,
        steps: Optional[int] = None,
        radius: Optional[float] = None,
        freq: Optional[float] = None,
    ) -> Reference:
        bc = self.config.benchmark
        steps = steps or bc.steps
        radius = bc.radius if radius is None else radius
        freq = bc.freq if freq is None else freq
        key = (steps, radius, freq)
        if key not in self.reference_cache:
            self.reference_cache[key] = orbit_reference(
                steps=steps, radius=radius, freq=freq,
                dt=self.config.plant.dt, device="cpu",
            )
        return self.reference_cache[key]

    # -- batch API ---------------------------------------------------------- #
    def build_controller(self, name: str) -> BaseController:
        try:
            return self.registry.build(name)
        except KeyError as exc:
            raise EngineError(str(exc)) from exc

    def run(
        self,
        name: str,
        *,
        reference: Optional[Reference] = None,
        config: Optional[BenchmarkConfig] = None,
    ) -> TrajectoryResult:
        """Run one controller over the full closed-loop orbit."""
        ctrl = self.build_controller(name)
        init = torch.tensor(self.config.plant.init_state, dtype=torch.float32)
        return run_closed_loop(
            ctrl,
            reference=reference or self.reference(),
            init_state=init,
            config=config or self.config.benchmark,
            name=name,
        )

    def run_benchmark(
        self,
        names: Optional[Sequence[str]] = None,
        *,
        include_trace: bool = True,
        as_dict: bool = True,
    ):
        """Evaluate several controllers on a shared reference trajectory."""
        names = list(names) if names else list(self.config.default_controllers)
        controllers = {n: self.build_controller(n) for n in names}
        report = evaluate(
            controllers,
            reference=self.reference(),
            init_state=torch.tensor(self.config.plant.init_state, dtype=torch.float32),
            config=self.config.benchmark,
        )
        return report.to_dict(include_trace=include_trace) if as_dict else report

    # -- interactive API ---------------------------------------------------- #
    def session(self, name: str = "pid", **kwargs) -> "SimulationSession":
        """Create a step-by-step interactive session."""
        return SimulationSession(self, name, **kwargs)

    def new_session(self, name: str = "pid", **kwargs) -> str:
        """Create a session and return its id (for stateless web backends)."""
        sess = self.session(name, **kwargs)
        with self._lock:
            self._sessions[sess.id] = sess
        return sess.id

    def get_session(self, session_id: str) -> "SimulationSession":
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise EngineError(f"no such session: {session_id!r}") from exc

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    # -- introspection ------------------------------------------------------ #
    def describe(self) -> dict:
        return to_jsonable(
            {
                "engine": "ann2snn.sim_engine",
                "device": self.device,
                "config": self.config.to_dict(),
                "trained": self.registry.trained,
                "controllers": self.registry.describe_all(),
                "training": (
                    None
                    if self.last_training is None
                    else {
                        "final_loss": self.last_training["final_loss"],
                        "config": self.last_training["config"],
                        "epochs_logged": len(self.last_training["history"]["dense"]),
                    }
                ),
            }
        )

    def save_weights(self, path: str) -> str:
        return training_mod.save_weights(
            path,
            dense=self.registry.dense,
            connectome=self.registry.connectome,
            training=self.last_training,
        )


class SimulationSession:
    """One plant + one controller + one cursor into the reference trajectory.

    Every :meth:`step` advances exactly one 20 ms frame and returns a plain dict
    (numpy arrays, no tensors) ready to be JSON-encoded by a web layer.
    """

    def __init__(
        self,
        engine: Engine,
        name: str = "pid",
        *,
        steps: Optional[int] = None,
        radius: Optional[float] = None,
        freq: Optional[float] = None,
        init_state: Optional[Sequence[float]] = None,
        auto_reset: bool = True,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.engine = engine
        self.name = engine.registry.resolve(name)
        self.reference = engine.reference(steps=steps, radius=radius, freq=freq)
        self.auto_reset = auto_reset
        self.init_state = tuple(
            init_state if init_state is not None else engine.config.plant.init_state
        )
        self.plant = BallPlatePlant(
            init_state=self.init_state,
            dt=engine.config.plant.dt,
            max_tilt=engine.config.plant.max_tilt,
            device=engine.device,
        )
        self.controller = engine.build_controller(self.name)
        self.k = 0
        self.done = False
        self.history: Dict[str, list] = {"state": [], "tilt": [], "error_cm": [], "target": []}
        self._record_current()

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> dict:
        self.controller.reset()
        self.plant.reset()
        self.k = 0
        self.done = False
        self.history = {"state": [], "tilt": [], "error_cm": [], "target": []}
        self._record_current()
        return self.observe()

    # -- stepping ----------------------------------------------------------- #
    def step(self, action: Optional[Sequence[float]] = None) -> dict:
        """Advance one frame.

        If *action* is given, it overrides the controller (manual joystick mode);
        otherwise the session's controller computes the tilt.
        """
        if self.done and self.auto_reset:
            self.reset()

        state = self.plant.state
        if action is None:
            with torch.no_grad():
                tilt = self.controller.act(state, self.reference.at(self.k))
        else:
            tilt = torch.as_tensor(action, dtype=torch.float32, device=self.engine.device)

        tilt_vec = tilt.detach().cpu().numpy()
        self.history["tilt"].append(tilt_vec)
        self.plant.step(tilt)
        self.k += 1
        self.done = self.k >= len(self.reference)
        return self.observe(tilt=tilt_vec)

    def step_n(self, n: int = 1, action: Optional[Sequence[float]] = None) -> dict:
        for _ in range(int(n)):
            obs = self.step(action=action)
        return obs

    # -- observation -------------------------------------------------------- #
    def _record_current(self) -> None:
        st = self.plant.state.detach().cpu().numpy()
        ref = self.reference.at(self.k)
        target = ref.pos.detach().cpu().numpy()
        self.history["state"].append(st)
        self.history["target"].append(target)
        self.history["error_cm"].append(
            float(np.linalg.norm(st[:2] - target) * 100.0)
        )

    def observe(self, tilt=None) -> dict:
        st = self.plant.state.detach().cpu().numpy()
        idx = min(self.k, len(self.reference) - 1)
        ref = self.reference.at(idx)
        target = ref.pos.detach().cpu().numpy()
        err_cm = float(np.linalg.norm(st[:2] - target) * 100.0)
        spikes = self.controller.last_spikes()
        return {
            "session_id": self.id,
            "controller": self.name,
            "t": float(self.k * self.reference.dt),
            "step": int(self.k),
            "done": bool(self.done),
            "state": st.astype(float),
            "target": target.astype(float),
            "reference": {
                "pos": ref.pos.detach().cpu().numpy().astype(float),
                "vel": ref.vel.detach().cpu().numpy().astype(float),
                "acc": ref.acc.detach().cpu().numpy().astype(float),
            },
            "tilt": None if tilt is None else np.asarray(tilt, dtype=float),
            "error_cm": err_cm,
            "spikes": None if spikes is None else spikes.detach().cpu().numpy(),
        }

    def trajectory(self) -> dict:
        """The whole recorded trajectory of this session (JSON-ready)."""
        return to_jsonable(
            {
                "session_id": self.id,
                "controller": self.name,
                "steps": self.k,
                "dt": self.reference.dt,
                "state": np.asarray(self.history["state"]),
                "tilt": np.asarray(self.history["tilt"]) if self.history["tilt"] else [],
                "target": np.asarray(self.history["target"]),
                "error_cm": np.asarray(self.history["error_cm"]),
                "mean_error_cm": float(np.mean(self.history["error_cm"])),
            }
        )

    def describe(self) -> dict:
        return to_jsonable(
            {
                "session_id": self.id,
                "controller": self.name,
                "label": LABELS.get(self.name, self.name),
                "steps_total": len(self.reference),
                "step": self.k,
                "dt": self.reference.dt,
                "radius": self.reference.radius,
                "freq": self.reference.freq,
                "init_state": list(self.init_state),
                "controller_info": self.controller.describe(),
            }
        )

    def set_controller(self, name: str) -> dict:
        self.name = self.engine.registry.resolve(name)
        self.controller = self.engine.build_controller(self.name)
        self.controller.reset()
        return self.describe()
