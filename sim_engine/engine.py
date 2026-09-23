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
import logging
import threading
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import training as training_mod
from .benchmark import BenchmarkReport, TrajectoryResult, evaluate, run_closed_loop

log = logging.getLogger("sim_engine.engine")
from .config import (
    BenchmarkConfig,
    EngineConfig,
    NetworkConfig,
    PlantConfig,
    TrainingConfig,
)
from .controllers import BaseController
from .environment import EmbodiedEnv
from .estimators import KalmanFilter
from .examples import get_example
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
        self.spec = get_example(self.config.benchmark.example)

        dense = connectome = None
        trained = False
        loaded_training = None

        if self.config.weights_path:
            try:
                loaded = training_mod.load_weights(self.config.weights_path, device=self.device)
            except Exception as exc:  # noqa: BLE001
                # A stale or mismatched bundle (e.g. cached for another example,
                # whose input/output dims differ) must never take a request down:
                # warn and fall back to an untrained engine.
                log.warning(
                    "ignoring unusable weights bundle %s for example %r (%s: %s)",
                    self.config.weights_path, self.config.benchmark.example,
                    type(exc).__name__, exc,
                )
            else:
                dense = loaded.get("dense")
                connectome = loaded.get("connectome")
                loaded_training = loaded.get("training")
                trained = dense is not None and connectome is not None

        do_train = self.config.train_on_init if train is None else train
        if do_train:
            profile = getattr(self.config.training, "profile", "clean")
            # Closed-loop distillation for both profiles: the teacher acts on the
            # Kalman estimate x̂ (position-only camera), never the true state.
            train_cfg = dataclasses.replace(
                self.config.training,
                embodiment=self.config.training.embodiment
                or self.config.benchmark.embodiment,
            )
            distilled = training_mod.distill_profile(
                profile,
                self.config.network,
                train_cfg,
                benchmark=self.config.benchmark,
                log_fn=self._log,
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
            # Keep the provenance saved with a loaded bundle (how it was trained).
            self.last_training = loaded_training if trained else None

        self.registry = ControllerRegistry(
            self.config.network,
            device=self.device,
            seed=self.config.seed,
            dense=dense,
            connectome=connectome,
            micro_steps=self.config.network.micro_steps,
            trained=trained,
            action_limit=self.spec.control_limit,
            plant_gain=self.spec.plant_gain,
            pos_dim=self.spec.pos_dim,
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
        spec = get_example(bc.example)
        steps = steps or bc.steps
        radius = bc.radius if radius is None else radius
        freq = bc.freq if freq is None else freq
        key = (spec.name, steps, radius, freq)
        if key not in self.reference_cache:
            self.reference_cache[key] = spec.reference(
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
        """Run one controller over the full closed-loop trajectory."""
        ctrl = self.build_controller(name)
        cfg = config or self.config.benchmark
        spec = get_example(cfg.example)
        init = torch.tensor(spec.init_state, dtype=torch.float32)
        emb = getattr(cfg, "embodiment", None)
        env = None
        if emb is not None and emb.enable:
            env = spec.make_env(emb, dt=self.config.plant.dt, device=self.device)
        return run_closed_loop(
            ctrl,
            reference=reference or self.reference(),
            init_state=init,
            config=cfg,
            name=name,
            env=env,
            example=spec.name,
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
        spec = get_example(self.config.benchmark.example)
        report = evaluate(
            controllers,
            reference=self.reference(),
            init_state=torch.tensor(spec.init_state, dtype=torch.float32),
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
                        "final_loss": self.last_training.get("final_loss"),
                        "config": self.last_training.get("config"),
                        "epochs_logged": len(
                            self.last_training.get("history", {}).get("dense", [])
                        ),
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
        self.spec = get_example(engine.config.benchmark.example)
        self.reference = engine.reference(steps=steps, radius=radius, freq=freq)
        self.auto_reset = auto_reset
        self.init_state = tuple(
            init_state if init_state is not None else self.spec.init_state
        )
        self.plant = self.spec.make_plant(dt=engine.config.plant.dt, device=engine.device)
        self.controller = engine.build_controller(self.name)
        emb = engine.config.benchmark.embodiment
        self.env = None
        self.estimator = None
        if emb is not None and emb.enable:
            self.env = self.spec.make_env(emb, dt=engine.config.plant.dt, device=engine.device)
            self.estimator = self.spec.make_estimator(
                emb, dt=engine.config.plant.dt, device=engine.device
            )
        self._u_prev = torch.zeros(self.spec.control_dim, dtype=torch.float32,
                                   device=engine.device)
        self.k = 0
        self.done = False
        self.history: Dict[str, list] = {"state": [], "tilt": [], "error_cm": [], "target": []}
        self._record_current()

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> dict:
        self.controller.reset()
        self.plant.reset()
        if self.env is not None:
            self.env.reset()
        if self.estimator is not None:
            self.estimator.reset()
        self._u_prev = torch.zeros(self.spec.control_dim, dtype=torch.float32,
                                   device=self.engine.device)
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
            if self.env is not None:
                readings = self.env.sense(state)               # IMU + delayed channels
                xhat = self.estimator.update(                  # Kalman estimate
                    readings=readings, control=self._u_prev,
                    acceleration=readings.imu,
                )
            else:
                xhat = state
            with torch.no_grad():
                tilt = self.controller.act(xhat, self.reference.at(self.k))
        else:
            tilt = torch.as_tensor(action, dtype=torch.float32, device=self.engine.device)
        self._u_prev = tilt.detach()

        if self.env is not None:
            applied = self.env.actuate(tilt)
            self.plant.state = self.env.step(state, applied, self.k)
            self.plant.t += 1
        else:
            applied = tilt
            self.plant.step(tilt)

        tilt_vec = applied.detach().cpu().numpy()
        self.history["tilt"].append(tilt_vec)
        self.k += 1
        self.done = self.k >= len(self.reference)
        return self.observe(tilt=tilt_vec)

    def step_n(self, n: int = 1, action: Optional[Sequence[float]] = None) -> dict:
        for _ in range(int(n)):
            obs = self.step(action=action)
        return obs

    # -- observation -------------------------------------------------------- #
    def _error_cm(self, state: np.ndarray, idx: int) -> float:
        d = self.spec.pos_dim
        ref = self.reference.at(idx)
        target = ref.pos.detach().cpu().numpy()
        return float(np.linalg.norm(state[:d] - target) * 100.0)

    def _record_current(self) -> None:
        st = self.plant.state.detach().cpu().numpy()
        ref = self.reference.at(self.k)
        target = ref.pos.detach().cpu().numpy()
        self.history["state"].append(st)
        self.history["target"].append(target)
        self.history["error_cm"].append(self._error_cm(st, self.k))

    def observe(self, tilt=None) -> dict:
        st = self.plant.state.detach().cpu().numpy()
        idx = min(self.k, len(self.reference) - 1)
        ref = self.reference.at(idx)
        target = ref.pos.detach().cpu().numpy()
        err_cm = self._error_cm(st, idx)
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
