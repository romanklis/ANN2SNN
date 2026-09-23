"""Flask (+flask-cors) HTTP API for the interactive ANN2SNN dashboard.

A thin, transport-only layer over :class:`sim_engine.api.EngineService` and
:mod:`sim_engine.benchmark`.  It exposes the controller catalogue, batch and
single-controller rollouts, interactive sessions, behavioural distillation and a
server-side MP4 exporter, then serves the built frontend bundle at ``/``.

Design notes
------------
* **No maths here.**  Every trajectory/error/spike value comes from the engine;
  this module only validates JSON, normalises field names and serialises.
* **Deterministic.**  A request's ``seed`` selects an engine instance built with
  that seed (cached), so identical requests give identical traces.
* **Bounded.**  ``steps`` is clamped to ``[1, MAX_STEPS]`` and distillation
  ``epochs`` to ``[1, MAX_EPOCHS]``.
* **CPU-only**, torch/OpenMP threads capped by ``ANN2SNN_THREADS``.

Run (inside the DinD ``ann2snn`` image)::

    gunicorn -b 0.0.0.0:8080 --workers 2 --timeout 300 server.wsgi:application
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Environment must be configured *before* importing torch/matplotlib.
# --------------------------------------------------------------------------- #
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
_THREADS = os.environ.get("ANN2SNN_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("TORCH_NUM_THREADS", _THREADS)

# Make the repo root importable no matter the CWD, so `sim_engine` and `tools`
# resolve when running `python server/app.py` or via gunicorn.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flask import (  # noqa: E402
    Flask,
    after_this_request,
    jsonify,
    request,
    send_file,
    send_from_directory,
)
from flask_cors import CORS  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

import torch  # noqa: E402
import numpy as np  # noqa: E402

from sim_engine.api import EngineService, list_controllers  # noqa: E402
from sim_engine.benchmark import evaluate, run_closed_loop  # noqa: E402
from sim_engine.config import (  # noqa: E402
    DEFAULT_STEPS,
    BenchmarkConfig,
    EmbodimentConfig,
    NetworkConfig,
    TrainingConfig,
)
from sim_engine.engine import EngineError  # noqa: E402
from sim_engine.environment import (  # noqa: E402
    EmbodiedEnv,
    embodiment_specs,
)
from sim_engine.examples import example_names, get_example, list_examples  # noqa: E402
from sim_engine.physics import DT, MAX_TILT  # noqa: E402
from sim_engine.registry import CANONICAL_CONTROLLERS, LABELS  # noqa: E402
from sim_engine.robustness import DEFAULT_AXIS_POINTS, MAX_POINTS  # noqa: E402
from sim_engine.robustness import sweep as robustness_sweep  # noqa: E402
from sim_engine.serialization import to_jsonable  # noqa: E402

from server.validation import (  # noqa: E402
    BadRequest,
    _as_bool,
    _as_float,
    _as_int,
    _as_spike_format,
    _as_trace_level,
    _body,
    _env_from_body,
    _example_from_body,
    _profile_from_body,
    _spike_payload,
)

from server.runtime import DEFAULT_MICRO_STEPS, RUNTIME, WEIGHTS_PATH  # noqa: E402

try:  # thread-cap the shared torch runtime once
    torch.set_num_threads(max(1, int(_THREADS)))
except Exception:  # pragma: no cover - defensive
    pass


def _example_geometry(body: Dict[str, Any]) -> tuple:
    """Parse ``example`` + ``radius``/``freq``.

    Radius and frequency default to the **selected example's** own trajectory
    (the ball's 0.15 m / 0.5 Hz are not the drone's), so an API caller that only
    names the example still gets a sensible task.
    """
    example = _example_from_body(body) or RUNTIME.default_example
    defaults = get_example(example).defaults or {}
    radius = _as_float(body.get("radius"), "radius",
                       float(defaults.get("radius", 0.15)), 0.0, 1.0)
    freq = _as_float(body.get("freq"), "freq",
                     float(defaults.get("freq", 0.5)), 0.0, 10.0)
    return example, radius, freq

# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #
APP_NAME = "ann2snn-dashboard"
APP_VERSION = "1.0.0"

ENGINE_VERSION: str = getattr(__import__("sim_engine"), "__version__", "0.1.0")
TORCH_VERSION: str = getattr(torch, "__version__", "unknown")

#: Hard bounds so one request cannot trigger an unbounded rollout / training run.
MAX_STEPS: int = 2000
EXPORT_MAX_STEPS: int = 600
MAX_EPOCHS: int = 200
SEED_MIN: int = 0
SEED_MAX: int = 2**31 - 1

#: Half-side of the physical plate [m]; a ball beyond this has left the plate.
#: Shared with the frontend stage and the on-plate statistic.
PLATE_HALF_M: float = 0.25

#: Static provenance for each controller ("how it was obtained / what it is worth
#: / how it was trained").  Live facts (trained, spiking, n_neurons, epochs,
#: seed, final loss) are merged in by ``/api/controllers``.
MODEL_GUIDE: Dict[str, Dict[str, Any]] = {
    "pid": {
        "kind": "classical",
        "obtained": "Analytic PD control law with acceleration feed-forward "
                    "(closed form) — not learned.",
        "value": "The teacher and strongest tracker; guarantees the ball stays "
                 "on the plate.",
        "training_method": "none — fixed proportional/derivative gains",
        "teacher": None,
        "transferred_from": None,
    },
    "random_ann": {
        "kind": "random",
        "obtained": "Dense 4-N-2 feed-forward ANN with random weights "
                    "(engine seed + 1000).",
        "value": "Honest \"no learning\" baseline; expected to roll off the plate.",
        "training_method": "none — random initialisation",
        "teacher": None,
        "transferred_from": None,
    },
    "flylike_ann": {
        "kind": "learned",
        "obtained": "Sparse recurrent connectome ANN with a Dale's-law "
                    "excitatory/inhibitory topology.",
        "value": "Biological recurrent brain; the bridge that is later "
                 "transferred into the SNN.",
        "training_method": "behavioural distillation — MSE between the saturated "
                           "command and the PD label (Adam)",
        "teacher": "pid",
        "transferred_from": None,
    },
    "snn_transferred": {
        "kind": "spiking",
        "obtained": "Lossless micro-stepped integrate-and-fire SNN whose weights "
                    "are transferred from the fly-like ANN.",
        "value": "Energy-efficient spiking twin of the connectome brain.",
        "training_method": "none of its own — inherits the fly-like weights",
        "teacher": "pid",
        "transferred_from": "flylike_ann",
    },
    "dense_ann": {
        "kind": "learned",
        "obtained": "Dense feed-forward ANN.",
        "value": "Simple distilled learner; the plain ANN counterpart of PID.",
        "training_method": "behavioural distillation — MSE between the saturated "
                           "command and the PD label (Adam)",
        "teacher": "pid",
        "transferred_from": None,
    },
}

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]

#: directory holding the built frontend (``index.html`` + ``assets/``)
def _resolve_web_root() -> str:
    env = os.environ.get("ANN2SNN_WEB_ROOT", "").strip()
    if env:
        return env
    for candidate in (os.path.join(_HERE, "static"), os.path.join(_REPO_ROOT, "web", "dist")):
        if os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate
    return os.path.join(_HERE, "static")


WEB_ROOT: str = _resolve_web_root()
WEB_INDEX: str = os.path.join(WEB_ROOT, "index.html")



# --------------------------------------------------------------------------- #
# Request validation lives in server/validation.py
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Embodiment / profile parsing lives in server/validation.py
# --------------------------------------------------------------------------- #


# Engine runtime (services, weights cache, distillation jobs) lives in server/runtime.py


# --------------------------------------------------------------------------- #
# Result normalisation
# --------------------------------------------------------------------------- #
def _reference_payload(reference) -> dict:
    return to_jsonable(
        {
            "kind": "orbit",
            "steps": len(reference),
            "dt": reference.dt,
            "radius": reference.radius,
            "freq": reference.freq,
            "pos": reference.pos.tolist(),
            "vel": reference.vel.tolist(),
            "acc": reference.acc.tolist(),
        }
    )


def _run_one(controller: str, *, steps: int, radius: float, freq: float, seed: int,
             record_spikes: bool, spike_format: str, profile: str = "clean",
             embodiment: Optional[EmbodimentConfig] = None,
             example: str = "ball", trace_level: str = "short") -> dict:
    """Run a single controller and return the normalised dashboard payload."""
    svc = RUNTIME.service(seed, profile, embodiment, example)
    engine = svc.engine
    spec = get_example(example)
    canonical = engine.registry.resolve(controller)
    ctrl = engine.build_controller(canonical)

    reference = engine.reference(steps=steps, radius=radius, freq=freq)
    emb = embodiment or EmbodimentConfig()
    bc = BenchmarkConfig(steps=steps, radius=radius, freq=freq,
                         record_spikes=record_spikes, embodiment=emb,
                         example=example, trace_level=trace_level)
    init_state = torch.tensor(spec.init_state, dtype=torch.float32)

    env = None
    if emb.enable:
        env = spec.make_env(emb, dt=reference.dt)

    res = run_closed_loop(ctrl, reference=reference, init_state=init_state,
                          config=bc, name=canonical, env=env, example=example)
    result = res.to_dict(include_trace=True)
    spikes = result.pop("spikes", None)
    payload = _spike_payload(spikes, spike_format)

    out = {
        "ok": True,
        "controller": canonical,
        "label": LABELS.get(canonical, canonical),
        "trained": bool(engine.registry.trained),
        "profile": profile,
        "example": example,
        "seed": seed,
        "steps": steps,
        "radius": radius,
        "freq": freq,
        "dt": reference.dt,
        "trajectory": result["trajectory"],
        "tilts": result["tilts"],
        "error_cm": result["tracking_error"],
        "target": reference.pos.tolist(),
        "reference": _reference_payload(reference),
        "metrics": result["metrics"],
        "meta": result["meta"],
        "spike_format": spike_format,
    }
    if payload is not None:
        out["spikes"] = payload
    out["env"] = emb.to_dict() if emb.enable else None
    # base estimator traces (consistent with _run_many)
    for key in ("estimates", "measurements", "fix_events", "estimator"):
        if result.get(key) is not None:
            out[key] = result[key]
    # extended telemetry: only for the opt-in full trace level
    if trace_level == "full":
        for key in ("applied", "disturbances", "impulse_frames",
                    "measurements_by_channel", "innovations",
                    "covariance_diag", "success"):
            if result.get(key) is not None:
                out[key] = result[key]
    return out


def _run_many(controllers: Sequence[str], *, steps: int, radius: float, freq: float,
              seed: int, record_spikes: bool, spike_format: str,
              include_trace: bool, profile: str = "clean",
              embodiment: Optional[EmbodimentConfig] = None,
              example: str = "ball", trace_level: str = "short") -> dict:
    """Run several controllers on one shared reference trajectory."""
    svc = RUNTIME.service(seed, profile, embodiment, example)
    engine = svc.engine
    spec = get_example(example)
    names = [engine.registry.resolve(c) for c in controllers]

    reference = engine.reference(steps=steps, radius=radius, freq=freq)
    emb = embodiment or EmbodimentConfig()
    bc = BenchmarkConfig(steps=steps, radius=radius, freq=freq,
                         record_spikes=record_spikes, embodiment=emb,
                         example=example, trace_level=trace_level)
    init_state = torch.tensor(spec.init_state, dtype=torch.float32)

    built = {n: engine.build_controller(n) for n in names}
    report = evaluate(built, reference=reference, init_state=init_state, config=bc)
    body = report.to_dict(include_trace=include_trace)

    trained_registry = bool(engine.registry.trained)
    learned = {"dense_ann", "flylike_ann", "snn_transferred"}

    stats: Dict[str, Any] = {
        "example": example,
        "plate_half_m": PLATE_HALF_M,
        "bounds_high": list(spec.bounds_high),
        "trained": trained_registry,
        "ranking": list(body["ranking"]),
        "per_controller": {},
    }
    for name, result in body["results"].items():
        spikes = result.pop("spikes", None)
        payload = _spike_payload(spikes, spike_format)
        if payload is not None:
            result["spikes"] = payload
        result["label"] = LABELS.get(name, name)
        result["trained"] = bool(trained_registry and name in learned)
        result["spiking"] = bool(
            result.get("meta", {}).get("controller", {}).get("spiking")
        )
        # In-bounds fraction from the controller's own plant trajectory.
        traj = np.asarray(report.results[name].trajectory, dtype=float)
        on_plate = float(np.mean(spec.in_bounds(traj)) * 100.0) if traj.size else 0.0
        result["on_plate_pct"] = round(on_plate, 3)
        stats["per_controller"][name] = {
            **result["metrics"],
            "on_plate_pct": result["on_plate_pct"],
            "trained": result["trained"],
            "spiking": result["spiking"],
        }

    known = [n for n in stats["ranking"] if n in stats["per_controller"]]
    if known:
        means = [stats["per_controller"][n]["mean_error_cm"] for n in known]
        rms = [stats["per_controller"][n]["rms_error_cm"] for n in known]
        onplate = [stats["per_controller"][n]["on_plate_pct"] for n in known]
        best = min(known, key=lambda n: stats["per_controller"][n]["mean_error_cm"])
        worst = max(known, key=lambda n: stats["per_controller"][n]["mean_error_cm"])
        stats["summary"] = {
            "n": len(known),
            "best": best,
            "best_mean_error_cm": float(min(means)),
            "worst": worst,
            "worst_mean_error_cm": float(max(means)),
            "mean_of_means_cm": float(np.mean(means)),
            "mean_rms_cm": float(np.mean(rms)),
            "spread_cm": float(max(means) - min(means)),
            "mean_on_plate_pct": float(np.mean(onplate)),
            "trained": trained_registry,
        }
    else:
        stats["summary"] = {"n": 0, "trained": trained_registry}

    return {
        "ok": True,
        "trained": trained_registry,
        "profile": profile,
        "example": example,
        "env": emb.to_dict() if emb.enable else None,
        "seed": seed,
        "steps": steps,
        "radius": radius,
        "freq": freq,
        "plate_half_m": PLATE_HALF_M,
        "include_trace": include_trace,
        "reference": body["reference"],
        "init_state": body["init_state"],
        "config": body["config"],
        "ranking": body["ranking"],
        "results": body["results"],
        "stats": stats,
    }


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def _cors_origins():
    raw = os.environ.get("ANN2SNN_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        return "*"
    return origins or list(DEFAULT_CORS_ORIGINS)


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Build and configure the Flask application."""
    app = Flask(__name__)
    if config:
        app.config.update(config)

    app.config.setdefault("CORS_ORIGINS", _cors_origins())
    CORS(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})

    # ------------------------------------------------------------------ #
    # GET /api/health
    # ------------------------------------------------------------------ #
    @app.get("/api/health")
    def health():
        return jsonify(
            status="ok",
            service=APP_NAME,
            version=APP_VERSION,
            engine_version=ENGINE_VERSION,
            torch_version=TORCH_VERSION,
            controllers=CANONICAL_CONTROLLERS,
            trained=RUNTIME.trained,
            time=time.time(),
        )

    # ------------------------------------------------------------------ #
    # GET /api/controllers — catalogue + geometry for the UI
    # ------------------------------------------------------------------ #
    @app.get("/api/controllers")
    def controllers():
        try:
            description = RUNTIME.default_service().describe()
            described = description.get("controllers", [])
            engine_training = description.get("training")
        except Exception:  # pragma: no cover - defensive
            described = list_controllers()
            engine_training = None
        trained = RUNTIME.trained
        train_config = (engine_training or {}).get("config") or {}
        final_loss = (engine_training or {}).get("final_loss")
        payload = []
        for name in CANONICAL_CONTROLLERS:
            info = next((c for c in described if c.get("name") == name), {})
            info = dict(info)
            info.setdefault("name", name)
            info.setdefault("label", LABELS.get(name, name))
            model_trained = bool(trained and name in {"dense_ann", "flylike_ann", "snn_transferred"})
            info["trained"] = model_trained
            # Provenance: how it was obtained, what it is worth, how it was trained.
            guide = dict(MODEL_GUIDE.get(name, {}))
            guide["training"] = {
                "method": guide.pop("training_method", "unknown"),
                "teacher": guide.get("teacher"),
                "transferred_from": guide.get("transferred_from"),
                "epochs": train_config.get("epochs") if model_trained else None,
                "seed": train_config.get("seed") if model_trained else None,
                "final_loss": final_loss if model_trained else None,
            }
            info["guide"] = guide
            payload.append(info)
        # Report the *engine's* neuron count, which is authoritative when a
        # pre-trained bundle was loaded (it may differ from the env default).
        n_neurons = next(
            (c.get("n_neurons") for c in payload if c.get("n_neurons")),
            RUNTIME.n_neurons,
        )
        return jsonify(
            controllers=CANONICAL_CONTROLLERS,
            count=len(CANONICAL_CONTROLLERS),
            catalog=payload,
            dt=DT,
            max_tilt=MAX_TILT,
            plate_half=PLATE_HALF_M,
            radius=0.15,
            freq=0.5,
            default_steps=DEFAULT_STEPS,
            max_steps=MAX_STEPS,
            export_max_steps=EXPORT_MAX_STEPS,
            init_state=[-0.05, 0.05, 0.0, 0.0],
            reference={"kind": "orbit", "radius": 0.15, "freq": 0.5, "dt": DT},
            micro_steps=DEFAULT_MICRO_STEPS,
            n_neurons=n_neurons,
            torch_version=TORCH_VERSION,
            engine_version=ENGINE_VERSION,
            app_version=APP_VERSION,
            trained=trained,
            weights_path=RUNTIME.weights_path,
            auto_train=RUNTIME.auto_train,
            training=engine_training,
            profiles=["clean", "robust"],
            default_profile="clean",
            examples=list_examples(),
            example_names=example_names(),
            default_example=RUNTIME.default_example,
            estimator="kalman",
            estimator_note="controller input is r − x̂ (Kalman estimate from position-only measurements)",
            embodiment_presets=embodiment_specs(),
            default_embodiment_preset="embodied",
            robustness_axes=list(DEFAULT_AXIS_POINTS),
        )

    # ------------------------------------------------------------------ #
    # GET /api/engine — full engine description
    # ------------------------------------------------------------------ #
    @app.get("/api/engine")
    def engine_describe():
        return jsonify(RUNTIME.default_service().describe())

    # ------------------------------------------------------------------ #
    # POST /api/simulate — single controller (or {"controllers": [...]})
    # ------------------------------------------------------------------ #
    @app.post("/api/simulate")
    def simulate():
        try:
            body = _body()
            steps = _as_int(body.get("steps"), "steps", DEFAULT_STEPS, 1, MAX_STEPS)
            seed = _as_int(body.get("seed"), "seed", RUNTIME.default_seed, SEED_MIN, SEED_MAX)
            example, radius, freq = _example_geometry(body)
            record_spikes = _as_bool(body.get("record_spikes"), "record_spikes", True)
            spike_format = _as_spike_format(body)
            trace_level = _as_trace_level(body)
            profile = _profile_from_body(body)
            embodiment = _env_from_body(body)

            t0 = time.time()
            if "controllers" in body:
                requested = body["controllers"]
                if not isinstance(requested, (list, tuple)) or not requested:
                    raise BadRequest("'controllers' must be a non-empty list")
                if len(requested) > len(CANONICAL_CONTROLLERS):
                    raise BadRequest("too many controllers")
                out = _run_many(
                    list(requested), steps=steps, radius=radius, freq=freq, seed=seed,
                    record_spikes=record_spikes, spike_format=spike_format,
                    include_trace=_as_bool(body.get("include_trace"), "include_trace", True),
                    profile=profile, embodiment=embodiment, example=example,
                    trace_level=trace_level,
                )
            else:
                requested = body.get("controller", "pid")
                out = _run_one(
                    requested, steps=steps, radius=radius, freq=freq, seed=seed,
                    record_spikes=record_spikes, spike_format=spike_format,
                    profile=profile, embodiment=embodiment, example=example,
                    trace_level=trace_level,
                )
                out["requested_controller"] = requested
                out["engine_name"] = out["controller"]
            out["elapsed_ms"] = round((time.time() - t0) * 1000.0, 3)
            return jsonify(out)
        except BadRequest as exc:
            return jsonify(error=str(exc), controllers=CANONICAL_CONTROLLERS), 400
        except EngineError as exc:
            return jsonify(error=str(exc), controllers=CANONICAL_CONTROLLERS), 400
        except KeyError as exc:
            return jsonify(error=f"unknown controller: {exc}", controllers=CANONICAL_CONTROLLERS), 400

    # ------------------------------------------------------------------ #
    # POST /api/benchmark — multi-controller comparison
    # ------------------------------------------------------------------ #
    @app.post("/api/benchmark")
    def benchmark():
        try:
            body = _body()
            requested = body.get("controllers", list(CANONICAL_CONTROLLERS))
            if not isinstance(requested, (list, tuple)) or not requested:
                raise BadRequest("'controllers' must be a non-empty list")
            steps = _as_int(body.get("steps"), "steps", DEFAULT_STEPS, 1, MAX_STEPS)
            seed = _as_int(body.get("seed"), "seed", RUNTIME.default_seed, SEED_MIN, SEED_MAX)
            example, radius, freq = _example_geometry(body)
            record_spikes = _as_bool(body.get("record_spikes"), "record_spikes", True)
            spike_format = _as_spike_format(body)
            trace_level = _as_trace_level(body)
            profile = _profile_from_body(body)
            embodiment = _env_from_body(body)
            t0 = time.time()
            out = _run_many(
                list(requested), steps=steps, radius=radius, freq=freq, seed=seed,
                record_spikes=record_spikes, spike_format=spike_format,
                include_trace=_as_bool(body.get("include_trace"), "include_trace", True),
                profile=profile, embodiment=embodiment, example=example,
                trace_level=trace_level,
            )
            out["elapsed_ms"] = round((time.time() - t0) * 1000.0, 3)
            return jsonify(out)
        except BadRequest as exc:
            return jsonify(error=str(exc), controllers=CANONICAL_CONTROLLERS), 400
        except (EngineError, KeyError) as exc:
            return jsonify(error=f"unknown controller: {exc}", controllers=CANONICAL_CONTROLLERS), 400

    # ------------------------------------------------------------------ #
    # POST /api/robustness — environmental sweep
    # ------------------------------------------------------------------ #
    @app.post("/api/robustness")
    def robustness():
        try:
            body = _body()
            axis = str(body.get("axis", "preset")).strip().lower()
            if axis not in DEFAULT_AXIS_POINTS:
                raise BadRequest(f"'axis' must be one of: {sorted(DEFAULT_AXIS_POINTS)}")
            requested = body.get("controllers", ["pid", "flylike_ann", "snn_transferred"])
            if not isinstance(requested, (list, tuple)) or not requested:
                raise BadRequest("'controllers' must be a non-empty list")
            if len(requested) > 5:
                raise BadRequest("a sweep supports at most 5 controllers")
            steps = _as_int(body.get("steps"), "steps", DEFAULT_STEPS, 1, MAX_STEPS)
            seed = _as_int(body.get("seed"), "seed", RUNTIME.default_seed, SEED_MIN, SEED_MAX)
            profile = _profile_from_body(body)
            example, radius, freq = _example_geometry(body)
            points = body.get("points")
            if points is not None and (not isinstance(points, (list, tuple)) or len(points) > MAX_POINTS):
                raise BadRequest(f"'points' must be a list of at most {MAX_POINTS} items")
            service = RUNTIME.service(seed, profile, None, example)
            names = [service.engine.registry.resolve(c) for c in requested]
            cfg = BenchmarkConfig(steps=steps, radius=radius, freq=freq, example=example)
            t0 = time.time()
            out = robustness_sweep(
                lambda: {n: service.engine.build_controller(n) for n in names},
                config=cfg, axis=axis, points=list(points) if points else None, seed=seed,
            )
            out["profile"] = profile
            out["elapsed_ms"] = round((time.time() - t0) * 1000.0, 3)
            return jsonify(ok=True, **out)
        except BadRequest as exc:
            return jsonify(error=str(exc)), 400
        except (EngineError, KeyError) as exc:
            return jsonify(error=f"unknown controller: {exc}"), 400

    # ------------------------------------------------------------------ #
    # Training / distillation
    # ------------------------------------------------------------------ #
    @app.post("/api/train")
    def train():
        try:
            body = _body()
            epochs = _as_int(body.get("epochs"), "epochs", min(100, MAX_EPOCHS), 1, MAX_EPOCHS)
            n_neurons = body.get("n_neurons")
            n_neurons = None if n_neurons is None else _as_int(n_neurons, "n_neurons", RUNTIME.n_neurons, 2, 100000)
            seed = _as_int(body.get("seed"), "seed", RUNTIME.default_seed, SEED_MIN, SEED_MAX)
            profile = _profile_from_body(body)
            embodiment = _env_from_body(body)
            example = _example_from_body(body) or RUNTIME.default_example
            return jsonify(ok=True, **RUNTIME.start_training(
                epochs=epochs, n_neurons=n_neurons, seed=seed,
                profile=profile, embodiment=embodiment, example=example,
            ))
        except BadRequest as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/train/<job_id>")
    def train_status(job_id: str):
        job = RUNTIME.job_status(job_id)
        if job is None:
            return jsonify(error=f"no such training job: {job_id!r}"), 404
        return jsonify(ok=True, **job)

    # ------------------------------------------------------------------ #
    # POST /api/export/mp4 — render a single-controller MP4 on the server
    # ------------------------------------------------------------------ #
    @app.post("/api/export/mp4")
    def export_mp4():
        try:
            body = _body()
            steps = _as_int(body.get("steps"), "steps", DEFAULT_STEPS, 1, EXPORT_MAX_STEPS)
            seed = _as_int(body.get("seed"), "seed", RUNTIME.default_seed, SEED_MIN, SEED_MAX)
            example, radius, freq = _example_geometry(body)
            requested = body.get("controller", "pid")
            if example != "ball":
                raise BadRequest(
                    f"MP4 export currently supports the 'ball' example only (got {example!r})"
                )
            svc = RUNTIME.service(seed)
            engine = svc.engine
            canonical = engine.registry.resolve(requested)

            reference = engine.reference(steps=steps, radius=radius, freq=freq)
            bc = BenchmarkConfig(steps=steps, radius=radius, freq=freq, record_spikes=True)
            init_state = torch.tensor(engine.config.plant.init_state, dtype=torch.float32)
            res = run_closed_loop(engine.build_controller(canonical), reference=reference,
                                  init_state=init_state, config=bc, name=canonical)

            from tools.render import DEFAULT_COLORS, Rollout, render_video

            rollout = Rollout.from_result(res, name=canonical)
            label = LABELS.get(canonical, canonical)
            outdir = tempfile.mkdtemp(prefix="ann2snn_export_")
            out_path = os.path.join(outdir, f"ann2snn_{canonical}_{steps}_{seed}.mp4")
            render_video(
                rollout,
                ref_pos=reference.pos.cpu().numpy(),
                dt=reference.dt,
                radius=reference.radius,
                out_path=out_path,
                label=label,
                description=f"{steps} frames @ {1 / reference.dt:.0f} Hz · seed {seed}",
                color=DEFAULT_COLORS.get(canonical, "#1f77b4"),
            )

            @after_this_request
            def _cleanup(response):  # pragma: no cover - file lifetime
                try:
                    os.remove(out_path)
                    os.rmdir(outdir)
                except OSError:
                    pass
                return response

            return send_file(
                out_path,
                mimetype="video/mp4",
                as_attachment=True,
                download_name=os.path.basename(out_path),
            )
        except BadRequest as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503
        except (EngineError, KeyError) as exc:
            return jsonify(error=f"unknown controller: {exc}"), 400

    # ------------------------------------------------------------------ #
    # Interactive sessions
    # ------------------------------------------------------------------ #
    @app.post("/api/sessions")
    def new_session():
        try:
            body = _body()
            controller = body.get("controller", "pid")
            opts = {}
            if body.get("steps") is not None:
                opts["steps"] = _as_int(body["steps"], "steps", DEFAULT_STEPS, 1, MAX_STEPS)
            if body.get("radius") is not None:
                opts["radius"] = _as_float(body["radius"], "radius", 0.15, 0.0, 1.0)
            if body.get("freq") is not None:
                opts["freq"] = _as_float(body["freq"], "freq", 0.5, 0.0, 10.0)
            # Sessions always live in the default engine so every later
            # /step, /reset, /controller call resolves the same registry.
            info = RUNTIME.default_service().new_session(controller, **opts)
            return jsonify(ok=True, **info), 201
        except BadRequest as exc:
            return jsonify(error=str(exc)), 400
        except (EngineError, KeyError) as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/sessions/<session_id>")
    def session_info(session_id: str):
        try:
            return jsonify(ok=True, **RUNTIME.default_service().session_info(session_id))
        except EngineError as exc:
            return jsonify(error=str(exc)), 404

    @app.post("/api/sessions/<session_id>/step")
    def session_step(session_id: str):
        try:
            body = _body()
            action = body.get("action")
            if action is not None:
                if not isinstance(action, (list, tuple)) or len(action) != 2:
                    raise BadRequest("'action' must be [theta_x, theta_y]")
                action = [float(action[0]), float(action[1])]
            n = _as_int(body.get("n"), "n", 1, 1, MAX_STEPS)
            obs = RUNTIME.default_service().step(session_id, action=action, n=n)
            return jsonify(ok=True, **obs)
        except BadRequest as exc:
            return jsonify(error=str(exc)), 400
        except EngineError as exc:
            return jsonify(error=str(exc)), 404

    @app.post("/api/sessions/<session_id>/reset")
    def session_reset(session_id: str):
        try:
            return jsonify(ok=True, **RUNTIME.default_service().reset(session_id))
        except EngineError as exc:
            return jsonify(error=str(exc)), 404

    @app.put("/api/sessions/<session_id>/controller")
    def session_set_controller(session_id: str):
        try:
            body = _body()
            controller = body.get("controller")
            if not controller:
                raise BadRequest("'controller' is required")
            return jsonify(ok=True, **RUNTIME.default_service().set_controller(session_id, controller))
        except BadRequest as exc:
            return jsonify(error=str(exc)), 400
        except EngineError as exc:
            return jsonify(error=str(exc)), 404

    @app.get("/api/sessions/<session_id>/trajectory")
    def session_trajectory(session_id: str):
        try:
            return jsonify(ok=True, **RUNTIME.default_service().trajectory(session_id))
        except EngineError as exc:
            return jsonify(error=str(exc)), 404

    @app.delete("/api/sessions/<session_id>")
    def session_delete(session_id: str):
        try:
            RUNTIME.default_service().close_session(session_id)
            return jsonify(ok=True, closed=session_id)
        except EngineError as exc:
            return jsonify(error=str(exc)), 404

    # ------------------------------------------------------------------ #
    # Static dashboard bundle
    # ------------------------------------------------------------------ #
    @app.get("/")
    def dashboard_index():
        if os.path.isfile(WEB_INDEX):
            return send_from_directory(WEB_ROOT, "index.html")
        return jsonify(
            error="dashboard assets not found",
            web_root=WEB_ROOT,
            hint="build them with `cd web && npm ci && npm run build`",
            api="GET /api/controllers, POST /api/simulate, POST /api/benchmark",
        ), 404

    @app.get("/extended")
    def dashboard_extended():
        """The extended (full-information) view.

        An explicit route so the un-suffixed URL works: the catch-all below would
        otherwise fall back to ``index.html`` for an unknown path.
        """
        extended = os.path.join(WEB_ROOT, "extended.html")
        if os.path.isfile(extended):
            return send_from_directory(WEB_ROOT, "extended.html")
        return jsonify(
            error="extended dashboard not found",
            web_root=WEB_ROOT,
            hint="build it with `cd web && npm ci && npm run build`",
        ), 404

    @app.get("/<path:filename>")
    def dashboard_assets(filename: str):
        if filename.startswith("api/") or filename == "api":
            return jsonify(error="not found", path=request.path), 404
        candidate = os.path.join(WEB_ROOT, filename)
        if os.path.isfile(candidate):
            return send_from_directory(WEB_ROOT, filename)
        if os.path.isfile(WEB_INDEX):
            return send_from_directory(WEB_ROOT, "index.html")
        return jsonify(error="not found", path=request.path), 404

    # ------------------------------------------------------------------ #
    # JSON error handlers
    # ------------------------------------------------------------------ #
    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify(error="not found", path=request.path), 404

    @app.errorhandler(405)
    def method_not_allowed(_exc):
        return jsonify(error="method not allowed", path=request.path), 405

    @app.errorhandler(HTTPException)
    def http_exception(exc):
        """Client errors (including Werkzeug's WebsocketMismatch 400) as JSON.

        Without this, the broad ``Exception`` handler below would catch these
        and log a misleading "unhandled error" traceback for a mere bad request.
        """
        code = exc.code or 400
        if code >= 500:
            app.logger.exception("http error: %s", exc)
        return jsonify(
            error=exc.description or exc.name, code=code, path=request.path
        ), code

    @app.errorhandler(Exception)
    def server_error(exc):  # pragma: no cover - defensive
        app.logger.exception("unhandled error: %s", exc)
        return jsonify(error="internal server error", detail=str(exc)), 500

    return app


#: WSGI entry point
app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual dev only
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
