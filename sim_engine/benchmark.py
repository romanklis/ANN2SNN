"""Closed-loop trajectory benchmark.

Drives any :class:`~sim_engine.controllers.base.BaseController` through the
circular-orbit tracking task and reports the same *mean radial tracking error*
(in centimetres) that the original prototype printed, plus per-step traces that
a frontend can animate.

The loop is deliberately tiny and controller-agnostic::

    for k in range(steps):
        record(state)
        tilt = controller.act(state, reference.at(k))
        state = step_physics(state, tilt)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import BenchmarkConfig
from .controllers.base import BaseController
from .environment import EmbodiedEnv
from .estimators import KalmanFilter
from .examples import get_example
from .physics import DT, step_physics
from .reference import Reference, orbit_reference
from .serialization import to_jsonable

__all__ = [
    "TrajectoryResult",
    "BenchmarkReport",
    "run_closed_loop",
    "mean_radial_error_cm",
    "evaluate",
    "evaluate_many",
]


@dataclass
class TrajectoryResult:
    """Per-controller closed-loop trace + metrics."""

    name: str
    trajectory: np.ndarray            # (T, 2D) plant states
    tilts: np.ndarray                 # (T, D) applied actuator commands
    tracking_error: np.ndarray        # (T,) instantaneous radial error [m]
    mean_error_cm: float
    final_error_cm: float
    rms_error_cm: float
    max_error_cm: float
    settling_step: Optional[int] = None
    spikes: Optional[np.ndarray] = None   # (T, N)
    meta: dict = field(default_factory=dict)
    #: embodiment metrics (present only when the run used an EmbodiedEnv)
    on_plate_pct: Optional[float] = None
    impulse_count: Optional[int] = None
    recovery_steps: Optional[list] = None
    max_recovery_step: Optional[int] = None
    disturbance_rms: Optional[float] = None
    env: Optional[dict] = None
    #: estimation (sensor samples -> Kalman estimate) traces + errors
    measurements: Optional[np.ndarray] = None   # (T, D) absolute-position camera
    estimates: Optional[np.ndarray] = None      # (T, 2D) x̂ fed to the controller
    estimation_pos_rmse_cm: Optional[float] = None
    estimation_vel_rmse: Optional[float] = None
    #: per-axis position RMSE [cm], worst per-frame error [cm] and the first
    #: checkpoint fix (multi-channel sensor suites)
    estimation_pos_rmse_cm_axes: Optional[list] = None
    estimation_max_pos_err_cm: Optional[float] = None
    estimation_first_fix_step: Optional[int] = None
    fix_events: Optional[list] = None
    estimator: Optional[dict] = None
    #: ------------------------------------------------------------------ #
    #: extended telemetry, populated only for ``trace_level == "full"``
    #: (the extended dashboard's per-frame inspector)
    #: ------------------------------------------------------------------ #
    trace_level: str = "short"
    applied: Optional[np.ndarray] = None            # (T, D) actuated/limited command
    disturbances: Optional[np.ndarray] = None       # (T, D) equivalent disturbance accel
    impulse_frames: Optional[list] = None           # velocity-kick frames
    measurements_by_channel: Optional[dict] = None  # {kind: [ (dim,) | None ]}
    innovations: Optional[dict] = None              # {kind: [ (dim,) | None ]}
    covariance_diag: Optional[list] = None           # [ (2D,) | None ] posterior diag(P)
    success: Optional[list] = None                  # (T,) in-bounds flags

    def to_dict(self, include_trace: bool = True) -> dict:
        d = {
            "name": self.name,
            "metrics": {
                "mean_error_cm": self.mean_error_cm,
                "rms_error_cm": self.rms_error_cm,
                "max_error_cm": self.max_error_cm,
                "final_error_cm": self.final_error_cm,
                "settling_step": self.settling_step,
            },
            "meta": dict(self.meta),
        }
        if self.on_plate_pct is not None:
            d["metrics"]["on_plate_pct"] = self.on_plate_pct
        if self.impulse_count is not None:
            d["metrics"]["impulse_count"] = self.impulse_count
        if self.max_recovery_step is not None:
            d["metrics"]["max_recovery_step"] = self.max_recovery_step
        if self.disturbance_rms is not None:
            d["metrics"]["disturbance_rms"] = self.disturbance_rms
        if self.estimation_pos_rmse_cm is not None:
            d["metrics"]["estimation_pos_rmse_cm"] = self.estimation_pos_rmse_cm
        if self.estimation_vel_rmse is not None:
            d["metrics"]["estimation_vel_rmse"] = self.estimation_vel_rmse
        if self.estimation_pos_rmse_cm_axes is not None:
            d["metrics"]["estimation_pos_rmse_cm_axes"] = list(self.estimation_pos_rmse_cm_axes)
        if self.estimation_max_pos_err_cm is not None:
            d["metrics"]["estimation_max_pos_err_cm"] = self.estimation_max_pos_err_cm
        if self.estimation_first_fix_step is not None:
            d["metrics"]["estimation_first_fix_step"] = self.estimation_first_fix_step
        if self.fix_events is not None:
            d["fix_events"] = list(self.fix_events)
        if self.recovery_steps is not None:
            d["recovery_steps"] = list(self.recovery_steps)
        if self.env is not None:
            d["env"] = dict(self.env)
        if self.estimator is not None:
            d["estimator"] = dict(self.estimator)
        if include_trace:
            d["trajectory"] = self.trajectory.tolist()
            d["tilts"] = self.tilts.tolist()
            d["tracking_error"] = self.tracking_error.tolist()
            if self.measurements is not None:
                d["measurements"] = self.measurements.tolist()
            if self.estimates is not None:
                d["estimates"] = self.estimates.tolist()
            if self.trace_level == "full":
                # extended telemetry: per-frame estimator internals + environment
                # side-channels.  ``None`` entries mark frames without a sample
                # (e.g. a checkpoint that was not in range); `to_jsonable` turns
                # them into JSON null.
                if self.applied is not None:
                    d["applied"] = self.applied.tolist()
                if self.disturbances is not None:
                    d["disturbances"] = self.disturbances.tolist()
                if self.impulse_frames is not None:
                    d["impulse_frames"] = list(self.impulse_frames)
                if self.measurements_by_channel is not None:
                    d["measurements_by_channel"] = dict(self.measurements_by_channel)
                if self.innovations is not None:
                    d["innovations"] = dict(self.innovations)
                if self.covariance_diag is not None:
                    d["covariance_diag"] = list(self.covariance_diag)
                if self.success is not None:
                    d["success"] = [bool(v) for v in self.success]
        if self.spikes is not None:
            d["spikes"] = self.spikes.tolist()
        return to_jsonable(d)


@dataclass
class BenchmarkReport:
    """Aggregate report over several controllers on one reference trajectory."""

    reference: Reference
    results: Dict[str, TrajectoryResult]
    init_state: tuple
    config: dict

    @property
    def ranking(self) -> List[str]:
        """Controller names ordered from best (lowest mean error) to worst."""
        return sorted(self.results, key=lambda n: self.results[n].mean_error_cm)

    def to_dict(self, include_trace: bool = True) -> dict:
        return to_jsonable(
            {
                "reference": {
                    "kind": "orbit",
                    "steps": len(self.reference),
                    "dt": self.reference.dt,
                    "radius": self.reference.radius,
                    "freq": self.reference.freq,
                    "pos": self.reference.pos.tolist(),
                    "vel": self.reference.vel.tolist(),
                    "acc": self.reference.acc.tolist(),
                },
                "init_state": list(self.init_state),
                "config": self.config,
                "ranking": self.ranking,
                "results": {
                    name: res.to_dict(include_trace=include_trace)
                    for name, res in self.results.items()
                },
            }
        )


def mean_radial_error_cm(
    trajectory: np.ndarray,
    reference: Reference,
) -> np.ndarray:
    """Instantaneous tracking error ``‖p - p_ref‖`` in centimetres (any dim)."""
    d = int(reference.pos.shape[1])
    T = min(len(trajectory), len(reference))
    ref = reference.pos.cpu().numpy()[:T, :d]
    return np.sqrt(np.sum((trajectory[:T, :d] - ref) ** 2, axis=1)) * 100.0


def run_closed_loop(
    controller: BaseController,
    reference: Optional[Reference] = None,
    init_state: Optional[torch.Tensor] = None,
    config: Optional[BenchmarkConfig] = None,
    *,
    record_spikes: Optional[bool] = None,
    name: Optional[str] = None,
    env: Optional[EmbodiedEnv] = None,
    estimator: Optional[KalmanFilter] = None,
    example: Optional[str] = None,
) -> TrajectoryResult:
    """Roll the plant + controller forward over the whole reference trajectory.

    When ``env`` is given the controller only sees ``env.measure(state)`` and its
    command passes through ``env.actuate`` / ``env.step``; the recorded
    ``trajectory`` is always the **true** plant state.  With ``env=None`` an
    idealized (no sensing/no actuator) loop runs, used by unit tests.

    ``example`` selects the task family (``ball``/``drone``) for the default
    reference, initial state and error metric.
    """
    config = config or BenchmarkConfig()
    spec = get_example(example or config.example)
    if reference is None:
        reference = spec.reference(steps=config.steps, radius=config.radius,
                                   freq=config.freq)
    steps = min(config.steps, len(reference))
    if record_spikes is None:
        record_spikes = config.record_spikes

    device = getattr(controller, "device", torch.device("cpu"))
    if init_state is None:
        init_state = torch.tensor(spec.init_state, device=device)
    state = torch.as_tensor(init_state, dtype=torch.float32, device=device).clone()

    use_env = env is not None
    kf: Optional[KalmanFilter] = None
    if use_env:
        env.reset()
        # The example owns the sensor model, so the estimator must be built from
        # the spec (not inline) or a multi-channel suite would be ignored.
        kf = estimator or spec.make_estimator(env.config, dt=reference.dt, device=device)
        kf.reset()

    controller.reset()

    traj: List[np.ndarray] = []
    tilts: List[np.ndarray] = []
    spikes: List[np.ndarray] = []
    disturbances: List[np.ndarray] = []
    impulse_frames: List[int] = []
    measurements: List[np.ndarray] = []
    estimates: List[np.ndarray] = []
    fix_events: List[dict] = []
    wants_spikes = bool(record_spikes and getattr(controller, "spiking", False))
    u_prev = torch.zeros(spec.control_dim, dtype=state.dtype, device=device)
    cam_kind = None
    if use_env and env.sensor.camera_channel is not None:
        cam_kind = env.sensor.camera_channel.kind

    # ---- extended telemetry (trace_level == "full") ------------------------ #
    full = str(getattr(config, "trace_level", "short")) == "full"
    channel_kinds = [ch.kind for ch in env.sensor.channels] if use_env else []
    applied: List[np.ndarray] = []
    channel_frames: List[dict] = []       # kind -> (dim,) sample, per frame

    with torch.no_grad():
        for k in range(steps):
            traj.append(state.detach().cpu().numpy().copy())
            ref_k = reference.at(k)
            if use_env:
                readings = env.sense(state)            # IMU + delayed channels
                xhat = kf.update(                      # KF: full-state estimate
                    readings=readings, control=u_prev, acceleration=readings.imu,
                )
                if cam_kind is not None and cam_kind in readings.channels:
                    measurements.append(
                        readings.channels[cam_kind].detach().cpu().numpy().copy()
                    )
                if full:
                    channel_frames.append({
                        kind: val.detach().cpu().numpy().copy()
                        for kind, val in readings.channels.items()
                    })
                if env.last_fix_new:
                    fix_events.append({
                        "k": int(k),
                        "pos": state[:spec.pos_dim].detach().cpu().numpy().tolist(),
                        "anchor": (
                            None if env.last_fix_anchor is None
                            else env.last_fix_anchor.detach().cpu().numpy().tolist()
                        ),
                    })
                estimates.append(xhat.detach().cpu().numpy().copy())
                u = controller.act(xhat, ref_k)        # policy acts on r − x̂
            else:
                u = controller.act(state, ref_k)
            tilts.append(u.detach().cpu().numpy().copy())
            if wants_spikes:
                spk = controller.last_spikes()
                if spk is not None:
                    spikes.append(spk.detach().cpu().numpy().copy())
            if use_env:
                u_eff = env.actuate(u)
                if full:
                    applied.append(u_eff.detach().cpu().numpy().copy())
                state = env.step(state, u_eff, k)
                disturbances.append(env.last_disturbance.detach().cpu().numpy().copy())
                if env.last_impulse:
                    impulse_frames.append(k)
            else:
                if full:
                    # no body model: the commanded control is what the plant sees
                    applied.append(u.detach().cpu().numpy().copy())
                state = spec.step_fn(
                    state, u, dt=reference.dt, limit=spec.control_limit,
                    gain=spec.plant_gain, damping=0.0, disturbance=None,
                )
            u_prev = u

    trajectory = np.asarray(traj)
    tilt_arr = np.asarray(tilts)
    err = spec.tracking_error_cm(trajectory, reference)

    # Settling: first index after which the error never again exceeds 2x its
    # steady-state (last-decile) median.
    steady = float(np.median(err[int(0.9 * len(err)):])) if len(err) else 0.0
    tol = max(2.0 * steady, 0.5)  # cm
    settling = None
    for i in range(len(err)):
        if np.all(err[i:] <= tol):
            settling = i
            break

    on_plate_pct = impulse_count = recovery = max_recovery = drms = env_meta = None
    est_pos_rmse = est_vel_rmse = None
    est_axes = est_max = first_fix = None
    inside = spec.in_bounds(trajectory)
    if use_env:
        d = spec.pos_dim
        on_plate_pct = float(100.0 * np.mean(inside)) if len(inside) else 0.0

        recovery = []
        for f in impulse_frames:
            rec = None
            for j in range(f, len(err)):
                if err[j] <= tol:
                    rec = int(j - f)
                    break
            recovery.append(rec)
        valid = [r for r in recovery if r is not None]
        max_recovery = int(max(valid)) if valid else None
        impulse_count = int(len(impulse_frames))
        if disturbances:
            da = np.asarray(disturbances)
            drms = float(np.sqrt(np.mean(np.sum(da ** 2, axis=1))))
        env_meta = env.describe()
        if fix_events:
            first_fix = int(fix_events[0]["k"])

        # estimation error: x̂ (what the controller saw) vs the true state
        if estimates:
            ea = np.asarray(estimates)
            T = min(len(err), len(ea))
            pos_err = np.linalg.norm(ea[:T, :d] - trajectory[:T, :d], axis=1) * 100.0
            vel_err = np.linalg.norm(ea[:T, d:2 * d] - trajectory[:T, d:2 * d], axis=1)
            est_pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))
            est_vel_rmse = float(np.sqrt(np.mean(vel_err ** 2)))
            est_axes = [
                float(np.sqrt(np.mean((ea[:T, i] - trajectory[:T, i]) ** 2))) * 100.0
                for i in range(d)
            ]
            est_max = float(np.max(pos_err))
            if fix_events and first_fix is not None and first_fix < len(pos_err):
                # drift *after* the first checkpoint: the pre-fix transient is the
                # launch-pad initialisation, not a localisation failure.
                est_max = float(np.max(pos_err[first_fix:]))

    return TrajectoryResult(
        name=name or controller.name,
        trajectory=trajectory,
        tilts=tilt_arr,
        tracking_error=err,
        mean_error_cm=float(np.mean(err)) if len(err) else float("nan"),
        rms_error_cm=float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        max_error_cm=float(np.max(err)) if len(err) else float("nan"),
        final_error_cm=float(err[-1]) if len(err) else float("nan"),
        settling_step=settling,
        spikes=np.asarray(spikes) if spikes else None,
        meta={"controller": controller.describe()},
        on_plate_pct=on_plate_pct,
        impulse_count=impulse_count,
        recovery_steps=recovery,
        max_recovery_step=max_recovery,
        disturbance_rms=drms,
        env=env_meta,
        measurements=np.asarray(measurements) if measurements else None,
        estimates=np.asarray(estimates) if estimates else None,
        estimation_pos_rmse_cm=est_pos_rmse,
        estimation_vel_rmse=est_vel_rmse,
        estimation_pos_rmse_cm_axes=est_axes,
        estimation_max_pos_err_cm=est_max,
        estimation_first_fix_step=first_fix,
        fix_events=fix_events or None,
        estimator=kf.describe() if kf is not None else None,
        trace_level=str(getattr(config, "trace_level", "short")),
        applied=np.asarray(applied) if applied else None,
        disturbances=np.asarray(disturbances) if (full and disturbances) else None,
        impulse_frames=list(impulse_frames) if full else None,
        measurements_by_channel=(
            {kind: [frame.get(kind) for frame in channel_frames] for kind in channel_kinds}
            if full and channel_kinds else None
        ),
        innovations=(
            # index == control frame, and a sample only becomes available at
            # ``index + latency``, so read the filter's map once the run is done.
            {
                kind: [
                    None if kf.innovation(i, kind) is None
                    else kf.innovation(i, kind).detach().cpu().numpy().copy()
                    for i in range(len(trajectory))
                ]
                for kind in channel_kinds
            }
            if full and channel_kinds and kf is not None else None
        ),
        covariance_diag=(
            # like the innovations: read the filter's map once the run is done, so
            # an index corrected only when a delayed sample arrived shows its
            # final posterior rather than the stale pre-correction snapshot.
            [
                None if kf.covariance_diag(i) is None
                else kf.covariance_diag(i).detach().cpu().numpy().copy()
                for i in range(len(trajectory))
            ]
            if full and kf is not None else None
        ),
        success=[bool(v) for v in inside] if full else None,
    )


def evaluate(
    controllers: Dict[str, BaseController],
    reference: Optional[Reference] = None,
    init_state: Optional[torch.Tensor] = None,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkReport:
    """Evaluate several named controllers on a **shared** reference trajectory."""
    config = config or BenchmarkConfig()
    spec = get_example(config.example)
    if reference is None:
        reference = spec.reference(
            steps=config.steps, radius=config.radius, freq=config.freq, device="cpu",
        )
    if init_state is None:
        init_state = torch.tensor(spec.init_state)

    embodiment = getattr(config, "embodiment", None)
    # The estimator is always on: even a "clean" run estimates velocity from
    # position-only measurements (zero noise/delay), so the controller never sees
    # the true state.
    use_env = bool(embodiment is not None and embodiment.enable)
    init_tuple = tuple(float(v) for v in init_state)

    results: Dict[str, TrajectoryResult] = {}
    for name, ctrl in controllers.items():
        env = None
        if use_env:
            # A fresh env per controller, same seed -> identical disturbances.
            env = spec.make_env(embodiment, dt=reference.dt)
        results[name] = run_closed_loop(
            ctrl, reference=reference, init_state=init_state, config=config,
            name=name, env=env,
        )

    return BenchmarkReport(
        reference=reference,
        results=results,
        init_state=init_tuple,
        config=config.to_dict(),
    )


def evaluate_many(*args, **kwargs) -> BenchmarkReport:  # pragma: no cover - alias
    """Alias for :func:`evaluate`, kept for call-site readability."""
    return evaluate(*args, **kwargs)
