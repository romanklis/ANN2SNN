"""Onboard sensor suites: channels, resolution, delay and the observability ladder.

The GPS-denied drone is the first example whose estimator is *not* fed an
absolute-position camera.  These tests pin the pieces that make that honest:

* the channel/spec validation and how the embodiment config resolves onto it,
* per-channel latency (including a late-arriving checkpoint fix),
* the accelerometer as the prediction input,
* the observability ladder (inertial-only drifts, flow bounds it, fixes improve it),
* and that ``ball`` / ``drone`` still run the historical camera unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from sim_engine.benchmark import run_closed_loop
from sim_engine.config import BenchmarkConfig, EmbodimentConfig
from sim_engine.controllers import ClassicalPDController
from sim_engine.estimators import KalmanFilter
from sim_engine.environment import EmbodiedEnv
from sim_engine.examples import get_example
from sim_engine.physics import DT
from sim_engine.sensors import (
    ALTITUDE,
    FIX,
    FLOW,
    IMU,
    ResolvedChannel,
    SensorChannel,
    SensorModel,
    SensorReadings,
    SensorSpec,
    altitude_channel,
    camera_channel,
    checkpoint_channel,
    flow_channel,
    imu_channel,
    selection_matrix,
)

GPS = "drone_gps_denied"


# --------------------------------------------------------------------------- #
# channels and specs
# --------------------------------------------------------------------------- #
def test_selection_matrix():
    assert selection_matrix((0, 2), 4) == [[1, 0, 0, 0], [0, 0, 1, 0]]
    with pytest.raises(ValueError):
        selection_matrix((4,), 4)


def test_channel_validation():
    with pytest.raises(ValueError):
        SensorChannel(kind="lidar", axes=(0,))
    with pytest.raises(ValueError):
        SensorChannel(kind=ALTITUDE, axes=())
    with pytest.raises(ValueError):
        SensorChannel(kind=ALTITUDE, axes=(0, 0))
    with pytest.raises(ValueError):
        SensorChannel(kind=ALTITUDE, axes=(0,), base_noise=-1.0)
    with pytest.raises(ValueError):
        SensorChannel(kind=ALTITUDE, axes=(0,), latency=-1)
    with pytest.raises(ValueError):
        SensorChannel(kind=IMU, axes=(0,), gate_range=0.5)


def test_spec_validation():
    baro = altitude_channel(2, base_noise=0.01)
    with pytest.raises(ValueError):
        SensorSpec(channels=(baro, baro))                       # duplicate kind
    with pytest.raises(ValueError):
        SensorSpec(channels=(checkpoint_channel((0,), base_noise=0.1,
                                                gate_range=0.3),))  # no anchors
    with pytest.raises(ValueError):
        SensorSpec(channels=(altitude_channel(9, base_noise=0.01),)).validate(3)
    # nothing observes position and there is no launch point -> unobservable
    with pytest.raises(ValueError):
        SensorSpec(channels=(flow_channel((3, 4), base_noise=0.01),)).validate(3)
    # ... but a launch point anchors it
    SensorSpec(
        channels=(flow_channel((3, 4), base_noise=0.01),), launch=(0.0,) * 6
    ).validate(3)


def test_resolve_applies_embodiment_degradation():
    spec = SensorSpec(
        channels=(
            imu_channel(3, base_noise=0.05, latency=0),
            altitude_channel(2, base_noise=0.02, latency=2),
            flow_channel((3, 4), base_noise=0.03, latency=3),
            checkpoint_channel((0, 1, 2), base_noise=0.05, latency=5,
                               gate_range=0.4),
        ),
        anchors=((1.0, 0.0, 0.0),),
        launch=(0.0,) * 6,
    )
    clean = spec.resolve(EmbodimentConfig(), 3)
    assert clean.has_imu and clean.imu.latency == 0
    assert clean.max_latency == 5
    assert clean.measurement_dim == 6          # baro 1 + flow 2 + fix 3

    cfg = EmbodimentConfig(sensor_noise_scale=3.0, sensor_delay=2)
    noisy = spec.resolve(cfg, 3)
    assert noisy.imu.sigma == pytest.approx(0.15)
    assert noisy.channel(ALTITUDE).sigma == pytest.approx(0.06)
    assert noisy.channel(FLOW).latency == 5
    assert noisy.channel(FIX).latency == 7


def test_camera_channel_uses_the_config_values():
    """The historical channel takes its noise/latency from the preset."""
    spec = SensorSpec(channels=(camera_channel(2),))
    clean = spec.resolve(EmbodimentConfig(), 2)
    assert clean.imu is None
    assert clean.measurement_dim == 2
    assert clean.camera_channel is not None

    noisy = spec.resolve(EmbodimentConfig(sensor_noise_pos=0.005), 2)
    assert noisy.channel(FIX).sigma == pytest.approx(0.005)
    delayed = spec.resolve(EmbodimentConfig(sensor_delay=3), 2)
    assert delayed.channel(FIX).latency == 3
    # the noise-scale knob deliberately does not touch the camera channel
    scaled = spec.resolve(EmbodimentConfig(sensor_noise_scale=5.0), 2)
    assert scaled.channel(FIX).sigma == 0.0


# --------------------------------------------------------------------------- #
# environment sampling
# --------------------------------------------------------------------------- #
def _gps_env(embodiment=None, **spec_kwargs):
    sensor = get_example(GPS).sensor
    if spec_kwargs:
        sensor = dataclasses.replace(sensor, **spec_kwargs)
    spec = dataclasses.replace(get_example(GPS), sensor=sensor)
    return spec.make_env(embodiment or EmbodimentConfig())


def test_measurement_stream_is_deterministic():
    def run():
        env = _gps_env(EmbodimentConfig(sensor_noise_scale=2.0))
        env.reset(seed=5)
        state = torch.zeros(6)
        out = []
        for _ in range(30):
            r = env.sense(state)
            out.append(np.concatenate([
                r.imu.numpy(), r.channels[ALTITUDE].numpy(), r.channels[FLOW].numpy(),
            ]))
            state = env.step(state, torch.zeros(3), None)
        return np.array(out)

    a, b = run(), run()
    assert np.allclose(a, b)
    assert np.abs(a).sum() > 0                       # the sensors really are noisy


def test_altitude_channel_is_delayed_by_its_latency():
    sensor = SensorSpec(
        channels=(camera_channel(3), altitude_channel(2, base_noise=0.0, latency=2)),
        launch=(0.0,) * 6,
    )
    spec = dataclasses.replace(get_example(GPS), sensor=sensor)
    env = spec.make_env(EmbodimentConfig())
    env.reset(seed=0)
    trace = []
    for k in range(6):
        state = torch.tensor([0.0, 0.0, float(k), 0.0, 0.0, 0.0])
        trace.append(float(env.sense(state).channels[ALTITUDE][0]))
    assert trace == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_imu_reports_the_achieved_acceleration():
    sensor = SensorSpec(
        channels=(imu_channel(3, base_noise=0.0, latency=0), camera_channel(3)),
        launch=(0.0,) * 6,
    )
    spec = dataclasses.replace(get_example(GPS), sensor=sensor)
    env = spec.make_env(EmbodimentConfig())
    env.reset(seed=0)
    state = torch.zeros(6)
    u = torch.tensor([1.0, -0.5, 0.25])
    first = env.sense(state).imu
    assert torch.allclose(first, torch.zeros(3))      # nothing has happened yet
    nxt = env.step(state, u, 0)
    achieved = env.sense(nxt).imu
    assert torch.allclose(achieved, (nxt[3:] - state[3:]) / DT)
    assert torch.allclose(achieved, u, atol=1e-6)


def test_checkpoint_fix_is_relative_to_its_anchor():
    anchors = ((0.5, 0.0, 0.0),)
    sensor = SensorSpec(
        channels=(checkpoint_channel((0, 1, 2), base_noise=0.0, latency=0,
                                     gate_range=0.4),),
        anchors=anchors,
        launch=(0.0,) * 6,
    )
    spec = dataclasses.replace(get_example(GPS), sensor=sensor)
    env = spec.make_env(EmbodimentConfig())
    env.reset(seed=0)
    state = torch.tensor([0.6, 0.1, 0.2, 0.0, 0.0, 0.0])
    readings = env.sense(state)
    assert FIX in readings.channels
    assert torch.allclose(readings.channels[FIX], state[:3] - torch.tensor(anchors[0]))
    assert torch.allclose(readings.anchors[FIX], torch.tensor(anchors[0]))
    assert env.fix_count == 1 and env.last_fix_new
    # out of range -> no sample, no new acquisition
    env.sense(torch.tensor([-0.9, 0.0, 0.2, 0.0, 0.0, 0.0]))
    assert env.fix_count == 1
    assert not env.last_fix_new


# --------------------------------------------------------------------------- #
# estimator: prediction input, latency and out-of-order corrections
# --------------------------------------------------------------------------- #
def test_accelerometer_drives_the_prediction():
    """A measured acceleration predicts better than the commanded control."""
    D = 1
    sensor = SensorModel(
        channels=(ResolvedChannel(FIX, (0,), 0.0, 0),),
        imu=ResolvedChannel(IMU, (0,), 0.0, 0),
        launch=(0.0, 0.0),
        pos_dim=D,
    )
    kf = KalmanFilter(dt=DT, pos_dim=D, gain=1.0, sensor=sensor)
    kf.reset()
    # commanded 0.5, but the body achieved 2.0 (actuator gain 4x)
    step = torch.tensor([0.5])
    achieved = torch.tensor([2.0])
    n = 4
    for k in range(n):
        readings = SensorReadings(imu=achieved, channels={})
        kf.update(readings=readings, control=step, acceleration=readings.imu)
    # The launch seed anchors index 0 (frame 0), so the first update folds into
    # index 0 and only the remaining n-1 updates predict: v = a·m·dt and
    # p = a·dt²·m(m+1)/2 for the semi-implicit integrator, m = n - 1.
    m = n - 1
    assert float(kf.x[1]) == pytest.approx(2.0 * DT * m, rel=1e-6)
    assert float(kf.x[0]) == pytest.approx(2.0 * DT ** 2 * m * (m + 1) / 2.0, rel=1e-6)


def test_late_fix_is_applied_at_its_own_index():
    """A slow checkpoint fix lands at the index it refers to, after faster channels."""
    D = 1
    sensor = SensorModel(
        channels=(
            ResolvedChannel(FLOW, (1,), 0.0, 0),          # fast velocity channel
            ResolvedChannel(FIX, (0,), 0.0, 3, gate_range=0.5),
        ),
        launch=(0.0, 0.0),
        pos_dim=D,
    )
    kf = KalmanFilter(dt=DT, pos_dim=D, gain=1.0, sensor=sensor)
    kf.reset()
    for k in range(6):
        # v = 0 every frame; the position is unobserved until the fix arrives
        readings = SensorReadings(channels={FLOW: torch.zeros(1)})
        if k == 5:
            # a fix measured relative to an anchor at 1.0, seen at control frame 5,
            # so it refers to frame/index 2 (latency 3; index == control frame)
            readings.channels[FIX] = torch.tensor([0.0])
            readings.anchors[FIX] = torch.tensor([1.0])
        kf.update(readings=readings, control=torch.zeros(1))
    assert kf.describe()["fix_frames"] == [2]
    # the late fix is folded in at index 2 and survives: the faster channel's
    # corrections at later indices are re-applied on top of it, not instead of it
    assert float(kf.x[0]) == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------- #
# the GPS-denied example end to end
# --------------------------------------------------------------------------- #
def _run(channels, *, steps=500, profile="clean", seed=42, anchors=None):
    base = get_example(GPS)
    spec = dataclasses.replace(
        base,
        sensor=SensorSpec(
            channels=tuple(channels),
            anchors=tuple(anchors if anchors is not None else base.sensor.anchors),
            launch=(0.0,) * 6,
        ),
    )
    emb = EmbodimentConfig.from_preset(profile)
    emb.seed = seed
    env = spec.make_env(emb)
    bc = BenchmarkConfig(steps=steps, example=GPS, embodiment=emb)
    ctrl = ClassicalPDController(pos_dim=3, n_out=3, plant_gain=1.0, action_limit=3.0)
    return run_closed_loop(
        ctrl, reference=spec.reference(steps=steps), config=bc, env=env, example=GPS
    )


def _pos_error(res):
    return np.linalg.norm(np.asarray(res.estimates)[:, :3] - res.trajectory[:, :3], axis=1) * 100


def test_gps_denied_runs_and_fixes_a_few_times():
    res = _run([
        imu_channel(3, base_noise=0.05, latency=0),
        altitude_channel(2, base_noise=0.02, latency=2),
        flow_channel((3, 4), base_noise=0.03, latency=3),
        checkpoint_channel((0, 1, 2), base_noise=0.05, latency=5, gate_range=0.45),
    ])
    assert res.on_plate_pct is not None and res.on_plate_pct > 90.0
    assert res.measurements is None                        # no position camera
    assert res.fix_events and 4 <= len(res.fix_events) <= 25
    assert res.estimation_first_fix_step is not None
    assert len(res.estimation_pos_rmse_cm_axes) == 3
    assert res.estimation_pos_rmse_cm > 0.0
    assert res.estimation_max_pos_err_cm >= res.estimation_pos_rmse_cm
    assert res.estimator["multi_channel"] is True
    assert res.estimator["sensor"]["has_imu"] is True


def test_observability_ladder():
    """Inertial-only drifts, optical flow bounds it, checkpoint fixes improve it."""
    imu = imu_channel(3, base_noise=0.05, latency=0)
    baro = altitude_channel(2, base_noise=0.02, latency=2)
    flow = flow_channel((3, 4), base_noise=0.03, latency=3)
    fix = checkpoint_channel((0, 1, 2), base_noise=0.05, latency=5, gate_range=0.45)

    inertial = _run([imu, baro])
    flowed = _run([imu, baro, flow])
    anchored = _run([imu, baro, flow, fix])

    rmse = [r.estimation_pos_rmse_cm for r in (inertial, flowed, anchored)]
    late = [float(np.mean(_pos_error(r)[-200:])) for r in (inertial, flowed, anchored)]

    # without a position-observing channel the horizontal estimate drifts away
    assert rmse[0] > rmse[1] > rmse[2]
    assert late[0] > late[1] > late[2]
    assert late[1] < 4.0                     # flow keeps the estimate bounded
    assert late[2] < late[1]                 # fixes improve the anchored regime
    assert not inertial.fix_events and not flowed.fix_events
    assert anchored.fix_events


def test_ball_and_drone_keep_the_position_camera():
    for name, pos_dim in (("ball", 2), ("drone", 3)):
        spec = get_example(name)
        sensor = spec.sensor.resolve(EmbodimentConfig(), pos_dim)
        assert sensor.imu is None
        assert len(sensor.channels) == 1
        assert sensor.camera_channel is not None
        assert sensor.measurement_dim == pos_dim
        env = spec.make_env(EmbodimentConfig())
        y = env.measure(torch.zeros(2 * pos_dim))
        assert y.shape == (pos_dim,)


def test_ball_measurements_are_still_recorded():
    from sim_engine.engine import Engine

    res = Engine().run("pid", config=BenchmarkConfig(steps=120))
    assert res.measurements is not None
    assert res.measurements.shape == (120, 2)
    assert res.estimates is not None and res.estimates.shape == (120, 4)
    assert res.fix_events is None
