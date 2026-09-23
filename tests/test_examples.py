"""Example registry + drone (3-D point-mass) tests.

Also pins the ball-and-plate numbers so the dimension-generalisation cannot
silently change the original benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sim_engine import Engine
from sim_engine.config import BenchmarkConfig, EmbodimentConfig, EngineConfig, TrainingConfig
from sim_engine.estimators import KalmanFilter
from sim_engine.examples import (
    DEFAULT_EXAMPLE,
    example_names,
    get_example,
    list_examples,
)
from sim_engine.physics import C_CONST, MAX_THRUST, MAX_TILT, DT
from sim_engine.reference import lissajous_reference, setpoint_reference


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_lists_builtin_examples():
    assert DEFAULT_EXAMPLE == "ball"
    assert set(example_names()) >= {"ball", "drone"}
    catalog = {e["name"]: e for e in list_examples()}
    assert catalog["ball"]["label"] == "Balancing ball"
    assert catalog["drone"]["label"] == "Hovering drone"
    assert catalog["drone_gps_denied"]["label"] == "GPS-denied drone"


def test_unknown_example_is_rejected():
    with pytest.raises(KeyError):
        get_example("helicopter")


def test_spec_dimensions():
    ball, drone = get_example("ball"), get_example("drone")
    assert (ball.pos_dim, ball.n_in, ball.n_out) == (2, 6, 2)
    assert ball.control_limit == pytest.approx(MAX_TILT)
    assert ball.plant_gain == pytest.approx(-C_CONST)
    assert (drone.pos_dim, drone.n_in, drone.n_out) == (3, 9, 3)
    assert drone.control_limit == pytest.approx(MAX_THRUST)
    assert drone.plant_gain == pytest.approx(1.0)


def test_gps_denied_spec_is_the_drone_plant_with_onboard_sensors():
    gps = get_example("drone_gps_denied")
    drone = get_example("drone")
    # same plant/task/dims as the plain drone ...
    assert (gps.pos_dim, gps.n_in, gps.n_out) == (drone.pos_dim, drone.n_in, drone.n_out)
    assert gps.control_limit == pytest.approx(drone.control_limit)
    assert gps.plant_gain == pytest.approx(drone.plant_gain)
    assert gps.bounds_high == drone.bounds_high
    assert gps.renderer == "quad"
    assert gps.launch == (0.0,) * 6
    # ... but no absolute-position camera: an IMU-driven suite instead
    kinds = [c.kind for c in gps.sensor.channels]
    assert kinds == ["imu", "altitude", "flow_velocity", "position_fix"]
    assert gps.sensor.prediction_channel.kind == "imu"
    assert gps.sensor.fix_channel.gate_range is not None
    assert len(gps.sensor.anchors) == 3
    # the estimator fed by that suite reports a different `h`
    est = gps.make_estimator(EmbodimentConfig()).describe()
    assert est["multi_channel"] is True
    assert est["seeded_from_launch"] is True
    assert "altitude" in est["measurement"] and "position-only" not in est["measurement"]
    assert get_example("ball").make_estimator(EmbodimentConfig()).describe()[
        "measurement"
    ].startswith("position-only")


# --------------------------------------------------------------------------- #
# 3-D plant / references
# --------------------------------------------------------------------------- #
def test_point_mass_responds_to_thrust():
    spec = get_example("drone")
    state = torch.zeros(6)
    nxt = spec.step_fn(state, torch.tensor([1.0, 0.0, 0.0]),
                       dt=DT, limit=MAX_THRUST, gain=1.0, damping=0.0, disturbance=None)
    assert float(nxt[3]) == pytest.approx(DT)          # vx increased
    assert float(nxt[0]) == pytest.approx(DT * DT)     # x integrated
    assert float(nxt[4]) == pytest.approx(0.0)


def test_drone_references_are_3d_and_in_bounds():
    liss = lissajous_reference(steps=200)
    assert liss.pos_dim == 3 and liss.pos.shape == (200, 3)
    assert np.abs(np.asarray(liss.pos)).max() <= 1.0

    hover = setpoint_reference(steps=200, setpoint=(0.5, -0.3, 0.6))
    assert hover.pos_dim == 3
    assert np.allclose(np.asarray(hover.pos)[0], [0, 0, 0], atol=1e-6)
    assert np.allclose(np.asarray(hover.pos)[-1], [0.5, -0.3, 0.6], atol=1e-6)


def test_kalman_filter_in_3d_converges():
    spec = get_example("drone")
    kf = KalmanFilter(dt=DT, pos_dim=3, gain=1.0, process_noise=1e-6, meas_noise=1e-6)
    kf.reset()
    state = torch.zeros(6)
    u = torch.tensor([0.5, -0.3, 0.2])
    est = state.clone()
    for _ in range(120):
        state = spec.step_fn(state, u, dt=DT, limit=MAX_THRUST, gain=1.0,
                             damping=0.0, disturbance=None)
        est = kf.update(state[:3], u)
    assert est.shape == (6,)
    assert float((est[:3] - state[:3]).abs().max()) < 1e-2
    assert float((est[3:] - state[3:]).abs().max()) < 5e-2


# --------------------------------------------------------------------------- #
# closed loop: ball regression + drone tracking
# --------------------------------------------------------------------------- #
def test_ball_numbers_unchanged():
    """The dimension work must not move the original benchmark."""
    engine = Engine()
    res = engine.run("pid", config=BenchmarkConfig(steps=500))
    assert res.mean_error_cm == pytest.approx(1.447, abs=0.05)
    assert res.trajectory.shape[1] == 4
    assert res.measurements.shape[1] == 2


def test_drone_closed_loop_tracks():
    cfg = EngineConfig(
        example="drone", n_neurons=64, train_on_init=True,
        training=TrainingConfig(epochs=20, episodes=1, episode_steps=80,
                                coverage_samples=512, connectome_unroll=2, log_every=1000),
    )
    engine = Engine(cfg)
    b = BenchmarkConfig(example="drone", steps=150,
                        embodiment=EmbodimentConfig.from_preset("clean"))
    res = engine.run("pid", config=b)
    assert res.trajectory.shape == (150, 6)
    assert res.measurements.shape == (150, 3)
    assert res.estimates.shape == (150, 6)
    # lissajous amplitude is 60 cm; a working controller stays well inside that
    assert res.mean_error_cm < 30.0
    assert res.trajectory.shape[1] == 6
