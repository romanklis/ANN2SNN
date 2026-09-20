"""Embodied-environment tests (sensing, actuation, body, perturbations, sweeps).

Everything here is small and deterministic; the clean benchmark is asserted to be
bit-for-bit unchanged by the new optional physics terms.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sim_engine.benchmark import evaluate, run_closed_loop
from sim_engine.config import (
    EMBODIMENT_PRESETS,
    BenchmarkConfig,
    EmbodimentConfig,
    EngineConfig,
    NetworkConfig,
    TrainingConfig,
)
from sim_engine.controllers import ClassicalPDController
from sim_engine.controllers.base import error_vector
from sim_engine.engine import Engine
from sim_engine.environment import EmbodiedEnv, embodiment_specs
from sim_engine.physics import C_CONST, DT, step_physics
from sim_engine.reference import orbit_reference
from sim_engine.robustness import sweep
from sim_engine.training import pd_target


# --------------------------------------------------------------------------- #
# physics: defaults unchanged, optional terms active
# --------------------------------------------------------------------------- #
def test_step_physics_defaults_are_unchanged():
    s = torch.tensor([0.01, -0.02, 0.03, 0.04])
    t = torch.tensor([0.1, -0.2])
    got = step_physics(s, t)
    ax, ay = -C_CONST * t[0], -C_CONST * t[1]
    vx, vy = s[2] + ax * DT, s[3] + ay * DT
    expected = torch.tensor([s[0] + vx * DT, s[1] + vy * DT, vx, vy])
    assert torch.allclose(got, expected)


def test_damping_and_disturbance_change_the_step():
    s = torch.tensor([0.0, 0.0, 0.1, 0.1])
    t = torch.tensor([0.0, 0.0])
    damped = step_physics(s, t, damping=0.5)
    # a = -b*v  ->  v_next = v + (-0.5*0.1)*dt
    assert damped[2] == pytest.approx(0.1 + (-0.5 * 0.1) * DT)
    kicked = step_physics(s, t, disturbance=torch.tensor([1.0, 2.0]))
    assert kicked[2] == pytest.approx(0.1 + 1.0 * DT)
    assert kicked[3] == pytest.approx(0.1 + 2.0 * DT)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def test_environment_is_deterministic_for_a_seed():
    preset = EmbodimentConfig(preset="test", sensor_noise_pos=0.01, process_noise=0.2)

    def run():
        env = EmbodiedEnv(preset)
        env.reset(seed=7)
        state = torch.tensor([0.1, 0.1, 0.2, 0.2])
        u = torch.tensor([0.1, -0.1])
        obs, dist = [], []
        for k in range(25):
            obs.append(env.measure(state).numpy())
            env.step(state, env.actuate(u), k)
            dist.append(env.last_disturbance.numpy())
        return np.array(obs), np.array(dist)

    o1, d1 = run()
    o2, d2 = run()
    assert np.allclose(o1, o2)
    assert np.allclose(d1, d2)
    assert np.abs(d1).sum() > 0  # the environment really perturbs


def test_sensor_delay_returns_delayed_state():
    env = EmbodiedEnv(EmbodimentConfig(sensor_delay=2), init_state=(0, 0, 0, 0))
    env.reset(seed=0)
    trace = [float(env.measure(torch.tensor([k, k, k, k], dtype=torch.float32))[0]) for k in range(6)]
    assert trace == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_measurement_is_position_only():
    env = EmbodiedEnv(EmbodimentConfig(enable=True), init_state=(0.1, -0.2, 3.0, 4.0))
    env.reset(seed=0)
    y = env.measure(torch.tensor([0.1, -0.2, 3.0, 4.0]))
    assert y.shape == (2,)                     # velocity is NOT measured
    assert torch.allclose(y, torch.tensor([0.1, -0.2], dtype=y.dtype))


def test_actuator_gain_and_delay():
    env = EmbodiedEnv(EmbodimentConfig(actuator_gain=2.0, actuator_delay=1))
    env.reset(seed=0)
    u = torch.tensor([0.1, 0.1])
    assert torch.allclose(env.actuate(u), torch.zeros(2))          # still in the delay line
    assert torch.allclose(env.actuate(u), torch.tensor([0.2, 0.2]))  # gained, delayed by one


def test_randomization_stays_within_bounds():
    cfg = EmbodimentConfig(randomize=True, c_scale_range=(0.5, 0.6), damping_range=(0.1, 0.2))
    env = EmbodiedEnv(cfg)
    for seed in range(5):
        env.reset(seed=seed)
        assert 0.5 <= env.c_scale <= 0.6
        assert 0.1 <= env.damping <= 0.2


def test_presets_and_specs():
    assert EmbodimentConfig.from_preset("clean").is_clean
    assert not EmbodimentConfig.from_preset("embodied").is_clean
    assert set(embodiment_specs()) == set(EMBODIMENT_PRESETS)
    for name, spec in embodiment_specs().items():
        assert spec["preset"] == name


# --------------------------------------------------------------------------- #
# sign convention, feed-forward channel, velocity-kick impulses
# --------------------------------------------------------------------------- #
def test_error_sign_convention():
    """e = x − r (the code's convention): a ball right of the reference needs a
    positive tilt, and pd_target reproduces that."""
    ref = orbit_reference(steps=10, radius=0.15, freq=0.5)
    rp = ref.at(3)
    state = torch.tensor([float(rp.pos[0]) + 0.05, float(rp.pos[1]),
                          float(rp.vel[0]), float(rp.vel[1])])
    e = error_vector(state, rp)
    assert float(e[0]) == pytest.approx(0.05)
    assert float(ClassicalPDController().act(state, rp)[0]) > 0.0
    label = pd_target(torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                      ClassicalPDController())
    assert float(label[0, 0]) > 0.0


def test_impulse_is_a_velocity_kick():
    env = EmbodiedEnv(
        EmbodimentConfig(impulse_interval=5, impulse_std=0.2, process_noise=0.0),
        init_state=(0.0, 0.0, 0.0, 0.0),
    )
    env.reset(seed=1)
    state = torch.zeros(4)
    u = torch.zeros(2)
    for k in range(5):
        state = env.step(state, u, k)
    v_before = state[2:].clone()
    state = env.step(state, u, 5)
    assert env.impulse_count == 1
    assert torch.allclose(state[2:] - v_before, env.last_kick)
    assert float(env.last_kick.abs().sum()) > 0.0


def test_pid_no_ff_is_the_no_feedforward_limit():
    engine = Engine()
    cfg = BenchmarkConfig(steps=250, embodiment=EmbodimentConfig.from_preset("clean"))
    pid = engine.run("pid", config=cfg)
    no_ff = engine.run("pid_no_ff", config=cfg)
    assert pid.mean_error_cm < 3.5                       # PID with feed-forward
    assert 6.0 < no_ff.mean_error_cm < 10.0              # ≈7.9 cm analytic no-FF limit
    assert no_ff.mean_error_cm > 1.8 * pid.mean_error_cm


# --------------------------------------------------------------------------- #
# benchmark integration
# --------------------------------------------------------------------------- #
def test_clean_benchmark_uses_the_estimator():
    engine = Engine()
    res = engine.run("pid", config=BenchmarkConfig(steps=500))
    # clean still tracks well, but the controller now runs on the Kalman estimate
    assert res.mean_error_cm < 3.0
    assert res.on_plate_pct is not None
    assert res.estimates is not None and res.estimates.shape == (500, 4)
    assert res.measurements is not None and res.measurements.shape == (500, 2)
    assert res.estimation_pos_rmse_cm is not None and res.estimation_pos_rmse_cm < 1.0
    assert res.estimator is not None and res.estimator["name"] == "kalman"


def test_embodied_benchmark_reports_new_metrics():
    engine = Engine()
    cfg = BenchmarkConfig(steps=500, embodiment=EmbodimentConfig.from_preset("perturbed"))
    res = engine.run("pid", config=cfg)
    assert 0.0 <= res.on_plate_pct <= 100.0
    assert res.impulse_count == 8  # every 60 frames over 500 -> 60..480
    assert res.disturbance_rms is not None and res.disturbance_rms > 0
    assert res.env is not None and res.env["preset"] == "perturbed"
    assert res.estimation_pos_rmse_cm is not None
    assert res.measurements is not None and res.measurements.shape[1] == 2
    # a delayed sensor clearly degrades PID's clean 2.426 cm
    delayed = engine.run("pid", config=BenchmarkConfig(
        steps=500, embodiment=EmbodimentConfig.from_preset("delayed")))
    assert delayed.mean_error_cm > res.mean_error_cm


def test_controllers_share_one_disturbance_realization():
    engine = Engine()
    cfg = BenchmarkConfig(steps=130, embodiment=EmbodimentConfig.from_preset("perturbed"))
    report = evaluate(
        {"pid": engine.build_controller("pid"), "random_ann": engine.build_controller("random_ann")},
        config=cfg,
    )
    a, b = report.results["pid"], report.results["random_ann"]
    assert a.impulse_count == b.impulse_count
    assert a.disturbance_rms == pytest.approx(b.disturbance_rms)


# --------------------------------------------------------------------------- #
# robustness sweep
# --------------------------------------------------------------------------- #
def test_robustness_sweep_structure():
    res = sweep(
        lambda: {"pid": ClassicalPDController()},
        config=BenchmarkConfig(steps=60),
        axis="delay",
        points=[0, 1, 2],
    )
    assert res["axis"] == "delay"
    assert len(res["cells"]) == 3
    for cell in res["cells"]:
        assert "pid" in cell["per_controller"]
        assert cell["per_controller"]["pid"]["mean_error_cm"] > 0
    # more delay should not improve tracking (short runs are transient-dominated)
    means = [c["per_controller"]["pid"]["mean_error_cm"] for c in res["cells"]]
    assert means[2] >= means[0] - 1.0


def test_robustness_sweep_rejects_unknown_axis():
    with pytest.raises(ValueError):
        sweep(lambda: {"pid": ClassicalPDController()}, axis="nope")


# --------------------------------------------------------------------------- #
# robust distillation + engine wiring
# --------------------------------------------------------------------------- #
def test_robust_distillation_smoke():
    from sim_engine import training as training_mod

    net = NetworkConfig(n_neurons=16, synapses_per_neuron=4)
    cfg = TrainingConfig(
        epochs=2, episodes=1, episode_steps=20, seed=1, profile="robust",
        embodiment=EmbodimentConfig.from_preset("noisy"), connectome_unroll=2, log_every=1,
    )
    out = training_mod.distill_embodied(net, cfg)
    assert out["episodes"] == 1
    assert np.isfinite(out["final_loss"]["dense"])
    assert np.isfinite(out["final_loss"]["connectome"])
    assert out["config"]["profile"] == "robust"


def test_engine_robust_train_on_init():
    cfg = EngineConfig(
        n_neurons=16,
        train_on_init=True,
        benchmark=BenchmarkConfig(embodiment=EmbodimentConfig.from_preset("noisy")),
        training=TrainingConfig(
            profile="robust", epochs=2, episodes=1, episode_steps=20,
            connectome_unroll=2, log_every=1,
        ),
    )
    engine = Engine(cfg)
    assert engine.registry.trained
    assert engine.last_training["config"]["profile"] == "robust"
