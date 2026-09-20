"""Smoke tests for the refactored simulation engine.

These are deliberately fast (small neuron counts, short rollouts) so they can run
on every commit. The dedicated ``write-tests`` step is expected to extend this
with full-fidelity numerical tests; this file guards the refactor's contracts.

    pytest -q
"""

from __future__ import annotations

import json
import math

import torch

from sim_engine import (
    C_CONST,
    DT,
    MAX_TILT,
    ClassicalPDController,
    ConnectomeANNController,
    DenseNNController,
    Engine,
    EngineConfig,
    LosslessConnectomeSNN,
    NetworkConfig,
    TrainingConfig,
    orbit_reference,
    step_physics,
)


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def test_physics_matches_analytic_integration():
    state = torch.tensor([0.0, 0.0, 0.0, 0.0])
    tilt = torch.tensor([0.1, -0.2])
    nxt = step_physics(state, tilt)
    # forward Euler with v-update before x-update
    ax, ay = -C_CONST * 0.1, -C_CONST * -0.2
    vx, vy = ax * DT, ay * DT
    assert torch.allclose(nxt, torch.tensor([vx * DT, vy * DT, vx, vy]), atol=1e-6)


def test_physics_saturates_tilt():
    state = torch.zeros(4)
    big = torch.tensor([10.0, -10.0])
    a = step_physics(state, big)
    b = step_physics(state, torch.tensor([MAX_TILT, -MAX_TILT]))
    assert torch.allclose(a, b)


def test_constant_value():
    assert abs(C_CONST - (5.0 / 7.0) * 9.81) < 1e-9


# --------------------------------------------------------------------------- #
# controllers
# --------------------------------------------------------------------------- #
def _ref(k=0):
    return orbit_reference(steps=10).at(k)


def test_pd_controller_saturates_and_is_finite():
    ctrl = ClassicalPDController()
    out = ctrl.act(torch.tensor([1.0, 1.0, 1.0, 1.0]), _ref())
    assert out.shape == (2,)
    assert torch.all(torch.abs(out) <= MAX_TILT + 1e-6)


def test_pd_zero_error_gives_feedforward_only():
    ctrl = ClassicalPDController()
    ref = _ref(0)
    state = torch.tensor([ref.pos[0], ref.pos[1], ref.vel[0], ref.vel[1]])
    out = ctrl.act(state, ref)
    expected = torch.tensor([-ref.acc[0] / C_CONST, -ref.acc[1] / C_CONST])
    assert torch.allclose(out, expected, atol=1e-5)


def test_dense_controller_shapes_and_determinism():
    a = DenseNNController(n_neurons=32, seed=7)
    b = DenseNNController(n_neurons=32, seed=7)
    x = torch.tensor([0.01, -0.02, 0.03, -0.04])
    assert torch.allclose(a.act(x, _ref()), b.act(x, _ref()))
    assert a.act(x, _ref()).shape == (2,)


def test_connectome_topology_is_seeded_and_dales_law():
    c1 = ConnectomeANNController(n_neurons=50, synapses_per_neuron=5, seed=1)
    c2 = ConnectomeANNController(n_neurons=50, synapses_per_neuron=5, seed=1)
    assert torch.equal(c1.topology.indices, c2.topology.indices)
    # sign of every synaptic weight equals the polarity of its presynaptic neuron
    w = c1.topology.signed_weights(c1.raw_weights)
    src = c1.topology.indices[1]
    signs = torch.sign(w)
    expected = torch.sign(c1.topology.polarity[src])
    assert torch.equal(signs, expected)
    assert (~c1.topology.is_inhibitory).sum() > 0
    assert c1.topology.is_inhibitory.sum() > 0


def test_connectome_ann_step_and_reset():
    ctrl = ConnectomeANNController(n_neurons=32, synapses_per_neuron=4, seed=3)
    x = torch.tensor([0.01, 0.02, 0.0, 0.0])
    ctrl.reset()
    out1 = ctrl.act(x, _ref())
    ctrl.reset()
    out2 = ctrl.act(x, _ref())
    assert out1.shape == (2,)
    assert torch.allclose(out1, out2)  # reset makes behaviour reproducible


def test_snn_transfer_is_finite_and_spiking():
    ann = ConnectomeANNController(n_neurons=64, synapses_per_neuron=8, seed=5)
    snn = LosslessConnectomeSNN(ann, micro_steps=5)
    snn.reset()
    tilt, spikes = snn.step(torch.tensor([0.05, -0.05, 0.0, 0.0, 0.0, 0.0]))
    assert tilt.shape == (2,)
    assert spikes.shape == (64,)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}
    assert torch.all(torch.abs(tilt) <= MAX_TILT + 1e-6)
    assert snn.last_spikes() is not None


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_distillation_reduces_loss():
    from sim_engine.training import distill

    net = NetworkConfig(n_neurons=128)
    cfg = TrainingConfig(epochs=40, batch_size=32, device="cpu", seed=0)
    result = distill(net, cfg)
    hist = result["history"]["dense"]
    assert hist[-1] < hist[0]
    assert math.isfinite(result["final_loss"]["connectome"])


# --------------------------------------------------------------------------- #
# benchmark + engine + api
# --------------------------------------------------------------------------- #
def test_closed_loop_runs_and_reports_metrics():
    engine = Engine(EngineConfig(steps=40))
    res = engine.run("pid")
    assert res.trajectory.shape[0] == 40
    assert res.mean_error_cm > 0
    assert res.tilts.shape == (40, 2)


def test_engine_benchmark_json_serializable():
    engine = Engine(EngineConfig(steps=30))
    report = engine.run_benchmark(["pid", "random_ann"], include_trace=False)
    json.dumps(report)  # must not raise
    assert set(report["ranking"]) == {"pid", "random_ann"}


def test_engine_aliases_resolve():
    engine = Engine(EngineConfig(steps=10))
    assert engine.build_controller("SNN").name == "snn_transferred"
    assert engine.build_controller("connectome").name == "flylike_ann"
    assert engine.build_controller("pd").name == "pid"


def test_session_stepping_and_api():
    from sim_engine.api import EngineService

    svc = EngineService({"steps": 25})
    info = svc.new_session("snn_transferred")
    sid = info["session_id"]
    obs = svc.step(sid, n=3)
    assert obs["step"] == 3
    assert len(obs["spikes"]) == 1000
    json.dumps(obs)
    traj = svc.trajectory(sid)
    assert traj["steps"] == 3
    assert svc.reset(sid)["step"] == 0
    assert svc.set_controller(sid, "random")["controller"] == "random_ann"
    svc.close_session(sid)


def test_session_runs_to_completion_and_autoresets():
    engine = Engine(EngineConfig(steps=5))
    sess = engine.session("pid")
    for _ in range(5):
        obs = sess.step()
    assert obs["done"] is True
    obs = sess.step()  # auto-reset kicks in
    assert obs["step"] == 1


def test_weights_roundtrip(tmp_path):
    from sim_engine.training import load_weights, save_weights

    d = DenseNNController(n_neurons=32, seed=1)
    c = ConnectomeANNController(n_neurons=32, synapses_per_neuron=4, seed=1)
    path = str(tmp_path / "w.pt")
    save_weights(path, dense=d, connectome=c)
    loaded = load_weights(path)
    x = torch.tensor([0.01, 0.02, 0.03, 0.04])
    assert torch.allclose(d.act(x, _ref()), loaded["dense"].act(x, _ref()), atol=1e-6)


def test_loaded_bundle_exposes_training_metadata(tmp_path):
    from sim_engine.training import load_weights, save_weights

    d = DenseNNController(n_neurons=32, seed=1)
    c = ConnectomeANNController(n_neurons=32, synapses_per_neuron=4, seed=1)
    path = str(tmp_path / "w.pt")
    training = {
        "history": {"dense": [1.0, 0.5], "connectome": [2.0, 1.0]},
        "final_loss": {"dense": 0.5, "connectome": 1.0},
        "config": TrainingConfig(epochs=2, seed=7).to_dict(),
    }
    save_weights(path, dense=d, connectome=c, training=training)

    loaded = load_weights(path)
    assert loaded["training"]["final_loss"]["dense"] == 0.5

    engine = Engine(EngineConfig(weights_path=path, n_neurons=32, device="cpu"))
    desc = engine.describe()
    assert desc["trained"] is True
    # provenance survives the load, so the dashboard can show "how it was trained"
    assert desc["training"] is not None
    assert desc["training"]["final_loss"]["dense"] == 0.5
    assert desc["training"]["config"]["seed"] == 7
    assert desc["training"]["epochs_logged"] == 2
