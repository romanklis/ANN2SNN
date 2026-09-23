"""Weight-bundle guards.

A weights bundle is dimensioned by its example (6→2 for the ball, 9→3 for the
drones) and by the policy profile.  Two failure modes have bitten:

* an override registered for one example was reused by another (the runtime keyed
  overrides by profile alone), and
* a stale/mismatched ``weights_path`` was loaded unconditionally, so a bad cache
  file raised ``RuntimeError`` out of engine construction and surfaced as a 500.

The engine must therefore treat an unusable bundle as "no bundle": warn and fall
back to an untrained engine.
"""

from __future__ import annotations

import logging

import pytest

from sim_engine.config import BenchmarkConfig, EngineConfig
from sim_engine.engine import Engine
from sim_engine.training import load_weights


def _boom(*_args, **_kwargs):
    raise RuntimeError(
        "Error(s) in loading state_dict for DenseNNController: "
        "size mismatch for net.2.weight: copying a param with shape "
        "torch.Size([3, 1000]) from checkpoint, the shape in current model is "
        "torch.Size([2, 1000])."
    )


def test_engine_ignores_an_unusable_weights_bundle(tmp_path, monkeypatch, caplog):
    bundle = tmp_path / "stale.pt"
    bundle.write_bytes(b"not a real bundle")
    monkeypatch.setattr("sim_engine.training.load_weights", _boom)

    cfg = EngineConfig(benchmark=BenchmarkConfig(example="ball", steps=30))
    cfg.weights_path = str(bundle)

    with caplog.at_level(logging.WARNING, logger="sim_engine.engine"):
        engine = Engine(cfg)

    assert engine.registry.trained is False
    assert "ignoring unusable weights bundle" in caplog.text
    # and the engine is still usable
    res = engine.run("pid", config=BenchmarkConfig(steps=30, example="ball"))
    assert res.mean_error_cm > 0.0


def test_engine_ignores_a_bundle_with_the_wrong_format(tmp_path, monkeypatch):
    """Anything ``load_weights`` rejects is treated the same way."""
    bundle = tmp_path / "old-format.pt"
    bundle.write_bytes(b"x")
    monkeypatch.setattr(
        "sim_engine.training.load_weights",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("unsupported weights bundle format")),
    )
    cfg = EngineConfig(benchmark=BenchmarkConfig(example="drone_gps_denied", steps=10))
    cfg.weights_path = str(bundle)
    engine = Engine(cfg)
    assert engine.registry.trained is False


def test_engine_still_loads_a_good_bundle(tmp_path):
    """The guard must not swallow working bundles."""
    path = tmp_path / "good.pt"
    ref = Engine(EngineConfig(benchmark=BenchmarkConfig(example="ball", steps=10)))
    ref.save_weights(str(path))
    cfg = EngineConfig(benchmark=BenchmarkConfig(example="ball", steps=10))
    cfg.weights_path = str(path)
    engine = Engine(cfg)
    assert engine.registry.trained is True
    assert path.is_file()


def test_a_bundle_for_another_example_round_trips(tmp_path):
    """A 9→3 bundle must load as 9→3, not be forced into the 6→2 policy shape.

    This is the regression behind the dashboard 500: once the drone's bundle was
    cached, every later request rebuilt a 2-output model and failed to load it.
    """
    drone = Engine(EngineConfig(benchmark=BenchmarkConfig(example="drone_gps_denied",
                                                         steps=10)))
    path = tmp_path / "drone.pt"
    drone.save_weights(str(path))

    loaded = load_weights(str(path))
    assert (loaded["dense"].n_in, loaded["dense"].n_out) == (9, 3)
    assert (loaded["connectome"].n_in, loaded["connectome"].n_out) == (9, 3)

    cfg = EngineConfig(benchmark=BenchmarkConfig(example="drone_gps_denied", steps=10))
    cfg.weights_path = str(path)
    engine = Engine(cfg)
    assert engine.registry.trained is True


def test_ball_bundle_still_loads_as_6_to_2(tmp_path):
    ball = Engine(EngineConfig(benchmark=BenchmarkConfig(example="ball", steps=10)))
    path = tmp_path / "ball.pt"
    ball.save_weights(str(path))
    loaded = load_weights(str(path))
    assert (loaded["dense"].n_in, loaded["dense"].n_out) == (6, 2)
    assert (loaded["connectome"].n_in, loaded["connectome"].n_out) == (6, 2)
