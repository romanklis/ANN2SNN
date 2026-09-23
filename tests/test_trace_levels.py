"""The extended (``trace_level="full"``) telemetry contract.

The hero dashboard sends ``short`` and must keep its exact payload; the extended
view opts into the estimator internals and the environment side-channels.  These
tests pin the wire contract, the index alignment of the per-channel series, and
the physical sanity of the numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_engine.benchmark import evaluate
from sim_engine.config import BenchmarkConfig, EmbodimentConfig, EngineConfig
from sim_engine.engine import Engine

FULL_KEYS = (
    "applied",
    "disturbances",
    "impulse_frames",
    "measurements_by_channel",
    "innovations",
    "covariance_diag",
    "success",
)


def _run(example="ball", *, level="full", steps=60, profile="clean",
         controller="pid", **emb_kwargs) -> dict:
    emb = EmbodimentConfig.from_preset(profile)
    for key, value in emb_kwargs.items():
        setattr(emb, key, value)
    bc = BenchmarkConfig(steps=steps, example=example, embodiment=emb,
                        trace_level=level)
    engine = Engine(EngineConfig(benchmark=bc))
    report = evaluate(
        {controller: engine.build_controller(controller)},
        reference=engine.reference(steps=steps),
        config=bc,
    )
    return report.to_dict(include_trace=True)["results"][controller]


# --------------------------------------------------------------------------- #
# the wire contract
# --------------------------------------------------------------------------- #
def test_short_payload_is_unchanged():
    body = _run("ball", level="short", steps=40)
    for key in FULL_KEYS:
        assert key not in body, f"{key} must not appear under trace_level='short'"
    # the base traces the hero dashboard relies on are still there
    assert len(body["trajectory"]) == 40
    assert body["estimates"] and body["tilts"]


def test_full_payload_adds_every_field():
    body = _run("ball", level="full", steps=40)
    for key in FULL_KEYS:
        assert key in body, f"{key} missing under trace_level='full'"
    assert np.asarray(body["applied"]).shape == (40, 2)
    assert np.asarray(body["disturbances"]).shape == (40, 2)
    assert np.asarray(body["covariance_diag"]).shape == (40, 4)
    assert len(body["success"]) == 40
    assert isinstance(body["impulse_frames"], list)
    assert set(body["measurements_by_channel"]) == {"position_fix"}
    assert set(body["innovations"]) == {"position_fix"}


def test_full_precision_is_finite_and_json_safe():
    import json

    body = _run("drone_gps_denied", level="full", steps=80)
    text = json.dumps(body)
    assert "NaN" not in text and "Infinity" not in text
    # None gaps survive the round trip as JSON null
    assert json.loads(text)["innovations"]["position_fix"][0] is None


def test_trace_level_is_validated():
    with pytest.raises(ValueError):
        BenchmarkConfig(trace_level="deep")


def test_full_traces_are_deterministic():
    a = _run("drone_gps_denied", level="full", steps=60)
    b = _run("drone_gps_denied", level="full", steps=60)
    for key in ("applied", "disturbances", "covariance_diag"):
        assert np.allclose(np.asarray(a[key]), np.asarray(b[key]))
    for kind in ("altitude", "flow_velocity"):
        va, vb = a["innovations"][kind], b["innovations"][kind]
        assert [None if x is None else list(x) for x in va] == [
            None if x is None else list(x) for x in vb
        ]


# --------------------------------------------------------------------------- #
# per-channel alignment and innovations
# --------------------------------------------------------------------------- #
def test_meas_camera_innovations_are_zero_when_noiseless():
    """A noiseless, exactly-modelled camera leaves nothing to correct."""
    body = _run("ball", level="full", steps=60)
    series = body["measurements_by_channel"]["position_fix"]
    assert all(v is not None for v in series)      # a sample every frame
    innov = [v for v in body["innovations"]["position_fix"] if v is not None]
    assert len(innov) == 59                        # index 0 seeds from the first fix
    assert max(abs(x) for v in innov for x in v) == pytest.approx(0.0, abs=1e-9)


def test_delayed_channel_innovations_are_index_aligned():
    """A delayed sample is attributed to the state it measured, not the frame."""
    body = _run("drone_gps_denied", level="full", steps=80)
    channels = {c["kind"]: c for c in body["estimator"]["sensor"]["channels"]}
    lat = channels["altitude"]["latency"]
    assert lat > 0
    alt = body["innovations"]["altitude"]
    # the sample for index 0 arrives at frame `lat`, so it is present at index 0...
    assert alt[0] is not None
    # ... and the last `lat` indices can never be filled within the horizon
    assert all(v is None for v in alt[-lat:])
    assert sum(1 for v in alt if v is not None) == len(alt) - lat


def test_measurements_by_channel_marks_the_gaps():
    body = _run("drone_gps_denied", level="full", steps=120)
    series = body["measurements_by_channel"]
    assert set(series) == {"altitude", "flow_velocity", "position_fix"}
    # the gated checkpoint is silent until the drone first comes in range
    fixes = series["position_fix"]
    assert any(v is None for v in fixes) and any(v is not None for v in fixes)
    # the continuous channels sample every frame (behind their delay lines)
    assert all(v is not None for v in series["altitude"])
    assert all(v is not None for v in series["flow_velocity"])


def test_innovation_magnitude_matches_sensor_noise():
    body = _run("drone_gps_denied", level="full", steps=200)
    channels = {c["kind"]: c for c in body["estimator"]["sensor"]["channels"]}
    for kind in ("altitude", "flow_velocity"):
        sigma = channels[kind]["sigma"]
        vals = [abs(x) for v in body["innovations"][kind] if v is not None for x in v]
        mean = float(np.mean(vals))
        assert 0.3 * sigma < mean < 3.0 * sigma, (kind, sigma, mean)


# --------------------------------------------------------------------------- #
# covariance, actuators and impulses
# --------------------------------------------------------------------------- #
def test_covariance_is_positive_and_drops_at_a_fix():
    steps = 200
    body = _run("drone_gps_denied", level="full", steps=steps)
    cov = np.asarray(body["covariance_diag"])
    assert cov.shape == (steps, 6)
    assert np.all(cov >= 0.0) and np.all(np.isfinite(cov))
    assert body["fix_events"], "expected at least one checkpoint fix"
    # A fix sensed at frame f measures the state at frame f; its latency only
    # delays *delivery*, so the filter folds it into index f (index == frame).
    idx = int(body["fix_events"][0]["k"])
    assert idx > 0
    assert cov[idx][:3].sum() < cov[idx - 1][:3].sum()   # uncertainty collapses


def test_applied_matches_commanded_with_an_identity_actuator():
    clean = _run("ball", level="full", steps=40)
    assert np.allclose(np.asarray(clean["applied"]), np.asarray(clean["tilts"]))

    biased = _run("ball", level="full", steps=40, actuator_gain=2.0)
    assert not np.allclose(np.asarray(biased["applied"]), np.asarray(biased["tilts"]))
    # the applied command is the gain-scaled (and clamped) commanded one
    assert np.abs(np.asarray(biased["applied"])).max() >= np.abs(
        np.asarray(biased["tilts"])).max()


def test_impulse_frames_match_the_impulse_count():
    body = _run("ball", level="full", steps=150, profile="perturbed")
    assert body["impulse_frames"]
    assert len(body["impulse_frames"]) == body["metrics"]["impulse_count"]
    assert all(
        body["impulse_frames"][i] < body["impulse_frames"][i + 1]
        for i in range(len(body["impulse_frames"]) - 1)
    )


def test_success_tracks_the_example_bounds():
    body = _run("drone_gps_denied", level="full", steps=120)
    traj = np.asarray(body["trajectory"])
    ok = (np.abs(traj[:, :3]) <= 1.0).all(axis=1)
    assert body["success"] == [bool(v) for v in ok]
