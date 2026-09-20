"""Flask test-client tests for the ANN2SNN dashboard API.

Runs entirely in-process (``app.test_client()``) -- no server is started::

    python3 -m pytest server/tests -q

Neural rollouts use a handful of steps so the suite stays fast; the PD
controller also runs the full default-length trace.  The distillation test uses
a tiny network (``n_neurons=8``, one epoch) and restores the runtime afterwards.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from server.app import MAX_STEPS, RUNTIME, WEIGHTS_PATH, app

CANONICAL = ["pid", "random_ann", "flylike_ann", "snn_transferred", "dense_ann"]


@pytest.fixture(scope="module")
def client():
    app.config.update(TESTING=True)
    return app.test_client()


# --------------------------------------------------------------------------- #
# health / catalogue
# --------------------------------------------------------------------------- #
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["service"] == "ann2snn-dashboard"
    assert body["controllers"] == CANONICAL
    assert body["torch_version"]


def test_controllers_payload(client):
    r = client.get("/api/controllers")
    assert r.status_code == 200
    body = r.get_json()
    assert body["controllers"] == CANONICAL
    assert body["count"] == 5
    assert body["dt"] == pytest.approx(0.02)
    assert body["max_steps"] == MAX_STEPS
    assert body["micro_steps"] == 10
    assert body["torch_version"] and body["engine_version"]

    names = {c["name"] for c in body["catalog"]}
    assert names == set(CANONICAL)
    meta = {c["name"]: c for c in body["catalog"]}
    assert meta["snn_transferred"]["spiking"] is True
    assert meta["flylike_ann"]["recurrent"] is True
    assert meta["pid"]["label"]
    assert body["plate_half"] == pytest.approx(0.25)


def test_model_guide_and_training_metadata(client):
    """Every model carries provenance: how obtained, value, how trained."""
    catalog = {c["name"]: c for c in client.get("/api/controllers").get_json()["catalog"]}
    for name in CANONICAL:
        guide = catalog[name]["guide"]
        assert guide["obtained"] and guide["value"]
        assert "method" in guide["training"]
    pid = catalog["pid"]["guide"]["training"]
    assert pid["method"].startswith("none") and pid["teacher"] is None
    fly = catalog["flylike_ann"]["guide"]["training"]
    assert fly["teacher"] == "pid" and fly["method"].startswith("behavioural")
    snn = catalog["snn_transferred"]["guide"]["training"]
    assert snn["transferred_from"] == "flylike_ann"
    # live training facts are merged in once trained
    assert catalog["flylike_ann"]["trained"] is True
    assert fly["epochs"] == 100 and fly["seed"] == 42
    assert fly["final_loss"] is not None


def test_auto_distils_and_caches_when_untrained(client):
    """The learned brains must be distilled on first use, not left diverging."""
    if not RUNTIME.auto_train:
        pytest.skip("auto-distillation disabled (ANN2SNN_AUTO_TRAIN=0)")
    body = client.get("/api/controllers").get_json()
    assert body["trained"] is True
    assert body["weights_path"] and os.path.isfile(body["weights_path"])


# --------------------------------------------------------------------------- #
# single-controller simulate
# --------------------------------------------------------------------------- #
def test_simulate_pid_default(client):
    r = client.post("/api/simulate", json={"controller": "pid"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["controller"] == "pid"
    assert body["steps"] == 500
    assert len(body["trajectory"]) == 500
    assert len(body["tilts"]) == 500
    assert len(body["error_cm"]) == 500
    assert len(body["reference"]["pos"]) == 500
    assert len(body["target"]) == 500
    assert body["metrics"]["mean_error_cm"] > 0
    assert "spikes" not in body


def test_simulate_empty_body_defaults_to_pid(client):
    assert client.post("/api/simulate", json={}).get_json()["controller"] == "pid"
    assert client.post("/api/simulate").get_json()["controller"] == "pid"


@pytest.mark.parametrize("controller", CANONICAL)
def test_simulate_every_controller(client, controller):
    r = client.post("/api/simulate",
                    json={"controller": controller, "steps": 15, "seed": 42})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["controller"] == controller
    assert len(body["trajectory"]) == 15
    assert len(body["reference"]["pos"]) == 15
    assert body["metrics"]["mean_error_cm"] == body["metrics"]["mean_error_cm"]


def test_simulate_aliases(client):
    for alias, expected in [
        ("pd", "pid"),
        ("connectome", "flylike_ann"),
        ("snn", "snn_transferred"),
        ("dense", "dense_ann"),
        ("random", "random_ann"),
    ]:
        r = client.post("/api/simulate", json={"controller": alias, "steps": 6})
        assert r.status_code == 200, alias
        assert r.get_json()["controller"] == expected


def test_simulate_snn_spike_formats(client):
    n_neurons = client.get("/api/controllers").get_json()["n_neurons"]
    full = client.post("/api/simulate",
                       json={"controller": "snn_transferred", "steps": 8, "seed": 1}).get_json()
    assert full["spikes"]["format"] == "events"  # default
    assert full["spikes"]["shape"] == [8, n_neurons]

    events = client.post(
        "/api/simulate",
        json={"controller": "snn_transferred", "steps": 8, "seed": 1,
              "spike_format": "events"},
    ).get_json()["spikes"]
    assert events["format"] == "events"
    for t, n in events["data"]:
        assert 0 <= t < 8 and 0 <= n < n_neurons

    counts = client.post(
        "/api/simulate",
        json={"controller": "snn_transferred", "steps": 8, "spike_format": "counts"},
    ).get_json()["spikes"]
    assert counts["format"] == "counts" and len(counts["data"]) == n_neurons

    none = client.post(
        "/api/simulate",
        json={"controller": "snn_transferred", "steps": 8, "spike_format": "none"},
    ).get_json()
    assert "spikes" not in none


def test_simulate_is_deterministic(client):
    body = {"controller": "snn_transferred", "steps": 10, "seed": 7}
    a = client.post("/api/simulate", json=body).get_json()
    b = client.post("/api/simulate", json=body).get_json()
    assert a["trajectory"] == b["trajectory"]
    assert a["error_cm"] == b["error_cm"]


def test_simulate_custom_geometry(client):
    ref = client.post(
        "/api/simulate",
        json={"controller": "pid", "steps": 20, "radius": 0.2, "freq": 0.25},
    ).get_json()["reference"]
    assert ref["radius"] == pytest.approx(0.2)
    assert ref["freq"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# multi-controller / benchmark
# --------------------------------------------------------------------------- #
def test_simulate_multi_controller(client):
    r = client.post("/api/simulate",
                    json={"controllers": ["pid", "dense_ann"], "steps": 10})
    assert r.status_code == 200
    body = r.get_json()
    assert set(body["results"]) == {"pid", "dense_ann"}
    assert len(body["ranking"]) == 2
    assert body["ok"] is True
    assert len(body["results"]["pid"]["trajectory"]) == 10
    assert body["plate_half_m"] == pytest.approx(0.25)
    # per-controller and aggregate statistics are attached
    assert set(body["stats"]["per_controller"]) == {"pid", "dense_ann"}
    for name, st in body["stats"]["per_controller"].items():
        assert 0.0 <= st["on_plate_pct"] <= 100.0
        assert st["mean_error_cm"] > 0
    summary = body["stats"]["summary"]
    assert summary["n"] == 2
    assert summary["best"] in {"pid", "dense_ann"}
    assert summary["best_mean_error_cm"] <= summary["worst_mean_error_cm"]
    assert summary["spread_cm"] >= 0


def test_benchmark_endpoint(client):
    r = client.post("/api/benchmark",
                    json={"controllers": ["pid", "random_ann"], "steps": 10})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ranking"][0] == "pid"
    assert len(body["reference"]["pos"]) == 10
    assert body["stats"]["summary"]["best"] == "pid"
    assert body["stats"]["plate_half_m"] == pytest.approx(0.25)
    assert body["stats"]["per_controller"]["pid"]["on_plate_pct"] >= 99.0


# --------------------------------------------------------------------------- #
# validation / errors
# --------------------------------------------------------------------------- #
def test_unknown_controller_is_400(client):
    r = client.post("/api/simulate", json={"controller": "nope"})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_bad_types_are_400(client):
    assert client.post("/api/simulate", json={"steps": "many"}).status_code == 400
    assert client.post("/api/simulate", json={"seed": "x"}).status_code == 400
    assert client.post("/api/simulate", json={"steps": 0}).status_code == 400
    assert client.post("/api/simulate", json={"spike_format": "wat"}).status_code == 400
    assert client.post("/api/simulate", json={"controllers": []}).status_code == 400


def test_steps_are_capped(client):
    r = client.post("/api/simulate", json={"controller": "pid", "steps": 10**9})
    assert r.status_code == 400
    assert "steps" in r.get_json()["error"]


def test_invalid_json_is_400(client):
    r = client.post("/api/simulate", data="{not json", content_type="application/json")
    assert r.status_code == 400


def test_404_is_json(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.get_json()["error"] == "not found"


def test_websocket_upgrade_is_a_clean_400(client):
    # A WebSocket upgrade to an HTTP route makes Werkzeug raise WebsocketMismatch
    # (a BadRequest). It must come back as a JSON 400, never a logged 500.
    r = client.get("/", headers={"Upgrade": "websocket", "Connection": "Upgrade"})
    assert r.status_code == 400
    assert r.is_json
    assert r.get_json()["code"] == 400


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
def test_session_lifecycle(client):
    r = client.post("/api/sessions", json={"controller": "pid", "steps": 20})
    assert r.status_code == 201
    sid = r.get_json()["session_id"]

    info = client.get(f"/api/sessions/{sid}").get_json()
    assert info["controller"] == "pid" and info["steps_total"] == 20

    obs = client.post(f"/api/sessions/{sid}/step", json={"n": 3}).get_json()
    assert obs["step"] == 3
    assert "error_cm" in obs and "tilt" in obs

    traj = client.get(f"/api/sessions/{sid}/trajectory").get_json()
    assert traj["steps"] == 3

    switched = client.put(f"/api/sessions/{sid}/controller",
                          json={"controller": "snn"}).get_json()
    assert switched["controller"] == "snn_transferred"

    assert client.post(f"/api/sessions/{sid}/reset").status_code == 200
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_session_manual_action(client):
    sid = client.post("/api/sessions", json={"controller": "pid", "steps": 10}).get_json()["session_id"]
    obs = client.post(f"/api/sessions/{sid}/step",
                      json={"action": [0.1, -0.05]}).get_json()
    assert obs["tilt"][0] == pytest.approx(0.1, abs=1e-3)
    assert obs["tilt"][1] == pytest.approx(-0.05, abs=1e-3)


# --------------------------------------------------------------------------- #
# embodiment / profile / robustness
# --------------------------------------------------------------------------- #
def test_controllers_expose_embodiment(client):
    body = client.get("/api/controllers").get_json()
    assert "clean" in body["profiles"] and "robust" in body["profiles"]
    assert "embodied" in body["embodiment_presets"]
    assert "delay" in body["robustness_axes"]


def test_simulate_with_embodiment_and_profile(client):
    r = client.post("/api/simulate", json={
        "controller": "pid", "steps": 40, "embodiment": "noisy", "profile": "clean",
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["env"]["preset"] == "noisy"
    assert body["profile"] == "clean"
    assert body["metrics"]["on_plate_pct"] >= 0.0


def test_embodiment_validation(client):
    assert client.post("/api/simulate", json={"embodiment": {"sensor_delay": 999}}).status_code == 400
    assert client.post("/api/simulate", json={"embodiment": "not-a-preset"}).status_code == 400
    assert client.post("/api/simulate", json={"profile": "bogus"}).status_code == 400


def test_benchmark_embodied_metrics(client):
    r = client.post("/api/benchmark", json={
        "controllers": ["pid", "random_ann"], "steps": 130, "embodiment": "perturbed",
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["env"]["preset"] == "perturbed"
    assert body["stats"]["per_controller"]["pid"]["impulse_count"] == 2


def test_robustness_endpoint(client):
    r = client.post("/api/robustness", json={
        "controllers": ["pid"], "axis": "delay", "steps": 40, "points": [0, 1],
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["axis"] == "delay"
    assert len(body["cells"]) == 2
    assert body["cells"][0]["per_controller"]["pid"]["mean_error_cm"] > 0


# --------------------------------------------------------------------------- #
# distillation job (tiny network; restores runtime state)
# --------------------------------------------------------------------------- #
def test_train_job_runs_and_registers_weights(client):
    before = RUNTIME.weights_path
    try:
        started = client.post("/api/train",
                              json={"epochs": 1, "n_neurons": 8}).get_json()
        assert started["ok"] is True
        job_id = started["job_id"]

        deadline = time.time() + 120
        state = "queued"
        body = {}
        while time.time() < deadline:
            body = client.get(f"/api/train/{job_id}").get_json()
            state = body["state"]
            if state in {"done", "error"}:
                break
            time.sleep(0.2)
        assert state == "done", body
        assert body["loss"] is not None
        assert os.path.isfile(WEIGHTS_PATH)
    finally:
        # restore the pre-test runtime and clean up the weights file
        RUNTIME.set_weights(before)
        if os.path.exists(WEIGHTS_PATH):
            os.remove(WEIGHTS_PATH)


# --------------------------------------------------------------------------- #
# server-side MP4 export (skipped without ffmpeg)
# --------------------------------------------------------------------------- #
def test_export_mp4(client):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not available")
    r = client.post("/api/export/mp4",
                    json={"controller": "pid", "steps": 6, "seed": 42})
    assert r.status_code == 200, r.get_json()
    assert r.headers["Content-Type"].startswith("video/mp4")
    assert len(r.data) > 1000


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #
def test_cors_header_for_default_origin(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    if app.config.get("CORS_ORIGINS") == "*":
        pytest.skip("CORS is configured as a wildcard in this environment")
    r = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert r.headers.get("Access-Control-Allow-Origin") is None


# --------------------------------------------------------------------------- #
# static bundle (skipped until `npm run build` has produced server/static)
# --------------------------------------------------------------------------- #
def test_root_serves_bundle_or_hint(client):
    r = client.get("/")
    if r.status_code == 404:
        body = r.get_json()
        assert body["error"] == "dashboard assets not found"
        assert "npm" in body["hint"]
    else:
        assert r.status_code == 200
        assert r.headers["Content-Type"].startswith("text/html")
