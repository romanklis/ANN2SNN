"""End-to-end checks for the built dashboard bundle + its Flask wiring.

Exercises the *real* app via ``app.test_client()`` (no server) and asserts the
built ``index.html`` exists, every referenced asset is served, the JSON API the
dashboard depends on is intact, ``/api/*`` is never shadowed by the static
catch-all and the bundle carries no obvious secrets.

Skips gracefully until ``cd web && npm ci && npm run build`` has produced
``server/static/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from server import app as app_module

STATIC_DIR = Path(app_module.WEB_ROOT)
INDEX = STATIC_DIR / "index.html"


def _require_bundle() -> Path:
    if not INDEX.is_file():
        pytest.skip(f"dashboard bundle not built at {INDEX}")
    return INDEX


@pytest.fixture(scope="module")
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


@pytest.fixture(scope="module")
def built_index() -> Path:
    return _require_bundle()


# --------------------------------------------------------------------------- #
# build integrity
# --------------------------------------------------------------------------- #
def test_static_assets_present(built_index: Path):
    html = built_index.read_text()
    refs = re.findall(r'(?:src|href)="\.?/?assets/([^"]+)"', html)
    assert refs, "index.html should reference at least one bundled asset"
    for ref in refs:
        assert (STATIC_DIR / "assets" / ref).is_file(), f"missing asset {ref}"


def test_bundle_has_no_obvious_secrets(built_index: Path):
    assets = STATIC_DIR / "assets"
    pattern = re.compile(r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|gh[pous]_[A-Za-z0-9]{20,})")
    for asset in assets.iterdir():
        if asset.suffix in {".js", ".css"}:
            assert not pattern.search(asset.read_text(errors="ignore")), asset.name


def test_single_screen_layout(built_index: Path):
    """The experiment dashboard must fit one screen: hero stage, controller
    pipeline, spike activity, control output, tracking, result bar, one Play."""
    html = built_index.read_text()
    for required in ("stage", "spikes", "control-out", "tracking",
                     "controller-pipeline", "result-ann", "result-snn",
                     "result-delta", "play-btn", "api-pill", "trained-pill",
                     "record-btn", "export-btn"):
        assert f'id="{required}"' in html, f"missing #{required}"
    for removed in ("controller", "seed", "compare-all-btn", "ranking-table",
                    "err-chart", "tilt-chart", "meta-body", "run-form",
                    "model-guide", "models-sub", "panel-models", "baselines"):
        assert f'id="{removed}"' not in html, f"#{removed} should be gone"
    assert "compare-btn" not in html
    assert "BASELINES" not in html
    assert "FLY-LIKE ANN" in html and "SNN TRANSFERRED" in html

    # the no-scroll rule lives in the extracted stylesheet
    css_refs = re.findall(r'href="\.?/?assets/([^"]+\.css)"', html)
    assert css_refs, "expected a bundled stylesheet"
    css = (STATIC_DIR / "assets" / css_refs[0]).read_text(errors="ignore")
    assert "overflow:hidden" in css.replace(" ", "")


# --------------------------------------------------------------------------- #
# Flask wiring
# --------------------------------------------------------------------------- #
def test_root_serves_dashboard(client, built_index: Path):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("text/html")
    assert b"<title>" in r.data


def test_assets_are_served(client, built_index: Path):
    html = built_index.read_text()
    refs = re.findall(r'(?:src|href)="\.?/?assets/([^"]+)"', html)
    assert refs
    for ref in refs:
        r = client.get(f"/assets/{ref}")
        assert r.status_code == 200, ref


def test_api_is_not_shadowed(client):
    assert client.get("/api/controllers").status_code == 200
    assert client.get("/api/definitely-not-real").status_code == 404


def test_dashboard_contract(client):
    cat = client.get("/api/controllers").get_json()
    assert cat["controllers"]
    assert cat["catalog"]
    assert isinstance(cat["micro_steps"], int) and cat["micro_steps"] > 0
    assert cat["torch_version"] and cat["engine_version"]

    pid = client.post("/api/simulate", json={"controller": "pid", "steps": 20, "seed": 42}).get_json()
    assert len(pid["trajectory"]) == len(pid["error_cm"]) == 20
    assert pid["metrics"]["mean_error_cm"] > 0

    snn = client.post(
        "/api/simulate",
        json={"controller": "snn_transferred", "steps": 20, "seed": 42,
              "spike_format": "events"},
    ).get_json()
    assert snn["spikes"]["format"] == "events" and snn["spikes"]["shape"][0] == 20
