"""Smoke test: the rollout renderer produces a valid, probe-able MP4.

Skipped when ffmpeg/ffprobe are unavailable. Kept short (6 frames) so it can run
with the normal suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

from tools.render import Rollout, ffmpeg_available, probe_video, render_video  # noqa: E402


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not available")
def test_render_smoke(tmp_path: Path):
    steps = 6
    t = np.arange(steps) * 0.02
    radius = 0.15
    reference = np.stack([radius * np.cos(2 * np.pi * 0.5 * t),
                          radius * np.sin(2 * np.pi * 0.5 * t)], axis=1)
    trajectory = np.zeros((steps, 4))
    trajectory[:, :2] = reference * 0.9
    rollout = Rollout(
        name="pid",
        trajectory=trajectory,
        tilts=np.full((steps, 2), 0.05),
        tracking_error=np.full(steps, 1.5),
        mean_error_cm=1.5,
    )
    out = tmp_path / "smoke.mp4"
    info = render_video(
        rollout,
        ref_pos=reference,
        dt=0.02,
        radius=radius,
        out_path=str(out),
        label="Smoke",
        color="#1f77b4",
        fps=5,
        dpi=70,
    )
    assert out.is_file() and out.stat().st_size > 1000
    meta = probe_video(str(out))
    assert meta["frames"] >= 1 and meta["duration"] > 0
    assert info.width > 0 and info.height > 0
