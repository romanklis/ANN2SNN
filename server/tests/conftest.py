"""Shared pytest fixtures for the dashboard backend tests.

Ensures the repo root and ``server/`` are importable regardless of the CWD, and
pins matplotlib's config dir before anything imports it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
REPO_ROOT = SERVER_DIR.parent

for _p in (str(REPO_ROOT), str(SERVER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("ANN2SNN_THREADS", "2")
# Force a throwaway weights bundle and small brains so tests never pick up a
# baked/production weights file and the auto-distillation path stays cheap.
os.environ["ANN2SNN_WEIGHTS"] = os.path.join(
    "/tmp", f"ann2snn_test_weights_{os.getpid()}.pt"
)
os.environ["ANN2SNN_N_NEURONS"] = "16"
os.environ["ANN2SNN_AUTO_TRAIN"] = "1"
