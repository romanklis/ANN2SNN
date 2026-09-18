"""Allow ``python -m sim_engine ...`` without referencing :mod:`cli` explicitly."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
