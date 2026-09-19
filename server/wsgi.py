"""WSGI entry point.

Works both as ``wsgi:application`` (gunicorn launched from ``server/``) and as
``server.wsgi:application`` (launched from the repo root)::

    gunicorn -b 0.0.0.0:8080 --workers 2 --timeout 300 server.wsgi:application
"""

from __future__ import annotations

try:  # launched from server/
    from app import app as application
except ImportError:  # launched from the repo root
    from server.app import app as application

__all__ = ["application"]
