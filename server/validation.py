"""Request validation and embodiment/profile parsing for the dashboard API."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

from flask import request

from sim_engine.config import EmbodimentConfig


class BadRequest(ValueError):
    """Raised for a malformed body/query; turned into an HTTP 400."""


def _as_int(value: Any, field: str, default: int, lo: int, hi: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise BadRequest(f"'{field}' must be an integer")
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise BadRequest(f"'{field}' must be an integer, got {value!r}")
    if ivalue < lo or ivalue > hi:
        raise BadRequest(f"'{field}' must be within [{lo}, {hi}], got {ivalue}")
    return ivalue


def _as_float(value: Any, field: str, default: float, lo: float, hi: float) -> float:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise BadRequest(f"'{field}' must be a number")
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise BadRequest(f"'{field}' must be a number, got {value!r}")
    if not (lo <= fvalue <= hi):
        raise BadRequest(f"'{field}' must be within [{lo}, {hi}], got {fvalue}")
    return fvalue


def _as_bool(value: Any, field: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    raise BadRequest(f"'{field}' must be a boolean, got {value!r}")


def _body() -> Dict[str, Any]:
    """Parse the JSON body, accepting an empty body as ``{}``."""
    data = request.get_json(force=True, silent=True)
    if data is None:
        if not request.get_data(cache=True):
            return {}
        raise BadRequest("request body must be valid JSON")
    if not isinstance(data, dict):
        raise BadRequest("request body must be a JSON object")
    return data


def _as_trace_level(body: Dict[str, Any]) -> str:
    """Trace verbosity for a benchmark/simulate request (``short`` | ``full``)."""
    from sim_engine.config import TRACE_LEVELS

    raw = body.get("trace_level")
    if raw is None:
        return "short"
    level = str(raw).strip().lower()
    if level not in TRACE_LEVELS:
        raise BadRequest(f"'trace_level' must be one of: {list(TRACE_LEVELS)}")
    return level


def _spike_payload(spikes: Any, fmt: str) -> Optional[Dict[str, Any]]:
    """Convert a raw ``(T, N)`` spike matrix into the requested wire format."""
    if spikes is None or fmt == "none":
        return None
    T = len(spikes)
    N = len(spikes[0]) if T else 0
    if fmt == "full":
        return {"format": "full", "shape": [T, N], "data": spikes}
    if fmt == "events":
        events = [
            [t, n]
            for t, row in enumerate(spikes)
            for n, v in enumerate(row)
            if v
        ]
        return {"format": "events", "shape": [T, N], "data": events}
    if fmt == "counts":
        counts = [0] * N
        for row in spikes:
            for n, v in enumerate(row):
                if v:
                    counts[n] += 1
        return {"format": "counts", "shape": [T, N], "data": counts}
    raise BadRequest("'spike_format' must be one of: full, events, counts, none")


def _as_spike_format(body: Dict[str, Any]) -> str:
    fmt = str(body.get("spike_format", "events")).strip().lower()
    if fmt not in {"full", "events", "counts", "none"}:
        raise BadRequest("'spike_format' must be one of: full, events, counts, none")
    return fmt


def _validate_embodiment(cfg: EmbodimentConfig) -> None:
    """Bound environmental difficulty so a request cannot be pathological."""
    checks = [
        ("sensor_noise_pos", cfg.sensor_noise_pos, 0.0, 0.05),
        ("sensor_noise_scale", cfg.sensor_noise_scale, 0.0, 20.0),
        ("sensor_delay", cfg.sensor_delay, 0, 20),
        ("actuator_delay", cfg.actuator_delay, 0, 20),
        ("actuator_gain", cfg.actuator_gain, 0.1, 3.0),
        ("damping", cfg.damping, 0.0, 5.0),
        ("c_scale", cfg.c_scale, 0.3, 2.0),
        ("process_noise", cfg.process_noise, 0.0, 5.0),
        ("impulse_std", cfg.impulse_std, 0.0, 1.0),
        ("impulse_prob", cfg.impulse_prob, 0.0, 1.0),
        ("impulse_interval", cfg.impulse_interval, 0, 1000),
    ]
    for name, value, lo, hi in checks:
        if value < lo or value > hi:
            raise BadRequest(f"'embodiment.{name}' must be within [{lo}, {hi}], got {value}")


def _env_from_body(body: Dict[str, Any]) -> Optional[EmbodimentConfig]:
    """Build an :class:`EmbodimentConfig` from the request (or ``None``)."""
    raw = body.get("embodiment")
    if raw is None or raw is False:
        return None
    if raw is True:
        return EmbodimentConfig.from_preset("embodied")
    if isinstance(raw, str):
        try:
            return EmbodimentConfig.from_preset(raw)
        except ValueError as exc:
            raise BadRequest(str(exc))
    if isinstance(raw, dict):
        preset = raw.get("preset")
        base = EmbodimentConfig.from_preset(preset) if preset else EmbodimentConfig()
        data = dataclasses.asdict(base)
        for key, value in raw.items():
            if key in data and value is not None:
                data[key] = value
        try:
            cfg = EmbodimentConfig(**data)
        except TypeError as exc:
            raise BadRequest(f"invalid embodiment object: {exc}")
        _validate_embodiment(cfg)
        return cfg
    raise BadRequest("'embodiment' must be a preset name, a config object, or true")


def _profile_from_body(body: Dict[str, Any]) -> str:
    profile = str(body.get("profile", "clean")).strip().lower()
    if profile not in {"clean", "robust"}:
        raise BadRequest("'profile' must be 'clean' or 'robust'")
    return profile


def _example_from_body(body: Dict[str, Any]) -> Optional[str]:
    """Parse/validate the requested example, or ``None`` when not specified."""
    raw = body.get("example")
    if raw is None:
        return None
    from sim_engine.examples import example_names

    name = str(raw).strip().lower()
    if name not in example_names():
        raise BadRequest(f"'example' must be one of: {', '.join(example_names())}")
    return name
