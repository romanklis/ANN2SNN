"""Helpers that turn torch/numpy results into JSON-serializable primitives.

The web backend consumes the simulation engine through these helpers, so the
engine never leaks ``torch.Tensor`` objects across the API boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def to_numpy(value: Any) -> Any:
    """Recursively convert tensors/arrays inside *value* to numpy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: to_numpy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_numpy(v) for v in value]
    return value


def to_jsonable(value: Any, *, precision: int = 6) -> Any:
    """Recursively convert *value* into JSON-native types.

    Tensors/arrays become nested lists with floats rounded to *precision*
    decimals; numpy scalars become Python scalars; dataclasses become dicts.
    """
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy(), precision=precision)
    if isinstance(value, np.ndarray):
        return [to_jsonable(v, precision=precision) for v in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), precision=precision)
    if isinstance(value, float):
        return round(float(value), precision)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, precision=precision) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, precision=precision) for v in value]
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict(), precision=precision)
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return to_jsonable(dataclasses.asdict(value), precision=precision)
    return str(value)
