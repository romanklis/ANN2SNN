"""Controller package — every ball-and-plate brain in one importable place.

Importing this package gives you the four benchmark controllers named in the
project objective::

    from sim_engine.controllers import (
        ClassicalPDController,      # PID controlled
        DenseNNController,          # random ANN / distilled dense ANN
        ConnectomeANNController,    # fly-like sparse recurrent ANN
        LosslessConnectomeSNN,      # SNN transferred from the connectome ANN
    )
"""

from __future__ import annotations

from .base import BaseController, ControllerError, error_vector
from .classical import ClassicalPDController
from .connectome import ConnectomeANNController, ConnectomeTopology
from .dense import DenseNNController
from .snn import LosslessConnectomeSNN

__all__ = [
    "BaseController",
    "ControllerError",
    "error_vector",
    "ClassicalPDController",
    "DenseNNController",
    "ConnectomeANNController",
    "ConnectomeTopology",
    "LosslessConnectomeSNN",
]
