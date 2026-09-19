"""Behavioral distillation: teach a neural controller to imitate the PD baseline.

Objective
---------
The classical PD controller with acceleration feed-forward is a strong analytic
teacher.  We sample a batch of bounded tracking errors, label them with the PD
feedback law (``kp*e + kd*ev``), and minimise MSE between the student's
saturated plate command and that label::

    L = MSE( clamp(student(err)), clamp(kp*ex + kd*evx, kp*ey + kd*evy) )

The identical schedule is applied to the dense ANN and to the sparse recurrent
connectome ANN (unrolled ``connectome_unroll`` recurrent steps), which is what
lets us later transfer the connectome ANN into the SNN.

The teacher's feed-forward acceleration term is deliberately omitted from the
labels — it depends on the reference trajectory, not on the error, and the
prototype trained the students on error-only inputs.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from .config import NetworkConfig, TrainingConfig
from .controllers import ClassicalPDController, ConnectomeANNController, DenseNNController
from .physics import SYNAPSES_PER_NEURON

__all__ = [
    "sample_errors",
    "pd_target",
    "train_dense_controller",
    "train_connectome_controller",
    "distill",
    "save_weights",
    "load_weights",
]


def _device_of(module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:  # pragma: no cover - controllers always have params
        return torch.device("cpu")


def sample_errors(
    batch_size: int = 64,
    error_scale: float = 0.4,
    n_in: int = 4,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Uniform random tracking-error batch in ``[-scale/2, +scale/2]``."""
    return (torch.rand(batch_size, n_in, device=device, dtype=dtype) - 0.5) * error_scale


@torch.no_grad()
def pd_target(
    errors: torch.Tensor,
    teacher: ClassicalPDController,
) -> torch.Tensor:
    """PD feedback label for a batch of errors: ``(B, 2)`` saturated commands."""
    kp, kd = teacher.kp, teacher.kd
    tx = kp * errors[:, 0] + kd * errors[:, 2]
    ty = kp * errors[:, 1] + kd * errors[:, 3]
    return torch.stack([tx, ty], dim=1).clamp(-teacher.max_tilt, teacher.max_tilt)


def train_dense_controller(
    controller: DenseNNController,
    teacher: Optional[ClassicalPDController] = None,
    config: Optional[TrainingConfig] = None,
    *,
    log_fn=None,
) -> Dict[str, list]:
    """Distil the PD law into a dense ANN. Returns a loss history."""
    config = config or TrainingConfig()
    teacher = teacher or ClassicalPDController(device=config.device)
    device = _device_of(controller)
    dtype = next(controller.parameters()).dtype

    torch.manual_seed(config.seed)
    opt = optim.Adam(controller.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    history: List[float] = []

    controller.train()
    for epoch in range(1, config.epochs + 1):
        errs = sample_errors(config.batch_size, config.error_scale, controller.n_in, device, dtype)
        targets = pd_target(errs, teacher)

        opt.zero_grad()
        pred = controller.net(errs)  # unsaturated head; loss sees the raw output
        loss = criterion(pred.clamp(-controller.max_tilt, controller.max_tilt), targets)
        loss.backward()
        opt.step()

        history.append(float(loss.item()))
        if log_fn and (epoch % config.log_every == 0 or epoch == 1):
            log_fn("dense", epoch, history[-1])
    controller.eval()
    return {"loss": history, "final_loss": history[-1] if history else float("nan")}


def train_connectome_controller(
    controller: ConnectomeANNController,
    teacher: Optional[ClassicalPDController] = None,
    config: Optional[TrainingConfig] = None,
    *,
    log_fn=None,
) -> Dict[str, list]:
    """Distil the PD law into the sparse recurrent connectome ANN."""
    config = config or TrainingConfig()
    teacher = teacher or ClassicalPDController(device=config.device)
    device = _device_of(controller)
    dtype = next(controller.parameters()).dtype

    torch.manual_seed(config.seed)
    opt = optim.Adam(controller.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    history: List[float] = []

    controller.train()
    for epoch in range(1, config.epochs + 1):
        errs = sample_errors(config.batch_size, config.error_scale, controller.n_in, device, dtype)
        targets = pd_target(errs, teacher)

        opt.zero_grad()
        w_sparse = controller.get_sparse_matrix()
        h = torch.zeros(errs.shape[0], controller.n_neurons, device=device, dtype=dtype)
        for _ in range(config.connectome_unroll):
            recurrent_drive = torch.sparse.mm(w_sparse, h.T).T
            h = controller.relu(controller.w_in(errs) + recurrent_drive)
        pred = controller.w_out(h).clamp(-controller.max_tilt, controller.max_tilt)

        loss = criterion(pred, targets)
        loss.backward()
        opt.step()

        history.append(float(loss.item()))
        if log_fn and (epoch % config.log_every == 0 or epoch == 1):
            log_fn("connectome", epoch, history[-1])
    controller.eval()
    return {"loss": history, "final_loss": history[-1] if history else float("nan")}


def distill(
    network: Optional[NetworkConfig] = None,
    config: Optional[TrainingConfig] = None,
    *,
    seed: Optional[int] = None,
    dense: Optional[DenseNNController] = None,
    connectome: Optional[ConnectomeANNController] = None,
    log_fn=None,
) -> Dict[str, object]:
    """Train *both* students and return them with their loss histories.

    Returns
    -------
    dict with keys ``dense``, ``connectome``, ``history``, ``config``.
    """
    network = network or NetworkConfig()
    config = config or TrainingConfig()
    if seed is not None:
        config = dataclasses.replace(config, seed=seed)

    device = config.device
    if dense is None:
        dense = DenseNNController(
            n_in=network.n_in,
            n_neurons=network.n_neurons,
            n_out=network.n_out,
            device=device,
            seed=config.seed,
        )
    if connectome is None:
        connectome = ConnectomeANNController(
            n_in=network.n_in,
            n_neurons=network.n_neurons,
            n_out=network.n_out,
            synapses_per_neuron=network.synapses_per_neuron,
            inhibitory_fraction=network.inhibitory_fraction,
            excitatory_weight=network.excitatory_weight,
            inhibitory_weight=network.inhibitory_weight,
            seed=config.seed,
            device=device,
        )

    teacher = ClassicalPDController(device=device)
    dense_hist = train_dense_controller(dense, teacher, config, log_fn=log_fn)
    conn_hist = train_connectome_controller(connectome, teacher, config, log_fn=log_fn)

    return {
        "dense": dense,
        "connectome": connectome,
        "teacher": teacher,
        "history": {"dense": dense_hist["loss"], "connectome": conn_hist["loss"]},
        "final_loss": {
            "dense": dense_hist["final_loss"],
            "connectome": conn_hist["final_loss"],
        },
        "config": config.to_dict(),
    }


# --------------------------------------------------------------------------- #
# Weight serialisation (used by the CLI and by the web backend's cache)
# --------------------------------------------------------------------------- #
def save_weights(
    path: str,
    *,
    dense: Optional[DenseNNController] = None,
    connectome: Optional[ConnectomeANNController] = None,
    training: Optional[dict] = None,
) -> str:
    """Persist trained controller weights to a single ``.pt`` bundle."""
    bundle = {"format": "ann2snn.sim_engine.weights@1"}
    if dense is not None:
        bundle["dense"] = {
            "state_dict": dense.state_dict(),
            "n_neurons": dense.n_neurons,
            "n_in": dense.n_in,
        }
    if connectome is not None:
        bundle["connectome"] = {
            "state_dict": connectome.state_dict(),
            "n_neurons": connectome.n_neurons,
            "n_in": connectome.n_in,
            "topology": connectome.topology.to_dict(),
            "seed": connectome.topology.seed,
            # topology *shape* parameters are needed to rebuild an identical net
            "synapses_per_neuron": connectome.topology.synapses_per_neuron,
            "inhibitory_fraction": float(connectome.topology.is_inhibitory.float().mean().item()),
            "excitatory_weight": float(connectome.topology.polarity.max().item()),
            "inhibitory_weight": float(connectome.topology.polarity.min().item()),
        }
    if training is not None:
        bundle["training"] = training
    torch.save(bundle, path)
    return path


def load_weights(path: str, device="cpu") -> dict:
    """Load a bundle produced by :func:`save_weights` (``weights_only=False``).

    Returns ``{"dense": ..., "connectome": ..., "training": ...}`` with freshly
    constructed controllers that already hold the saved weights.  ``training`` is
    the optional metadata saved alongside the weights (loss history, final loss
    and the :class:`TrainingConfig`), or ``None`` for older bundles.
    """
    bundle = torch.load(path, map_location=device, weights_only=False)
    if bundle.get("format") != "ann2snn.sim_engine.weights@1":
        raise ValueError(f"unrecognised weights bundle format: {bundle.get('format')!r}")

    out: Dict[str, object] = {"training": bundle.get("training")}
    if "dense" in bundle:
        meta = bundle["dense"]
        ctrl = DenseNNController(
            n_in=meta.get("n_in", 4), n_neurons=meta["n_neurons"], device=device
        )
        ctrl.load_state_dict(meta["state_dict"])
        ctrl.eval()
        out["dense"] = ctrl
    if "connectome" in bundle:
        meta = bundle["connectome"]
        seed = meta.get("seed", 42)
        ctrl = ConnectomeANNController(
            n_in=meta.get("n_in", 4),
            n_neurons=meta["n_neurons"],
            synapses_per_neuron=meta.get("synapses_per_neuron", SYNAPSES_PER_NEURON),
            inhibitory_fraction=meta.get("inhibitory_fraction", 0.20),
            excitatory_weight=meta.get("excitatory_weight", 1.0),
            inhibitory_weight=meta.get("inhibitory_weight", -4.0),
            seed=seed,
            device=device,
        )
        ctrl.load_state_dict(meta["state_dict"])
        ctrl.eval()
        out["connectome"] = ctrl
    return out
