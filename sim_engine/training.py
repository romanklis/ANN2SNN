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

from .config import BenchmarkConfig, EmbodimentConfig, NetworkConfig, TrainingConfig
from .controllers import ClassicalPDController, ConnectomeANNController, DenseNNController
from .controllers.base import error_vector
from .environment import EmbodiedEnv
from .physics import MAX_TILT, SYNAPSES_PER_NEURON
from .reference import orbit_reference

__all__ = [
    "sample_errors",
    "pd_target",
    "train_dense_controller",
    "train_connectome_controller",
    "distill",
    "distill_embodied",
    "distill_profile",
    "generate_embodied_episodes",
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
# Embodied ("robust") distillation: distil the PD teacher inside the env
# --------------------------------------------------------------------------- #
def generate_embodied_episodes(
    teacher: ClassicalPDController,
    config: TrainingConfig,
    benchmark: Optional[BenchmarkConfig] = None,
    *,
    log_fn=None,
) -> list:
    """Roll the PD teacher inside an embodied env and collect (error, label) pairs.

    The teacher observes exactly what a student will observe (noisy, delayed) and
    acts through the same actuator model, so the labels are the deployed control
    law rather than the ideal one.  Returns one dict per episode with ``errors``
    ``(E, 4)`` and ``labels`` ``(E, 2)``.
    """
    benchmark = benchmark or BenchmarkConfig(steps=int(config.episode_steps))
    embodiment = config.embodiment or EmbodimentConfig.from_preset("embodied")
    steps = int(config.episode_steps)
    ref = orbit_reference(steps=steps, radius=benchmark.radius, freq=benchmark.freq)
    env = EmbodiedEnv(embodiment, dt=ref.dt, max_tilt=MAX_TILT, device=config.device)

    episodes = []
    for ep in range(int(config.episodes)):
        env.reset(seed=int(config.seed) + ep)
        state = torch.tensor([-0.05, 0.05, 0.0, 0.0], dtype=torch.float32, device=config.device)
        errs, labels = [], []
        with torch.no_grad():
            for k in range(steps):
                observation = env.observe(state)
                err = error_vector(observation, ref.at(k))
                errs.append(err)
                labels.append(pd_target(err.unsqueeze(0), teacher)[0])
                u = teacher.act(observation, ref.at(k))
                state = env.step(state, env.actuate(u), k)
        episodes.append({"errors": torch.stack(errs), "labels": torch.stack(labels)})
        if log_fn:
            log_fn("data", ep + 1, 0.0)
    return episodes


def train_dense_embodied(
    controller: DenseNNController,
    episodes: list,
    config: Optional[TrainingConfig] = None,
    *,
    log_fn=None,
) -> Dict[str, object]:
    """Distil pooled embodied (error, label) pairs into the dense ANN."""
    config = config or TrainingConfig()
    device = _device_of(controller)
    dtype = next(controller.parameters()).dtype
    torch.manual_seed(config.seed)
    opt = optim.Adam(controller.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    history: List[float] = []

    errs = torch.cat([e["errors"] for e in episodes]).to(device, dtype)
    labels = torch.cat([e["labels"] for e in episodes]).to(device, dtype)
    n = int(errs.shape[0])
    if n == 0:
        return {"loss": history, "final_loss": float("nan")}

    controller.train()
    for epoch in range(1, config.epochs + 1):
        idx = torch.randint(0, n, (min(config.batch_size, n),), device=device)
        x, y = errs[idx], labels[idx]
        if config.noise_augment:
            x = x + torch.randn_like(x) * config.noise_augment
        opt.zero_grad()
        pred = controller.net(x).clamp(-controller.max_tilt, controller.max_tilt)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if log_fn and (epoch % config.log_every == 0 or epoch == 1):
            log_fn("dense", epoch, history[-1])
    controller.eval()
    return {"loss": history, "final_loss": history[-1] if history else float("nan")}


def train_connectome_embodied(
    controller: ConnectomeANNController,
    episodes: list,
    config: Optional[TrainingConfig] = None,
    *,
    log_fn=None,
) -> Dict[str, object]:
    """Distil embodied pairs into the recurrent connectome on contiguous windows.

    Windows let the recurrence integrate the delayed/noisy observation stream,
    which is the only way delay compensation can be learned.
    """
    config = config or TrainingConfig()
    device = _device_of(controller)
    dtype = next(controller.parameters()).dtype
    W = max(1, int(config.connectome_unroll))

    windows = []
    for e in episodes:
        errs_e, labels_e = e["errors"], e["labels"]
        if errs_e.shape[0] < W:
            continue
        if errs_e.shape[0] == W:
            windows.append((errs_e, labels_e))
        else:
            for _ in range(max(1, config.batch_size)):
                s = int(torch.randint(0, errs_e.shape[0] - W, (1,)).item())
                windows.append((errs_e[s:s + W], labels_e[s:s + W]))
    if not windows:
        return {"loss": [], "final_loss": float("nan")}

    errs = torch.stack([w[0] for w in windows]).to(device, dtype)      # (N, W, 4)
    labels = torch.stack([w[1] for w in windows]).to(device, dtype)    # (N, W, 2)
    n = int(errs.shape[0])

    torch.manual_seed(config.seed)
    opt = optim.Adam(controller.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    history: List[float] = []

    controller.train()
    for epoch in range(1, config.epochs + 1):
        idx = torch.randint(0, n, (min(config.batch_size, n),), device=device)
        x, y = errs[idx], labels[idx]
        if config.noise_augment:
            x = x + torch.randn_like(x) * config.noise_augment
        opt.zero_grad()
        w_sparse = controller.get_sparse_matrix()
        h = torch.zeros(x.shape[0], controller.n_neurons, device=device, dtype=dtype)
        preds = []
        for t in range(W):
            recurrent_drive = torch.sparse.mm(w_sparse, h.T).T
            h = controller.relu(controller.w_in(x[:, t, :]) + recurrent_drive)
            preds.append(controller.w_out(h).clamp(-controller.max_tilt, controller.max_tilt))
        pred = torch.stack(preds, dim=1)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if log_fn and (epoch % config.log_every == 0 or epoch == 1):
            log_fn("connectome", epoch, history[-1])
    controller.eval()
    return {"loss": history, "final_loss": history[-1] if history else float("nan")}


def distill_embodied(
    network: Optional[NetworkConfig] = None,
    config: Optional[TrainingConfig] = None,
    *,
    dense: Optional[DenseNNController] = None,
    connectome: Optional[ConnectomeANNController] = None,
    benchmark: Optional[BenchmarkConfig] = None,
    log_fn=None,
) -> Dict[str, object]:
    """Robust profile: distil both students from the embodied PD teacher."""
    network = network or NetworkConfig()
    config = config or TrainingConfig(profile="robust")
    if config.profile != "robust":
        config = dataclasses.replace(config, profile="robust")

    device = config.device
    if dense is None:
        dense = DenseNNController(
            n_in=network.n_in, n_neurons=network.n_neurons,
            n_out=network.n_out, device=device, seed=config.seed,
        )
    if connectome is None:
        connectome = ConnectomeANNController(
            n_in=network.n_in, n_neurons=network.n_neurons, n_out=network.n_out,
            synapses_per_neuron=network.synapses_per_neuron,
            inhibitory_fraction=network.inhibitory_fraction,
            excitatory_weight=network.excitatory_weight,
            inhibitory_weight=network.inhibitory_weight,
            seed=config.seed, device=device,
        )

    teacher = ClassicalPDController(device=device)
    episodes = generate_embodied_episodes(teacher, config, benchmark, log_fn=log_fn)
    dense_hist = train_dense_embodied(dense, episodes, config, log_fn=log_fn)
    conn_hist = train_connectome_embodied(connectome, episodes, config, log_fn=log_fn)

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
        "episodes": len(episodes),
    }


def distill_profile(
    profile: str = "clean",
    network: Optional[NetworkConfig] = None,
    config: Optional[TrainingConfig] = None,
    *,
    dense: Optional[DenseNNController] = None,
    connectome: Optional[ConnectomeANNController] = None,
    benchmark: Optional[BenchmarkConfig] = None,
    log_fn=None,
) -> Dict[str, object]:
    """Dispatch to the clean or robust distillation path."""
    config = config or TrainingConfig()
    if profile and profile != config.profile:
        config = dataclasses.replace(config, profile=profile)
    if config.profile == "robust":
        return distill_embodied(
            network, config, dense=dense, connectome=connectome,
            benchmark=benchmark, log_fn=log_fn,
        )
    return distill(network, config, dense=dense, connectome=connectome, log_fn=log_fn)


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
