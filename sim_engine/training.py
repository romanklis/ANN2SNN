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

The policy input is ``[e, u_ff]`` (6-D): the tracking error plus the feed-forward
command ``u_ff = -a_ref/C`` reconstructed from the Kalman estimate (position-only
camera, known rolling gain).  Labels reproduce the **full** analytic teacher
``kp*e + kd*e_dot + u_ff``, so the learned arms are trained on the same channel the
PID reference arm uses.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .config import BenchmarkConfig, EmbodimentConfig, NetworkConfig, TrainingConfig
from .controllers import ClassicalPDController, ConnectomeANNController, DenseNNController
from .controllers.base import ReferenceAccelEstimator, error_vector
from .environment import EmbodiedEnv
from .estimators import KalmanFilter
from .examples import get_example
from .physics import MAX_TILT, N_IN, N_OUT, SYNAPSES_PER_NEURON
from .reference import orbit_reference

__all__ = [
    "sample_errors",
    "sample_policy_inputs",
    "pd_target",
    "train_dense_controller",
    "train_connectome_controller",
    "distill",
    "distill_embodied",
    "distill_closed_loop",
    "distill_profile",
    "generate_closed_loop_episodes",
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


def sample_policy_inputs(
    batch_size: int = 64,
    error_scale: float = 0.4,
    n_in: int = N_IN,
    device="cpu",
    dtype=torch.float32,
    ff_scale: float = MAX_TILT,
) -> torch.Tensor:
    """Random policy-input batch ``[e, u_ff]``.

    The first four columns are uniform tracking errors; when ``n_in >= 6`` the
    last two are a uniform feed-forward command in ``[-ff_scale, +ff_scale]`` so
    training covers both channels.
    """
    errs = sample_errors(batch_size, error_scale, max(2 * (n_in // 3), 4), device, dtype)
    extra = n_in - errs.shape[1]
    if extra <= 0:
        return errs[:, :n_in]
    ff = (torch.rand(batch_size, extra, device=device, dtype=dtype) * 2.0 - 1.0) * ff_scale
    return torch.cat([errs, ff], dim=1)


@torch.no_grad()
def pd_target(
    inputs: torch.Tensor,
    teacher: ClassicalPDController,
) -> torch.Tensor:
    """PD label for a batch of policy inputs ``(B, D)`` (``D = teacher.pos_dim``).

    Matches the analytic teacher exactly: ``kp*e + kd*ė + u_ff`` per axis (the
    feed-forward columns are the reconstructed reference acceleration command).
    """
    d = int(teacher.pos_dim)
    g = teacher.plant_gain
    # label = (−ω²·e − 2ζω·ė)/G + u_ff  (corrective feedback for the signed gain)
    cmd = -(teacher.omega_n ** 2 / g) * inputs[:, :d] \
        - (2.0 * teacher.zeta * teacher.omega_n / g) * inputs[:, d:2 * d]
    if inputs.shape[1] >= 3 * d:
        cmd = cmd + inputs[:, 2 * d:3 * d]
    return cmd.clamp(-teacher.action_limit, teacher.action_limit)


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
        inputs = sample_policy_inputs(
            config.batch_size, config.error_scale, controller.n_in, device, dtype,
            ff_scale=controller.action_limit,
        )
        targets = pd_target(inputs, teacher)

        opt.zero_grad()
        pred = controller.net(inputs)  # unsaturated head; loss sees the raw output
        loss = criterion(pred.clamp(-controller.action_limit, controller.action_limit), targets)
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
        errs = sample_policy_inputs(
            config.batch_size, config.error_scale, controller.n_in, device, dtype,
            ff_scale=controller.action_limit,
        )
        targets = pd_target(errs, teacher)

        opt.zero_grad()
        w_sparse = controller.get_sparse_matrix()
        h = torch.zeros(errs.shape[0], controller.n_neurons, device=device, dtype=dtype)
        for _ in range(config.connectome_unroll):
            recurrent_drive = torch.sparse.mm(w_sparse, h.T).T
            h = controller.relu(controller.w_in(errs) + recurrent_drive)
        pred = controller.w_out(h).clamp(-controller.action_limit, controller.action_limit)

        loss = criterion(pred, targets)
        loss.backward()
        opt.step()

        history.append(float(loss.item()))
        if log_fn and (epoch % config.log_every == 0 or epoch == 1):
            log_fn("connectome", epoch, history[-1])
    controller.eval()
    return {"loss": history, "final_loss": history[-1] if history else float("nan")}


def _student_factories(spec, network: NetworkConfig, device: str, seed: int,
                       dense=None, connectome=None):
    """Build dense/connectome students sized and configured for ``spec``."""
    if dense is None:
        dense = DenseNNController(
            n_in=spec.n_in, n_neurons=network.n_neurons, n_out=spec.n_out,
            device=device, seed=seed,
            plant_gain=spec.plant_gain, pos_dim=spec.pos_dim,
            max_tilt=spec.control_limit,
        )
    if connectome is None:
        connectome = ConnectomeANNController(
            n_in=spec.n_in, n_neurons=network.n_neurons, n_out=spec.n_out,
            synapses_per_neuron=network.synapses_per_neuron,
            inhibitory_fraction=network.inhibitory_fraction,
            excitatory_weight=network.excitatory_weight,
            inhibitory_weight=network.inhibitory_weight,
            seed=seed, device=device,
            plant_gain=spec.plant_gain, pos_dim=spec.pos_dim,
            max_tilt=spec.control_limit,
        )
    return dense, connectome


def _teacher_for(spec, device: str, *, use_feedforward: bool = True) -> ClassicalPDController:
    """The analytic teacher, dimensioned and signed for ``spec``."""
    return ClassicalPDController(
        device=device, pos_dim=spec.pos_dim, plant_gain=spec.plant_gain,
        action_limit=spec.control_limit, use_feedforward=use_feedforward,
    )


def distill(
    network: Optional[NetworkConfig] = None,
    config: Optional[TrainingConfig] = None,
    *,
    seed: Optional[int] = None,
    dense: Optional[DenseNNController] = None,
    connectome: Optional[ConnectomeANNController] = None,
    example: Optional[str] = None,
    log_fn=None,
) -> Dict[str, object]:
    """Train *both* students and return them with their loss histories.

    Returns
    -------
    dict with keys ``dense``, ``connectome``, ``history``, ``config``.
    """
    spec = get_example(example)
    network = dataclasses.replace(
        network or NetworkConfig(), n_in=spec.n_in, n_out=spec.n_out
    )
    config = config or TrainingConfig()
    if seed is not None:
        config = dataclasses.replace(config, seed=seed)

    device = config.device
    dense, connectome = _student_factories(
        spec, network, device, config.seed, dense=dense, connectome=connectome
    )

    teacher = _teacher_for(spec, device)
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
def generate_closed_loop_episodes(
    teacher: ClassicalPDController,
    config: TrainingConfig,
    benchmark: Optional[BenchmarkConfig] = None,
    *,
    embodiment: Optional[EmbodimentConfig] = None,
    log_fn=None,
) -> list:
    """Roll the PD teacher through the closed loop (camera + Kalman) and label it.

    The teacher observes only ``x̂`` — the Kalman estimate of the full state from
    position-only measurements — and acts through the same actuator model, so the
    labels are the *deployed* control law ``π(r − x̂)`` rather than the idealized
    ``π(r − x)``.  Returns one dict per episode with ``errors`` ``(E, 4)`` and
    ``labels`` ``(E, 2)``.
    """
    benchmark = benchmark or BenchmarkConfig(steps=int(config.episode_steps))
    spec = get_example(benchmark.example)
    if embodiment is None:
        embodiment = config.embodiment or EmbodimentConfig.from_preset(
            "embodied" if config.profile == "robust" else "clean"
        )
    steps = int(config.episode_steps)
    ref = spec.reference(steps=steps, radius=benchmark.radius, freq=benchmark.freq)
    env = spec.make_env(embodiment, dt=ref.dt, device=config.device)
    kf = spec.make_estimator(embodiment, dt=ref.dt, device=config.device)

    episodes = []
    ff_est = ReferenceAccelEstimator(dt=ref.dt, plant_gain=spec.plant_gain, pos_dim=spec.pos_dim)
    for ep in range(int(config.episodes)):
        env.reset(seed=int(config.seed) + ep)
        kf.reset()
        ff_est.reset()
        state = torch.tensor(spec.init_state, dtype=torch.float32, device=config.device)
        u_prev = torch.zeros(spec.control_dim, dtype=torch.float32, device=config.device)
        inputs, labels = [], []
        with torch.no_grad():
            for k in range(steps):
                readings = env.sense(state)               # IMU + delayed channels
                xhat = kf.update(                         # Kalman estimate
                    readings=readings, control=u_prev, acceleration=readings.imu,
                )
                err = error_vector(xhat, ref.at(k))       # e = x̂ − r
                u_ff = ff_est.update(xhat, ref.at(k))     # KF-reconstructed feed-forward
                inp = torch.cat([err, u_ff])              # policy input [e, u_ff]
                inputs.append(inp)
                labels.append(pd_target(inp.unsqueeze(0), teacher)[0])
                u = teacher.act(xhat, ref.at(k))          # analytic teacher, full law
                state = env.step(state, env.actuate(u), k)
                u_prev = u
        episodes.append({"inputs": torch.stack(inputs), "labels": torch.stack(labels)})
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

    errs = torch.cat([e["inputs"] for e in episodes]).to(device, dtype)
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
        errs_e, labels_e = e["inputs"], e["labels"]
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
    example: Optional[str] = None,
    log_fn=None,
) -> Dict[str, object]:
    """Robust profile: distil both students from the embodied closed-loop teacher."""
    config = config or TrainingConfig(profile="robust")
    if config.profile != "robust":
        config = dataclasses.replace(config, profile="robust")
    return distill_closed_loop(
        "robust", network, config,
        dense=dense, connectome=connectome, benchmark=benchmark,
        example=example, log_fn=log_fn,
    )


def distill_closed_loop(
    profile: str = "clean",
    network: Optional[NetworkConfig] = None,
    config: Optional[TrainingConfig] = None,
    *,
    dense: Optional[DenseNNController] = None,
    connectome: Optional[ConnectomeANNController] = None,
    benchmark: Optional[BenchmarkConfig] = None,
    example: Optional[str] = None,
    log_fn=None,
) -> Dict[str, object]:
    """Distil π(r − x̂) from closed-loop teacher rollouts through the Kalman filter.

    Used for **both** profiles: ``clean`` runs the nominal plant (no measurement
    noise, no delay, no disturbances) but still estimates velocity from
    position-only measurements; ``robust`` runs the embodied environment.
    """
    spec = get_example(example or (benchmark.example if benchmark else None))
    network = dataclasses.replace(
        network or NetworkConfig(), n_in=spec.n_in, n_out=spec.n_out
    )
    config = config or TrainingConfig(profile=profile)
    if config.profile != profile:
        config = dataclasses.replace(config, profile=profile)
    profile = config.profile

    device = config.device
    dense, connectome = _student_factories(
        spec, network, device, config.seed, dense=dense, connectome=connectome
    )

    setup = EmbodimentConfig.from_preset("embodied" if profile == "robust" else "clean")
    teacher = _teacher_for(spec, device)
    episodes = generate_closed_loop_episodes(
        teacher, config, benchmark,
        embodiment=config.embodiment or setup,
        log_fn=log_fn,
    )
    n_episodes = len(episodes)

    # Coverage: the closed-loop teacher tracks so well that it visits only tiny
    # errors, so add i.i.d. samples labelled by the same PD law. Without this the
    # student never learns the large-error corrections it needs to recover.
    if int(config.coverage_samples) > 0:
        extra = sample_policy_inputs(
            int(config.coverage_samples), config.error_scale,
            network.n_in, device, next(dense.parameters()).dtype,
            ff_scale=spec.control_limit,
        )
        episodes.append({"inputs": extra, "labels": pd_target(extra, teacher)})

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
        "episodes": n_episodes,
    }


def distill_profile(
    profile: str = "clean",
    network: Optional[NetworkConfig] = None,
    config: Optional[TrainingConfig] = None,
    *,
    dense: Optional[DenseNNController] = None,
    connectome: Optional[ConnectomeANNController] = None,
    benchmark: Optional[BenchmarkConfig] = None,
    example: Optional[str] = None,
    log_fn=None,
) -> Dict[str, object]:
    """Distil the closed-loop teacher for the requested profile (clean or robust)."""
    config = config or TrainingConfig()
    if profile and profile != config.profile:
        config = dataclasses.replace(config, profile=profile)
    return distill_closed_loop(
        config.profile, network, config,
        dense=dense, connectome=connectome, benchmark=benchmark,
        example=example, log_fn=log_fn,
    )


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
    bundle = {"format": "ann2snn.sim_engine.weights@2"}
    if dense is not None:
        bundle["dense"] = {
            "state_dict": dense.state_dict(),
            "n_neurons": dense.n_neurons,
            "n_in": dense.n_in,
            "n_out": getattr(dense, "n_out", None),
        }
    if connectome is not None:
        bundle["connectome"] = {
            "state_dict": connectome.state_dict(),
            "n_neurons": connectome.n_neurons,
            "n_in": connectome.n_in,
            "n_out": getattr(connectome, "n_out", None),
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


def _infer_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[Optional[int], Optional[int]]:
    """``(n_in, n_out)`` from a controller's state dict, else ``(None, None)``.

    A bundle is dimensioned by its example (6→2 for the ball, 9→3 for the
    drones), so the loader must not rely on the controller class defaults.
    """
    for first, last in (("net.0.weight", "net.2.weight"), ("w_in.weight", "w_out.weight")):
        if first in state_dict and last in state_dict:
            return int(state_dict[first].shape[1]), int(state_dict[last].shape[0])
    keys = [k for k in state_dict if k.endswith("weight")]
    if keys:
        a, b = state_dict[keys[0]], state_dict[keys[-1]]
        return (int(a.shape[1]) if a.dim() == 2 else None,
                int(b.shape[0]) if b.dim() == 2 else None)
    return None, None


def load_weights(path: str, device="cpu") -> dict:
    """Load a bundle produced by :func:`save_weights` (``weights_only=False``).

    Returns ``{"dense": ..., "connectome": ..., "training": ...}`` with freshly
    constructed controllers that already hold the saved weights.  ``training`` is
    the optional metadata saved alongside the weights (loss history, final loss
    and the :class:`TrainingConfig`), or ``None`` for older bundles.

    The input/output widths are taken from the saved ``state_dict`` (falling back
    to the stored metadata), so a bundle for any example loads correctly rather
    than being forced into the default 6→2 policy shape.
    """
    bundle = torch.load(path, map_location=device, weights_only=False)
    fmt = bundle.get("format")
    if fmt != "ann2snn.sim_engine.weights@2":
        raise ValueError(
            f"unsupported weights bundle format {fmt!r}; expected "
            "'ann2snn.sim_engine.weights@2' (the 6-input policy channel changed "
            "the bundle format — retrain with `python -m sim_engine train`)"
        )

    out: Dict[str, object] = {"training": bundle.get("training")}
    if "dense" in bundle:
        meta = bundle["dense"]
        n_in, n_out = _infer_dims(meta["state_dict"])
        ctrl = DenseNNController(
            n_in=n_in if n_in is not None else meta.get("n_in", 4),
            n_out=n_out if n_out is not None else meta.get("n_out", N_OUT),
            n_neurons=meta["n_neurons"],
            device=device,
        )
        ctrl.load_state_dict(meta["state_dict"])
        ctrl.eval()
        out["dense"] = ctrl
    if "connectome" in bundle:
        meta = bundle["connectome"]
        seed = meta.get("seed", 42)
        n_in, n_out = _infer_dims(meta["state_dict"])
        ctrl = ConnectomeANNController(
            n_in=n_in if n_in is not None else meta.get("n_in", 4),
            n_out=n_out if n_out is not None else meta.get("n_out", N_OUT),
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
