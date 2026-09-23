"""Sparse recurrent connectome ANN with Dale's law ("fly-like" brain).

Topology model
--------------
* 1,000 neurons, 40 incoming synapses each (40,000 edges).
* 20 % of neurons are inhibitory, 80 % excitatory (Dale's principle: a neuron's
  outgoing synapses all share its sign).
* Inhibitory weights are scaled x4 relative to excitatory ones to keep the
  network in the balanced E/I regime — the same trick the prototype used to
  avoid runaway excitation.

The recurrent weight *magnitudes* are trainable; the topology and the sign of
each synapse are fixed at construction, so behavioural distillation can only
change how strongly an existing anatomical connection is expressed.  This is
what makes the network biologically plausible and distinct from
:class:`~sim_engine.controllers.dense.DenseNNController`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..physics import (
    C_CONST,
    MAX_TILT,
    N_IN,
    N_NEURONS,
    N_OUT,
    SYNAPSES_PER_NEURON,
    TOTAL_SYNAPSES,
)
from .base import BaseController, error_vector

if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
    # Same global opt-out the prototype used: the topology is built from random
    # integer indices that may legitimately contain duplicates pre-coalesce.
    torch.sparse.check_sparse_tensor_invariants.disable()

__all__ = ["ConnectomeTopology", "ConnectomeANNController"]


class ConnectomeTopology:
    """Immutable anatomy: edge list, E/I polarity and synaptic signs.

    Parameters
    ----------
    n_neurons:
        Number of neurons ``N``.
    synapses_per_neuron:
        Fan-in ``K``; total edges ``N * K``.
    inhibitory_fraction:
        20 % by default (Dale's law / balanced E-I).
    excitatory_weight / inhibitory_weight:
        Sign-and-scale of outgoing synapses.
    seed:
        Deterministic topology generation. Two topologies built with the same
        seed are identical, which the tests rely on.
    """

    def __init__(
        self,
        n_neurons: int = N_NEURONS,
        synapses_per_neuron: int = SYNAPSES_PER_NEURON,
        inhibitory_fraction: float = 0.20,
        excitatory_weight: float = 1.0,
        inhibitory_weight: float = -4.0,
        seed: int = 42,
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        self.n_neurons = n_neurons
        self.synapses_per_neuron = synapses_per_neuron
        self.total_synapses = n_neurons * synapses_per_neuron
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        # 1) Dale's law: assign E/I identity per *presynaptic* neuron.
        is_inhibitory = torch.rand(n_neurons, generator=gen) < inhibitory_fraction
        polarity = torch.where(
            is_inhibitory,
            torch.tensor(inhibitory_weight, dtype=dtype),
            torch.tensor(excitatory_weight, dtype=dtype),
        )

        # 2) Random edge list (dst, src) — src is the presynaptic neuron.
        src = torch.randint(0, n_neurons, (self.total_synapses,), generator=gen)
        dst = torch.randint(0, n_neurons, (self.total_synapses,), generator=gen)
        indices = torch.stack([dst, src])

        # 3) Initial magnitudes: signs live in `polarity`, so we only learn a
        #    non-negative magnitude per edge.
        init_weights = torch.abs(torch.randn(self.total_synapses, generator=gen))
        init_weights = init_weights * (0.25 / (synapses_per_neuron ** 0.5))

        self.is_inhibitory = is_inhibitory.to(device)
        self.polarity = polarity.to(device=device, dtype=dtype)
        self.indices = indices.to(device=device)
        self.init_weights = init_weights.to(device=device, dtype=dtype)

    def signed_weights(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Apply Dale's law to learned magnitudes: ``|w| * polarity[src]``."""
        src = self.indices[1]
        return torch.abs(magnitudes) * self.polarity[src]

    def to_dict(self) -> dict:
        return {
            "n_neurons": self.n_neurons,
            "synapses_per_neuron": self.synapses_per_neuron,
            "total_synapses": self.total_synapses,
            "inhibitory_fraction": float(self.is_inhibitory.float().mean().item()),
            "excitatory_weight": float(self.polarity.max().item()),
            "inhibitory_weight": float(self.polarity.min().item()),
            "seed": self.seed,
        }


class ConnectomeANNController(nn.Module, BaseController):
    """Sparse recurrent ANN with input/output projections.

    Dynamics (one control frame)::

        h <- ReLU( W_in @ err  +  W_rec_sparse @ h_prev )
        u <- clamp( W_out @ h )
    """

    name = "flylike_ann"
    description = "Sparse recurrent connectome ANN (Dale's law, E/I balanced)"
    recurrent = True
    spiking = False
    uses_feedforward = False

    def __init__(
        self,
        n_in: int = N_IN,
        n_neurons: int = N_NEURONS,
        n_out: int = N_OUT,
        synapses_per_neuron: int = SYNAPSES_PER_NEURON,
        inhibitory_fraction: float = 0.20,
        excitatory_weight: float = 1.0,
        inhibitory_weight: float = -4.0,
        max_tilt: float = MAX_TILT,
        seed: int = 42,
        device="cpu",
        dtype=torch.float32,
        topology: ConnectomeTopology | None = None,
        plant_gain: float = -C_CONST,
        pos_dim: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        BaseController.__init__(
            self, n_in=n_in, n_out=n_out, device=device,
            action_limit=max_tilt, plant_gain=plant_gain, pos_dim=pos_dim,
        )
        self.n_neurons = n_neurons
        self.n_out = n_out
        self.max_tilt = max_tilt
        self.dtype = dtype

        self.topology = topology or ConnectomeTopology(
            n_neurons=n_neurons,
            synapses_per_neuron=synapses_per_neuron,
            inhibitory_fraction=inhibitory_fraction,
            excitatory_weight=excitatory_weight,
            inhibitory_weight=inhibitory_weight,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        self.n_neurons = self.topology.n_neurons

        self.w_in = nn.Linear(n_in, n_neurons, bias=False)
        self.w_out = nn.Linear(n_neurons, n_out, bias=False)
        self.raw_weights = nn.Parameter(self.topology.init_weights.clone())
        self.relu = nn.ReLU()

        self._h: torch.Tensor | None = None
        self.to(device=device, dtype=dtype)

    # -- sparse matrix ------------------------------------------------------ #
    def get_sparse_matrix(self) -> torch.Tensor:
        """Coalesced sparse recurrence matrix, signed by Dale's law."""
        w = self.topology.signed_weights(self.raw_weights)
        return torch.sparse_coo_tensor(
            self.topology.indices.to(self.raw_weights.device),
            w,
            (self.n_neurons, self.n_neurons),
        ).coalesce()

    # -- nn.Module.forward -------------------------------------------------- #
    def forward(self, err_state: torch.Tensor, h_prev: torch.Tensor):
        """One recurrent step for a batch of errors.

        ``err_state``: ``(B, n_in)``; ``h_prev``: ``(B, n_neurons)``.
        Returns ``(tilt (B, n_out), h_next (B, n_neurons))``.
        """
        w_sparse = self.get_sparse_matrix()
        recurrent_drive = torch.sparse.mm(w_sparse, h_prev.T).T
        h = self.relu(self.w_in(err_state) + recurrent_drive)
        tilt = torch.clamp(self.w_out(h), -self.action_limit, self.action_limit)
        return tilt, h

    # -- BaseController ----------------------------------------------------- #
    def reset(self) -> None:
        super().reset()
        self._h = None

    def hidden_state(self, batch: int = 1, dtype=None, device=None) -> torch.Tensor:
        return torch.zeros(
            batch,
            self.n_neurons,
            dtype=dtype or self.dtype,
            device=device or self.device,
        )

    def _input(self, state: torch.Tensor, ref) -> torch.Tensor:
        return self.policy_input(state, ref).to(self.device, dtype=self.dtype)

    def raw_act(self, state: torch.Tensor, ref) -> torch.Tensor:
        err = self._input(state, ref).unsqueeze(0)
        h_prev = self._h if self._h is not None else self.hidden_state(1)
        tilt, h = self.forward(err, h_prev)
        self._h = h
        return tilt.squeeze(0)

    def act(self, state: torch.Tensor, ref) -> torch.Tensor:
        with torch.no_grad():
            return self.raw_act(state, ref)

    def describe(self) -> dict:
        d = BaseController.describe(self)
        d.update({"n_neurons": self.n_neurons, "topology": self.topology.to_dict()})
        return d
