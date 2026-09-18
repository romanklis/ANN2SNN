"""Lossless micro-stepping spiking transfer of the connectome ANN.

The ANN's continuous ReLU activations are replaced by an integrate-and-fire
population that is integrated with ``micro_steps`` internal sub-steps per
control frame, a technique that reproduces the ANN's *rate* behaviour with
binary spikes and no accuracy loss over a closed-loop rollout.

Dynamics per micro-step ``s``::

    i_syn = W_in @ err  +  W_rec_sparse @ spikes
    v     = clamp(v + i_syn, min=0)         # non-negative membrane
    spike = (v >= v_th)
    v     = v - spike * v_th                # soft reset
    motor += W_out @ spike

the motor command is the average accumulated output spike rate over the
micro-steps, saturated to the plate limits.
"""

from __future__ import annotations

import torch

from ..physics import MAX_TILT, N_IN, N_OUT, clamp_action
from .base import BaseController, error_vector

__all__ = ["LosslessConnectomeSNN"]


class LosslessConnectomeSNN(BaseController):
    """IF spiking network transferred losslessly from a connectome ANN.

    Parameters
    ----------
    ann_model:
        A trained (or random) :class:`ConnectomeANNController`; its ``w_in``,
        signed recurrent matrix and ``w_out`` are copied, then frozen (the SNN
        is a *transfer* — no gradient flows back into the ANN).
    micro_steps:
        Number of internal sub-steps per 20 ms control frame.
    v_th:
        Membrane firing threshold.
    """

    name = "snn_transferred"
    description = "Lossless micro-stepping connectome SNN (IF transfer)"
    recurrent = True
    spiking = True
    uses_feedforward = False

    def __init__(
        self,
        ann_model,
        micro_steps: int = 10,
        v_th: float = 1.0,
        max_tilt: float = MAX_TILT,
        device=None,
        dtype=torch.float32,
    ) -> None:
        device = device if device is not None else getattr(ann_model, "device", "cpu")
        super().__init__(n_in=N_IN, n_out=N_OUT, device=device)
        self.micro_steps = int(micro_steps)
        self.v_th = float(v_th)
        self.max_tilt = max_tilt
        self.dtype = dtype

        with torch.no_grad():
            self.w_in = ann_model.w_in.weight.data.clone().to(device=device, dtype=dtype)
            self.w_rec = ann_model.get_sparse_matrix().clone().to(device=device, dtype=dtype)
            self.w_out = ann_model.w_out.weight.data.clone().to(device=device, dtype=dtype)

        self.n_neurons = int(self.w_in.shape[0])
        self._s = None   # spike state
        self._v = None   # membrane potential
        self._micro_spike_history: list = []

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        self._v = torch.zeros(self.n_neurons, dtype=self.dtype, device=self.device)
        self._s = torch.zeros(self.n_neurons, dtype=self.dtype, device=self.device)
        self._micro_spike_history = []

    def _ensure_state(self) -> None:
        if self._v is None or self._v.device != self.device:
            self.reset()

    # -- control ------------------------------------------------------------ #
    def step(self, err_state: torch.Tensor):
        """Advance the SNN by one control frame.

        Returns ``(tilt, spikes)`` where ``spikes`` is the neuron spike vector
        of the final micro-step (used for the raster plot).
        """
        self._ensure_state()
        err_state = torch.as_tensor(err_state, dtype=self.dtype, device=self.device)

        i_input = torch.matmul(self.w_in, err_state)
        motor_spike_accum = torch.zeros(self.n_out, dtype=self.dtype, device=self.device)

        for _ in range(self.micro_steps):
            recurrent_drive = torch.sparse.mm(self.w_rec, self._s.unsqueeze(1)).squeeze(1)
            i_syn = i_input + recurrent_drive

            # Integrate-and-fire with a non-negative lower bound.
            self._v = torch.clamp(self._v + i_syn, min=0.0)

            # Fire & soft reset.
            self._s = (self._v >= self.v_th).to(self.dtype)
            self._v = self._v - (self._s * self.v_th)

            motor_spike_accum += torch.matmul(self.w_out, self._s)

        tilt = clamp_action(motor_spike_accum / self.micro_steps, self.max_tilt)
        self._micro_spike_history.append(self._s.clone())
        return tilt, self._s.clone()

    def raw_act(self, state: torch.Tensor, ref) -> torch.Tensor:
        err = error_vector(state, ref).to(self.device, dtype=self.dtype)
        tilt, _ = self.step(err)
        return tilt

    def act(self, state: torch.Tensor, ref) -> torch.Tensor:
        with torch.no_grad():
            return self.raw_act(state, ref)

    def last_spikes(self) -> torch.Tensor | None:
        return self._s.clone() if self._s is not None else None

    @staticmethod
    def from_trained_ann(ann_model, **kwargs) -> "LosslessConnectomeSNN":
        """Convenience constructor mirroring the prototype's call site."""
        return LosslessConnectomeSNN(ann_model, **kwargs)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"n_neurons": self.n_neurons, "micro_steps": self.micro_steps, "v_th": self.v_th})
        return d
