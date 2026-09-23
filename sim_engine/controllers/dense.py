"""Dense feed-forward neural controller (1,000 ReLU units).

This is the "unconstrained dense" benchmark arm: a plain MLP mapping the
4-D tracking error straight to the 2-D plate command.

Two flavours are supported through the same class:

* **random** — freshly initialised, never trained.  Used by the interactive
  demo as the "Random ANN" baseline.
* **distilled** — trained by :mod:`sim_engine.training` to imitate the PD
  controller ("behavioral distillation").

The class is a :class:`torch.nn.Module` *and* a :class:`BaseController`, so it
can be handed to an optimiser and to the benchmark loop alike.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..physics import C_CONST, MAX_TILT, N_IN, N_NEURONS, N_OUT
from .base import BaseController

__all__ = ["DenseNNController"]


class DenseNNController(nn.Module, BaseController):
    """``Linear(4,1000) -> ReLU -> Linear(1000,2)``, no biases."""

    name = "dense_ann"
    description = "Dense feed-forward ANN (1000 ReLU units)"
    recurrent = False
    spiking = False
    uses_feedforward = False

    def __init__(
        self,
        n_in: int = N_IN,
        n_neurons: int = N_NEURONS,
        n_out: int = N_OUT,
        max_tilt: float = MAX_TILT,
        device="cpu",
        dtype=torch.float32,
        seed: int | None = None,
        plant_gain: float = -C_CONST,
        pos_dim: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        BaseController.__init__(
            self, n_in=n_in, n_out=n_out, device=device,
            action_limit=max_tilt, plant_gain=plant_gain, pos_dim=pos_dim,
        )
        self.n_neurons = n_neurons
        self.max_tilt = max_tilt

        if seed is not None:
            # Local RNG so seeding one controller cannot perturb another.
            gen_state = torch.get_rng_state()
            torch.manual_seed(seed)
        try:
            self.net = nn.Sequential(
                nn.Linear(n_in, n_neurons, bias=False),
                nn.ReLU(),
                nn.Linear(n_neurons, n_out, bias=False),
            )
        finally:
            if seed is not None:
                torch.set_rng_state(gen_state)

        self.to(device=device, dtype=dtype)

    # nn.Module machinery --------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Saturated actuator command for a batch ``x`` of policy inputs."""
        return torch.clamp(self.net(x), -self.action_limit, self.action_limit)

    # BaseController machinery --------------------------------------------- #
    def reset(self) -> None:
        super().reset()

    def raw_act(self, state: torch.Tensor, ref) -> torch.Tensor:
        return self.net(self._input(state, ref))

    def act(self, state: torch.Tensor, ref) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(self._input(state, ref))

    # helpers --------------------------------------------------------------- #
    def _dtype(self):
        return next(self.parameters()).dtype

    def _input(self, state: torch.Tensor, ref) -> torch.Tensor:
        """Policy input ``[e, u_ff]`` from the (Kalman) estimate and reference."""
        return self.policy_input(state, ref).to(self.device, dtype=self._dtype())

    def describe(self) -> dict:
        d = BaseController.describe(self)
        d.update({"n_neurons": self.n_neurons,
                  "architecture": f"{self.n_in}-{self.n_neurons}-{self.n_out} relu"})
        return d
