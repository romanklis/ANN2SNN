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

from ..physics import N_IN, N_NEURONS, N_OUT, MAX_TILT, clamp_action
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
    ) -> None:
        nn.Module.__init__(self)
        BaseController.__init__(self, n_in=n_in, n_out=n_out, device=device)
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
        """Saturated actuator command for a batch ``x`` of error vectors."""
        return clamp_action(self.net(x), self.max_tilt)

    # BaseController machinery --------------------------------------------- #
    def raw_act(self, state: torch.Tensor, ref) -> torch.Tensor:
        from .base import error_vector

        err = error_vector(state, ref).to(self.device, dtype=self._dtype())
        return self.net(err)

    def act(self, state: torch.Tensor, ref) -> torch.Tensor:
        err = self._err(state, ref)
        with torch.no_grad():
            return self.forward(err)

    # helpers --------------------------------------------------------------- #
    def _dtype(self):
        return next(self.parameters()).dtype

    def _err(self, state: torch.Tensor, ref) -> torch.Tensor:
        from .base import error_vector

        return error_vector(state, ref).to(self.device, dtype=self._dtype())

    def describe(self) -> dict:
        d = BaseController.describe(self)
        d.update({"n_neurons": self.n_neurons, "architecture": "4-1000-2 relu"})
        return d
