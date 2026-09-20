"""Name -> controller factory registry shared by the CLI and the web backend.

The four *brains* a user can try out in the interactive demo all live here:

======================  ==========================================================
name                    controller
======================  ==========================================================
``pid``                 classical PD + feed-forward (``ClassicalPDController``)
``random_ann``          freshly initialised dense ANN (``DenseNNController``)
``flylike_ann``         sparse recurrent connectome ANN (``ConnectomeANNController``)
``snn_transferred``     lossless IF SNN transferred from the connectome ANN
``dense_ann``           distilled dense ANN (falls back to random if untrained)
======================  ==========================================================

Aliases such as ``pd``, ``connectome`` or ``snn`` resolve to the canonical name.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .config import NetworkConfig
from .controllers import (
    BaseController,
    ClassicalPDController,
    ConnectomeANNController,
    DenseNNController,
    LosslessConnectomeSNN,
)

__all__ = ["ControllerRegistry", "CANONICAL_CONTROLLERS", "EXTRA_CONTROLLERS",
           "ALIASES", "normalize_name"]

#: Canonical names in display order.
CANONICAL_CONTROLLERS: List[str] = [
    "pid",
    "random_ann",
    "flylike_ann",
    "snn_transferred",
    "dense_ann",
]

#: Diagnostic arms that are resolvable but not part of the headline catalogue.
#: ``pid_no_ff`` is the same PD law without the acceleration feed-forward, i.e.
#: the no-feed-forward performance limit the learned arms were previously stuck at.
EXTRA_CONTROLLERS: List[str] = ["pid_no_ff"]

#: Human labels for the frontend.
LABELS: Dict[str, str] = {
    "pid": "PID controlled",
    "random_ann": "Random ANN",
    "flylike_ann": "Fly-like ANN (connectome)",
    "snn_transferred": "SNN transferred",
    "dense_ann": "Dense ANN (distilled)",
}

ALIASES: Dict[str, str] = {
    # pid
    "pd": "pid", "classical": "pid", "classical_pd": "pid", "pdcontroller": "pid",
    "classicalpdcontroller": "pid", "pid_controlled": "pid",
    # random dense
    "random": "random_ann", "randomann": "random_ann", "dense_random": "random_ann",
    "randomdense": "random_ann", "untrained_ann": "random_ann",
    # fly-like connectome
    "connectome": "flylike_ann", "connectome_ann": "flylike_ann",
    "connectomeann": "flylike_ann", "flylike": "flylike_ann", "fly": "flylike_ann",
    "flylikeanncontroller": "flylike_ann", "sparse_ann": "flylike_ann",
    # snn
    "snn": "snn_transferred", "lossless_snn": "snn_transferred",
    "losslessconnectomesnn": "snn_transferred", "snn_transfer": "snn_transferred",
    "spiking": "snn_transferred", "spiking_ann": "snn_transferred",
    # dense distilled
    "dense": "dense_ann", "dense_ann": "dense_ann", "densenetcontroller": "dense_ann",
    "denseanncontroller": "dense_ann", "distilled": "dense_ann",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip, and map separators so aliases match forgivingly."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key, key)


class ControllerRegistry:
    """Builds controller instances from human-friendly names.

    Shared (possibly distilled) ``dense``/``connectome`` models are held here so
    that ``build()`` can hand out *fresh, independent* controllers while the
    expensive weights are loaded only once.
    """

    def __init__(
        self,
        network: Optional[NetworkConfig] = None,
        *,
        device: str = "cpu",
        seed: int = 42,
        dense: Optional[DenseNNController] = None,
        connectome: Optional[ConnectomeANNController] = None,
        micro_steps: int = 10,
        trained: bool = False,
    ) -> None:
        self.network = network or NetworkConfig()
        self.device = device
        self.seed = seed
        self.micro_steps = micro_steps
        self.trained = trained

        self.dense = dense or DenseNNController(
            n_in=self.network.n_in,
            n_neurons=self.network.n_neurons,
            n_out=self.network.n_out,
            device=device,
            seed=seed,
        )
        self.connectome = connectome or ConnectomeANNController(
            n_in=self.network.n_in,
            n_neurons=self.network.n_neurons,
            n_out=self.network.n_out,
            synapses_per_neuron=self.network.synapses_per_neuron,
            inhibitory_fraction=self.network.inhibitory_fraction,
            excitatory_weight=self.network.excitatory_weight,
            inhibitory_weight=self.network.inhibitory_weight,
            seed=seed,
            device=device,
        )
        self.teacher = ClassicalPDController(device=device)

    # -- introspection ------------------------------------------------------ #
    def names(self) -> List[str]:
        return list(CANONICAL_CONTROLLERS)

    def resolve(self, name: str) -> str:
        canonical = normalize_name(name)
        if canonical not in CANONICAL_CONTROLLERS and canonical not in EXTRA_CONTROLLERS:
            raise KeyError(
                f"unknown controller {name!r}; available: "
                f"{', '.join(CANONICAL_CONTROLLERS + EXTRA_CONTROLLERS)}"
            )
        return canonical

    def describe_all(self) -> List[dict]:
        out = []
        for name in CANONICAL_CONTROLLERS:
            info = {
                "name": name,  # registry name is authoritative
                "label": LABELS.get(name, name),
                "trained": bool(self.trained),
            }
            try:
                ctrl = self.build(name)
                desc = ctrl.describe()
                desc.pop("name", None)  # never let the class name shadow the alias
                info["class"] = type(ctrl).__name__
                info.update(desc)
            except Exception as exc:  # pragma: no cover - defensive
                info["error"] = str(exc)
            out.append(info)
        return out

    # -- construction ------------------------------------------------------- #
    def build(self, name: str) -> BaseController:
        """Build a *new* controller instance for ``name`` (with fresh state)."""
        canonical = self.resolve(name)

        if canonical == "pid":
            return ClassicalPDController(device=self.device)

        if canonical == "pid_no_ff":
            return ClassicalPDController(use_feedforward=False, device=self.device)

        if canonical == "random_ann":
            # Deterministic but distinct from the distilled model: offset seed.
            return DenseNNController(
                n_in=self.network.n_in,
                n_neurons=self.network.n_neurons,
                n_out=self.network.n_out,
                device=self.device,
                seed=self.seed + 1000,
            )

        if canonical == "dense_ann":
            if self.trained:
                import copy

                ctrl = copy.deepcopy(self.dense)
                ctrl.to(self.device)
                ctrl.eval()
                return ctrl
            return DenseNNController(
                n_in=self.network.n_in,
                n_neurons=self.network.n_neurons,
                n_out=self.network.n_out,
                device=self.device,
                seed=self.seed,
            )

        if canonical == "flylike_ann":
            return self.connectome  # already a fresh, state-free-at-act-time model

        if canonical == "snn_transferred":
            return LosslessConnectomeSNN(
                self.connectome,
                micro_steps=self.micro_steps,
                device=self.device,
            )

        raise KeyError(canonical)  # pragma: no cover - guarded by resolve()
