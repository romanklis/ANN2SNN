"""Engine runtime: per-(seed, profile, embodiment) services, weights caching and
behavioural-distillation jobs for the dashboard API."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

from sim_engine.api import EngineService
from sim_engine.config import EmbodimentConfig, NetworkConfig, TrainingConfig

from server.validation import BadRequest

log = logging.getLogger("ann2snn.server")

#: fixed number of SNN micro-steps per control frame (from NetworkConfig)
DEFAULT_MICRO_STEPS: int = NetworkConfig().micro_steps

#: Where distilled weights are cached between runs.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH: str = os.environ.get(
    "ANN2SNN_WEIGHTS", os.path.join(_REPO_ROOT, "weights.pt")
).strip()


class Runtime:
    """Owns the engine instances, the weights cache and the training job."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._train_lock = threading.Lock()
        self.default_seed = int(os.environ.get("ANN2SNN_SEED", "42"))
        self.n_neurons = int(os.environ.get("ANN2SNN_N_NEURONS", str(NetworkConfig().n_neurons)))
        self.micro_steps = DEFAULT_MICRO_STEPS
        self.weights_path: str = WEIGHTS_PATH
        self.robust_epochs = int(os.environ.get("ANN2SNN_ROBUST_EPOCHS", "60"))
        # Distil on first use when no cached bundle exists, so the learned brains
        # actually balance out of the box. Set ANN2SNN_AUTO_TRAIN=0 to disable.
        self.auto_train = str(os.environ.get("ANN2SNN_AUTO_TRAIN", "1")).strip().lower() \
            not in {"0", "false", "no", "off"}
        self._engines: Dict[str, EngineService] = {}
        self._weight_overrides: Dict[str, str] = {}
        self.job: Optional[dict] = None

    # -- engine construction ------------------------------------------------ #
    def _weights_path(self, seed: int, profile: str = "clean") -> str:
        """Per-(profile, seed) weights bundle path."""
        override = self._weight_overrides.get(profile)
        base = override or self.weights_path
        if not base:
            return ""
        if profile != "clean" and override is None:
            root, ext = os.path.splitext(base)
            base = f"{root}_{profile}{ext}"
        if seed == self.default_seed:
            return base
        root, ext = os.path.splitext(base)
        return f"{root}_seed{seed}{ext}"

    @staticmethod
    def _embodiment_key(embodiment) -> str:
        if embodiment is None or embodiment.is_clean:
            return "clean"
        return json.dumps(embodiment.to_dict(), sort_keys=True)

    def _config(self, seed: int, profile: str = "clean", embodiment=None) -> dict:
        cfg: Dict[str, Any] = {
            "seed": seed,
            "device": "cpu",
            "n_neurons": self.n_neurons,
            "micro_steps": self.micro_steps,
            "train_on_init": False,
        }
        if profile and profile != "clean":
            cfg["training"] = {
                "profile": profile,
                "epochs": self.robust_epochs,
                "episodes": 3,
                "episode_steps": 200,
                "noise_augment": 0.002,
            }
        weights = self._weights_path(seed, profile)
        if weights and os.path.isfile(weights):
            cfg["weights_path"] = weights
        elif self.auto_train:
            cfg["train_on_init"] = True
        if embodiment is not None and not embodiment.is_clean:
            cfg["embodiment"] = embodiment.to_dict()
        return cfg

    def _build(self, seed: int, profile: str = "clean", embodiment=None) -> EngineService:
        weights = self._weights_path(seed, profile)
        svc = EngineService(self._config(seed, profile, embodiment), train=None)
        # Persist what we just distilled so later workers/restarts skip training.
        try:
            if (weights and not os.path.isfile(weights)
                    and svc.engine.registry.trained):
                os.makedirs(os.path.dirname(weights) or ".", exist_ok=True)
                svc.engine.save_weights(weights)
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            log.warning("could not cache distilled weights: %s", exc)
        return svc

    def service(self, seed: Optional[int] = None, profile: str = "clean", embodiment=None) -> EngineService:
        """Return (and cache) the engine for *(seed, profile, embodiment)*."""
        seed = self.default_seed if seed is None else int(seed)
        profile = profile or "clean"
        key = f"{seed}|{profile}|{self._embodiment_key(embodiment)}"
        with self._lock:
            svc = self._engines.get(key)
            if svc is None:
                svc = self._build(seed, profile, embodiment)
                self._engines[key] = svc
            return svc

    def default_service(self) -> EngineService:
        return self.service(self.default_seed)

    @property
    def trained(self) -> bool:
        try:
            return bool(self.default_service().engine.registry.trained)
        except Exception:  # pragma: no cover - defensive
            return False

    def set_weights(self, path: Optional[str], profile: str = "clean") -> None:
        with self._lock:
            if path:
                self._weight_overrides[profile] = path
            else:
                self._weight_overrides.pop(profile, None)
            self._engines.clear()

    # -- reference helpers -------------------------------------------------- #
    def reference(self, engine, steps, radius, freq):
        return engine.reference(steps=steps, radius=radius, freq=freq)

    # -- training ----------------------------------------------------------- #
    def start_training(
        self, *, epochs: int, n_neurons: Optional[int], seed: int,
        profile: str = "clean", embodiment=None,
    ) -> dict:
        if not self._train_lock.acquire(blocking=False):
            raise BadRequest("a distillation job is already running")
        job_id = uuid.uuid4().hex
        self.job = {
            "id": job_id,
            "state": "queued",
            "epoch": 0,
            "epochs": int(epochs),
            "profile": profile,
            "stage": "queued",
            "loss": None,
            "error": None,
            "started": time.time(),
            "finished": None,
        }
        thread = threading.Thread(
            target=self._train_worker,
            args=(job_id, int(epochs), n_neurons, int(seed), profile,
                  None if embodiment is None else embodiment.to_dict()),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "state": "queued", "epochs": int(epochs), "profile": profile}

    def _train_worker(self, job_id: str, epochs: int, n_neurons: Optional[int],
                      seed: int, profile: str, embodiment_dict: Optional[dict]) -> None:
        from sim_engine import training as training_mod

        job = self.job
        try:
            network = NetworkConfig(n_neurons=int(n_neurons) if n_neurons else self.n_neurons)
            embodiment = EmbodimentConfig(**embodiment_dict) if embodiment_dict else None
            training = TrainingConfig(
                epochs=epochs, seed=seed, device="cpu",
                profile=profile, embodiment=embodiment,
            )

            def log_fn(which: str, epoch: int, loss: float) -> None:
                if self.job is not None and self.job["id"] == job_id:
                    self.job["stage"] = which
                    self.job["epoch"] = int(epoch)
                    self.job["loss"] = float(loss)

            if job is not None and job["id"] == job_id:
                job["state"] = "running"
                job["stage"] = "data" if profile == "robust" else "dense"
            distilled = training_mod.distill_profile(profile, network, training, log_fn=log_fn)

            path = self._weights_path(seed, profile) or WEIGHTS_PATH
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            training_meta = {
                "history": distilled["history"],
                "final_loss": distilled["final_loss"],
                "config": distilled["config"],
            }
            training_mod.save_weights(
                path,
                dense=distilled["dense"],
                connectome=distilled["connectome"],
                training=training_meta,
            )
            self.set_weights(path, profile=profile)

            if self.job is not None and self.job["id"] == job_id:
                self.job["state"] = "done"
                self.job["stage"] = "done"
                self.job["loss"] = distilled["final_loss"]
                self.job["weights_path"] = path
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            if self.job is not None and self.job["id"] == job_id:
                self.job["state"] = "error"
                self.job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if self.job is not None and self.job["id"] == job_id:
                self.job["finished"] = time.time()
            self._train_lock.release()

    def job_status(self, job_id: str) -> Optional[dict]:
        job = self.job
        if job is None or job["id"] != job_id:
            return None
        return dict(job)


RUNTIME = Runtime()
