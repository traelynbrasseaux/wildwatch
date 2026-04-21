"""Model loading + image preprocessing + prediction.

Contains :class:`InferenceService`, a process-global singleton that is loaded
once on FastAPI startup and reused across requests. Two load paths:

1. **MLflow registry** — the Production version of ``registry.model_name``.
2. **Local checkpoint** — ``models/model.pt`` from the training stage, used
   as a fallback when the registry is unreachable.

Image preprocessing matches the eval transform used in training so the
distribution shown to the served model is the same one it saw during
validation.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from mlflow.tracking import MlflowClient
from PIL import Image
from torch import nn
from torchvision import transforms

from src.training.model import build_model

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ModelSource = Literal["mlflow", "local", "unavailable"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    latency_ms: float


@dataclass
class ModelBundle:
    model: nn.Module | None
    class_names: list[str] = field(default_factory=list)
    version: str | None = None
    source: ModelSource = "unavailable"
    image_size: int = 224


# ---------------------------------------------------------------------------
# Params helper
# ---------------------------------------------------------------------------


def _load_params(path: Path | str = "params.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Load strategies
# ---------------------------------------------------------------------------


def _load_from_mlflow(
    tracking_uri: str,
    model_name: str,
) -> ModelBundle | None:
    """Try to load the current Production version. Returns None on failure."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except mlflow.exceptions.MlflowException as exc:
        logger.warning("MLflow registry lookup failed: %s", exc)
        return None

    if not versions:
        logger.info("No Production version registered for '%s'.", model_name)
        return None

    version = versions[0]
    try:
        model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
    except Exception as exc:  # mlflow raises many types here
        logger.warning("Failed to load model from MLflow: %s", exc)
        return None

    class_tag = client.get_run(version.run_id).data.tags.get("class_names", "")
    class_names = class_tag.split(",") if class_tag else []

    model.eval()
    logger.info(
        "Loaded model '%s' v%s from MLflow registry (classes=%d).",
        model_name,
        version.version,
        len(class_names),
    )
    return ModelBundle(
        model=model,
        class_names=class_names,
        version=str(version.version),
        source="mlflow",
    )


def _load_from_local_checkpoint(
    checkpoint_path: Path,
    backbone: str,
) -> ModelBundle | None:
    """Rebuild the model from a local training checkpoint."""
    if not checkpoint_path.exists():
        logger.info("Local checkpoint not found at %s", checkpoint_path)
        return None

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Failed to read checkpoint %s: %s", checkpoint_path, exc)
        return None

    class_names = list(ckpt.get("class_names", []))
    if not class_names:
        logger.warning("Checkpoint has no class_names; refusing to serve.")
        return None

    # Build architecture, then load weights. pretrained=False keeps the load
    # offline so the container starts without network access.
    try:
        model = build_model(
            backbone=backbone,
            num_classes=len(class_names),
            freeze_backbone=False,
            pretrained=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as exc:
        logger.warning("Failed to rebuild model from checkpoint: %s", exc)
        return None

    model.eval()
    logger.info(
        "Loaded model from local checkpoint %s (classes=%d).",
        checkpoint_path,
        len(class_names),
    )
    return ModelBundle(
        model=model,
        class_names=class_names,
        version="local",
        source="local",
    )


# ---------------------------------------------------------------------------
# Inference service (singleton)
# ---------------------------------------------------------------------------


class InferenceService:
    """Thread-safe cache around a single :class:`ModelBundle`.

    We call :meth:`load` once on app startup; subsequent :meth:`predict` calls
    reuse the same model. A re-entrant lock protects the transform state so
    concurrent requests don't trip over each other during the rare reload.
    """

    def __init__(self) -> None:
        self._bundle = ModelBundle(model=None)
        self._lock = threading.RLock()
        self._transform: transforms.Compose | None = None

    # -- load ---------------------------------------------------------------

    def load(self, params: dict | None = None) -> ModelBundle:
        """Load from MLflow first, fall back to the local checkpoint."""
        params = params or _load_params()
        serving = params.get("serving", {})
        paths = params.get("paths", {})
        training = params.get("training", {})
        registry = params.get("registry", {})

        image_size = int(params.get("preprocess", {}).get("image_size", 224))
        # Env override lets docker-compose / Cloud Run point at a remote
        # tracking server without editing params.yaml.
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            training.get("mlflow_tracking_uri", "file:./mlruns"),
        )
        model_name = registry.get("model_name", "wildwatch-classifier")
        backbone = training.get("backbone", "efficientnet_b0")
        checkpoint_path = (
            Path(paths.get("models_dir", "models"))
            / training.get("checkpoint_name", "model.pt")
        )

        bundle: ModelBundle | None = _load_from_mlflow(tracking_uri, model_name)

        if bundle is None and bool(serving.get("local_checkpoint_fallback", True)):
            bundle = _load_from_local_checkpoint(checkpoint_path, backbone)

        if bundle is None:
            logger.error(
                "No model could be loaded — serving will return 503 on /predict."
            )
            bundle = ModelBundle(model=None, source="unavailable")

        bundle.image_size = image_size

        with self._lock:
            self._bundle = bundle
            self._transform = _build_transform(image_size)

        return bundle

    # -- accessors ----------------------------------------------------------

    @property
    def bundle(self) -> ModelBundle:
        return self._bundle

    @property
    def is_ready(self) -> bool:
        return self._bundle.model is not None

    # -- inference ----------------------------------------------------------

    def predict(self, image_bytes: bytes) -> InferenceResult:
        """Run inference on a raw image payload."""
        if not self.is_ready or self._transform is None:
            raise RuntimeError("No model is loaded.")

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                rgb = img.convert("RGB")
                tensor = self._transform(rgb).unsqueeze(0)
        except Exception as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc

        model = self._bundle.model
        class_names = self._bundle.class_names
        assert model is not None  # guarded by is_ready

        start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        latency_ms = (time.perf_counter() - start) * 1000.0

        idx = int(np.argmax(probs))
        predicted_class = class_names[idx] if idx < len(class_names) else str(idx)
        class_probs = {
            name: float(probs[i]) for i, name in enumerate(class_names)
        }

        return InferenceResult(
            predicted_class=predicted_class,
            confidence=float(probs[idx]),
            class_probabilities=class_probs,
            latency_ms=latency_ms,
        )


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# Module-level singleton used by app.py. Tests may replace this via
# dependency injection (see tests/test_serving.py).
service = InferenceService()
