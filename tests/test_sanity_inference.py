"""Smoke test: the current Production model classifies real test images sanely.

Loads the latest ``Production`` version of the registered model from MLflow,
runs it over a few images from ``data/processed/test/<class>/*.jpg``, and
asserts that every prediction maps to one of the known class names.

We deliberately don't assert exact-class correctness — a tiny synthetic
dataset won't give meaningful per-image accuracy — but we do assert that the
model produces valid outputs on real inputs.

This test skips cleanly when:

* No registered Production model exists (e.g., first-time run of CI).
* The processed test split is not present (e.g., data/ was gitignored).
"""

from __future__ import annotations

import random
from pathlib import Path

import mlflow
import pytest
import torch
import yaml
from mlflow.tracking import MlflowClient
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_params() -> dict:
    with (REPO_ROOT / "params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _preprocess(img_path: Path, size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    with Image.open(img_path) as img:
        return transform(img.convert("RGB")).unsqueeze(0)


def _sample_test_images(test_dir: Path, n: int, seed: int = 0) -> list[Path]:
    rng = random.Random(seed)
    candidates = [p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(candidates) < n:
        return candidates
    return rng.sample(candidates, n)


@pytest.fixture(scope="module")
def production_model_and_classes():
    """Load the latest Production model + its known class list.

    Returns ``(model, class_names)`` or skips the test.
    """
    params = _load_params()
    tracking_uri = params.get("training", {}).get(
        "mlflow_tracking_uri", "file:./mlruns"
    )
    model_name = params.get("registry", {}).get("model_name", "wildwatch-classifier")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        pytest.skip(f"Registered model '{model_name}' does not exist yet.")

    if not versions:
        pytest.skip(f"No Production version for '{model_name}' yet.")

    version = versions[0]
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    run = client.get_run(version.run_id)
    class_names_tag = run.data.tags.get("class_names")
    if not class_names_tag:
        pytest.skip("Production run is missing the 'class_names' tag.")
    class_names = class_names_tag.split(",")

    return model, class_names


def test_sanity_inference_on_real_test_images(production_model_and_classes):
    model, class_names = production_model_and_classes

    test_dir = REPO_ROOT / "data" / "processed" / "test"
    if not test_dir.exists():
        pytest.skip(f"Processed test split not found at {test_dir}")

    images = _sample_test_images(test_dir, n=5)
    if not images:
        pytest.skip("No images available under data/processed/test/")

    image_size = int(_load_params()["preprocess"]["image_size"])

    predictions: list[str] = []
    for img_path in images:
        tensor = _preprocess(img_path, image_size)
        with torch.no_grad():
            logits = model(tensor)
        assert logits.shape == (1, len(class_names)), (
            f"Model output shape {logits.shape} doesn't match "
            f"{len(class_names)} known classes."
        )
        idx = int(logits.argmax(dim=1).item())
        assert 0 <= idx < len(class_names)
        predicted = class_names[idx]
        assert predicted in class_names
        predictions.append(predicted)

    # All predictions must be valid class labels — we don't assert correctness
    # because the tiny synthetic dataset doesn't support it.
    assert len(predictions) == len(images)
    assert all(p in class_names for p in predictions)
