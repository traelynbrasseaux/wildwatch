"""FastAPI route tests.

We inject a tiny synthetic model into the inference service so tests don't
need real training artifacts, MLflow, or network access.
"""

from __future__ import annotations

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from src.serving import inference as inference_module
from src.serving.app import app
from src.serving.inference import ModelBundle, _build_transform


class _StubModel(nn.Module):
    """Deterministic classifier: returns a fixed logit pattern per batch item."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 224 * 224, num_classes)
        # Deterministic weights so the same image always produces the same logits.
        torch.manual_seed(0)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn_like(self.linear.weight) * 0.01)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


@pytest.fixture
def client():
    """Yield a TestClient with a loaded stub model; restore state afterwards."""
    stub = _StubModel(num_classes=3)
    stub.eval()
    bundle = ModelBundle(
        model=stub,
        class_names=["deer", "fox", "raccoon"],
        version="test-stub",
        source="local",
        image_size=224,
    )

    # Save then replace the service's internal state.
    saved_bundle = inference_module.service._bundle
    saved_transform = inference_module.service._transform
    inference_module.service._bundle = bundle
    inference_module.service._transform = _build_transform(224)

    # TestClient triggers the lifespan, which would overwrite our stub — so
    # we also short-circuit the service.load() call used on startup.
    original_load = inference_module.service.load
    inference_module.service.load = lambda params=None: bundle  # type: ignore[assignment]

    try:
        with TestClient(app) as c:
            # After entering the context, lifespan has run and our stub is in
            # place (the lambda above ignored the real load path).
            inference_module.service._bundle = bundle
            inference_module.service._transform = _build_transform(224)
            yield c
    finally:
        inference_module.service.load = original_load  # type: ignore[assignment]
        inference_module.service._bundle = saved_bundle
        inference_module.service._transform = saved_transform


def _make_jpeg_bytes(size: tuple[int, int] = (256, 256)) -> bytes:
    img = Image.new("RGB", size, color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200_and_model_metadata(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-stub"
    assert body["model_source"] == "local"
    assert body["classes"] == ["deer", "fox", "raccoon"]
    assert body["uptime_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_valid_image_returns_expected_schema(client: TestClient):
    jpeg = _make_jpeg_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", jpeg, "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()

    # Required fields
    assert isinstance(body["predicted_class"], str)
    assert body["predicted_class"] in {"deer", "fox", "raccoon"}
    assert 0.0 <= body["confidence"] <= 1.0

    # Probabilities cover every class and sum ~1.
    probs = body["class_probabilities"]
    assert set(probs.keys()) == {"deer", "fox", "raccoon"}
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4

    # Bookkeeping fields present.
    assert body["model_version"] == "test-stub"
    assert body["model_source"] == "local"
    assert "timestamp" in body
    assert body["latency_ms"] >= 0.0


def test_predict_with_no_file_returns_422(client: TestClient):
    resp = client.post("/predict")  # missing multipart
    assert resp.status_code == 422


def test_predict_with_corrupt_image_returns_422(client: TestClient):
    resp = client.post(
        "/predict",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "Could not decode" in resp.json()["detail"]


def test_predict_with_empty_file_returns_422(client: TestClient):
    resp = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_tracks_predictions(client: TestClient):
    # Pull baseline after the health check the client may have done on startup.
    baseline = client.get("/metrics").json()
    jpeg = _make_jpeg_bytes()

    for _ in range(3):
        resp = client.post(
            "/predict",
            files={"file": ("x.jpg", jpeg, "image/jpeg")},
        )
        assert resp.status_code == 200

    after = client.get("/metrics").json()
    assert after["prediction_count"] - baseline["prediction_count"] == 3
    assert after["request_count"] > baseline["request_count"]
    assert after["average_latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Degraded state (model not loaded)
# ---------------------------------------------------------------------------


def test_health_and_predict_when_model_unavailable():
    """When no model is loaded, /health is 'degraded' and /predict is 503."""
    saved_bundle = inference_module.service._bundle
    inference_module.service._bundle = ModelBundle(model=None, source="unavailable")
    inference_module.service._transform = None

    original_load = inference_module.service.load
    inference_module.service.load = lambda params=None: inference_module.service._bundle  # type: ignore[assignment]

    try:
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "degraded"
            assert health.json()["model_loaded"] is False

            predict = client.post(
                "/predict",
                files={"file": ("x.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
            assert predict.status_code == 503
    finally:
        inference_module.service.load = original_load  # type: ignore[assignment]
        inference_module.service._bundle = saved_bundle
