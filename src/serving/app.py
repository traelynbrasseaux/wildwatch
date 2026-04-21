"""FastAPI app that exposes the WildWatch classifier.

Endpoints:

* ``GET  /health``   — liveness + loaded model version.
* ``POST /predict``  — multipart image upload, returns top class + all probs.
* ``GET  /metrics``  — simple in-memory counters (request count, avg latency).

The inference service is loaded once on startup via the ASGI lifespan and
reused across requests. If the load fails the app still starts but /predict
returns 503 — this keeps ``/health`` reachable so orchestrators can see the
degraded state.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.serving.inference import service
from src.serving.schemas import HealthResponse, MetricsResponse, PredictionResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class _Metrics:
    """In-memory counters. Replace with Prometheus in production."""

    def __init__(self) -> None:
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0

    @property
    def average_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.request_count, 1)


metrics = _Metrics()
_app_start_time: float = time.monotonic()


def _load_params() -> dict:
    path = Path("params.yaml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Lifespan: load model on startup.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Booting WildWatch serving app — loading model...")
    try:
        service.load()
    except Exception as exc:
        # Never crash on boot; /health will report the failure.
        logger.exception("Model load failed at startup: %s", exc)
    yield


app = FastAPI(
    title="WildWatch Classifier",
    description="Wildlife camera-trap image classifier served via FastAPI.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


params = _load_params()
cors_origins = params.get("serving", {}).get("cors_origins", ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status, latency; also populate ``metrics``."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            metrics.error_count += 1
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics.request_count += 1
            metrics.total_latency_ms += elapsed_ms
            logger.info(
                '%s %s -> %s %.1fms',
                request.method,
                request.url.path,
                locals().get("status_code", "ERR"),
                elapsed_ms,
            )


app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = service.bundle
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        model_version=bundle.version,
        model_source=bundle.source,
        uptime_seconds=time.monotonic() - _app_start_time,
        classes=bundle.class_names,
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    return MetricsResponse(
        request_count=metrics.request_count,
        prediction_count=metrics.prediction_count,
        error_count=metrics.error_count,
        average_latency_ms=metrics.average_latency_ms,
    )


_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: Annotated[UploadFile, File(...)]) -> PredictionResponse:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model is loaded.",
        )

    # content_type check is advisory — PIL will give the real answer.
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        logger.warning("Unusual content type: %s", file.content_type)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Empty file.",
        )

    try:
        result = service.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    metrics.prediction_count += 1
    bundle = service.bundle
    return PredictionResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        class_probabilities=result.class_probabilities,
        model_version=bundle.version,
        model_source=bundle.source,
        timestamp=datetime.now(UTC),
        latency_ms=result.latency_ms,
    )
