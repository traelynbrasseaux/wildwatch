"""Pydantic request/response models for the serving API.

Schemas live in a dedicated module so they're easy to import from tests and
OpenAPI generation tooling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """One-shot classification result for a single image."""

    predicted_class: str = Field(..., description="Top-1 class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax score of the top class.")
    class_probabilities: dict[str, float] = Field(
        ..., description="Softmax score per known class (sums to ~1.0)."
    )
    model_version: str | None = Field(
        None, description="Registry version (or 'local' when served from the checkpoint)."
    )
    model_source: Literal["mlflow", "local", "unavailable"] = Field(
        ..., description="Which backend produced this prediction."
    )
    timestamp: datetime = Field(..., description="Server-side timestamp when inference ran.")
    latency_ms: float = Field(..., ge=0.0, description="Inference wall-clock latency in ms.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"] = Field(
        ..., description="'ok' when a model is loaded; 'degraded' if inference is unavailable."
    )
    model_loaded: bool
    model_version: str | None = None
    model_source: Literal["mlflow", "local", "unavailable"]
    uptime_seconds: float = Field(..., ge=0.0)
    classes: list[str] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    request_count: int = Field(..., ge=0)
    prediction_count: int = Field(..., ge=0)
    error_count: int = Field(..., ge=0)
    average_latency_ms: float = Field(..., ge=0.0)


class ErrorResponse(BaseModel):
    detail: str
