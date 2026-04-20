"""Pydantic config models for the training stage.

Centralises training hyperparameters so the training loop, DVC stage, and
tests all read the same values. Defaults come from ``params.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class AugmentationConfig(BaseModel):
    horizontal_flip: bool = True
    rotation_degrees: float = Field(default=15.0, ge=0.0, le=180.0)
    color_jitter: float = Field(default=0.2, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    backbone: str = "efficientnet_b0"
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=10, gt=0)
    freeze_backbone: bool = True
    num_workers: int = Field(default=2, ge=0)
    augmentation: AugmentationConfig = AugmentationConfig()

    mlflow_experiment: str = "wildwatch-classifier"
    mlflow_tracking_uri: str = "file:./mlruns"
    checkpoint_name: str = "model.pt"
    resume_from: str | None = None

    seed: int = 42

    @field_validator("resume_from", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


class FullConfig(BaseModel):
    """Bundle of the sections the training stage needs."""

    seed: int
    paths: PathsConfig
    training: TrainingConfig


def load_params(params_path: Path | str = "params.yaml") -> dict:
    """Read and parse ``params.yaml`` into a plain dict."""
    with Path(params_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_training_config(params_path: Path | str = "params.yaml") -> FullConfig:
    """Load ``params.yaml`` and return a validated :class:`FullConfig`.

    ``training.seed`` falls back to the top-level ``seed`` for reproducibility.
    """
    params = load_params(params_path)

    training_raw = dict(params.get("training", {}))
    training_raw.setdefault("seed", int(params.get("seed", 42)))

    return FullConfig(
        seed=int(params.get("seed", 42)),
        paths=PathsConfig(**params.get("paths", {})),
        training=TrainingConfig(**training_raw),
    )
