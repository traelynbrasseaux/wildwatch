"""Tests for the training package.

These tests deliberately avoid hitting the real dataset or MLflow so they can
run in CI without large downloads. ``timm`` model creation is offline-friendly
when ``pretrained=False``; we rely on that for the forward-pass smoke test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.training.config import TrainingConfig, load_training_config
from src.training.evaluate import (
    classification_metrics,
    confusion_matrix,
    flatten_metrics_for_mlflow,
)
from src.training.model import build_model, trainable_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_load_training_config_from_params_yaml():
    cfg = load_training_config(REPO_ROOT / "params.yaml")
    assert cfg.training.backbone == "efficientnet_b0"
    assert cfg.training.batch_size > 0
    assert cfg.training.epochs > 0
    assert 0 < cfg.training.learning_rate < 1
    # seed falls back correctly
    assert cfg.seed == cfg.training.seed


def test_training_config_rejects_bad_values():
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=-0.1)
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)


def test_training_config_resume_from_blank_becomes_none():
    cfg = TrainingConfig(resume_from="")
    assert cfg.resume_from is None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_classes", [2, 5, 8])
def test_model_forward_pass_shape(num_classes):
    model = build_model(
        backbone="efficientnet_b0",
        num_classes=num_classes,
        freeze_backbone=True,
        pretrained=False,
    )
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, num_classes)


def test_freeze_backbone_only_head_trains():
    model = build_model(
        backbone="efficientnet_b0",
        num_classes=5,
        freeze_backbone=True,
        pretrained=False,
    )
    trainable = trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable)
    # Head should be a small fraction of total params.
    assert 0 < trainable_count < total_params * 0.1


def test_build_model_rejects_too_few_classes():
    with pytest.raises(ValueError):
        build_model("efficientnet_b0", num_classes=1, pretrained=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_confusion_matrix_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    cm = confusion_matrix(y, y, num_classes=3)
    # Diagonal is the counts per class, off-diagonal all zero.
    assert np.array_equal(np.diag(cm), [2, 2, 2])
    assert cm.sum() == 6
    assert (cm - np.diag(np.diag(cm))).sum() == 0


def test_confusion_matrix_known_mistakes():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm[0, 0] == 1  # correct class-0
    assert cm[0, 1] == 1  # class-0 predicted as 1
    assert cm[1, 1] == 2  # both class-1s correct
    assert cm[2, 2] == 1
    assert cm[2, 0] == 1


def test_metrics_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    classes = ["a", "b", "c"]
    metrics = classification_metrics(y, y, classes)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    for name in classes:
        assert metrics["per_class"][name]["f1"] == pytest.approx(1.0)
        assert metrics["per_class"][name]["support"] == 2


def test_metrics_all_wrong():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    metrics = classification_metrics(y_true, y_pred, ["a", "b"])
    assert metrics["accuracy"] == pytest.approx(0.0)
    # Recall for class a is 0, precision for b is 0 → f1 macro stays 0.
    assert metrics["f1_macro"] == pytest.approx(0.0)


def test_flatten_metrics_for_mlflow_is_flat_floats():
    metrics = {
        "accuracy": 0.9,
        "f1_macro": 0.85,
        "per_class": {
            "cat": {"precision": 1.0, "recall": 0.8, "f1": 0.89, "support": 10},
        },
    }
    flat = flatten_metrics_for_mlflow(metrics)
    assert all(isinstance(v, float) for v in flat.values())
    assert flat["accuracy"] == pytest.approx(0.9)
    assert flat["per_class_cat_precision"] == pytest.approx(1.0)
    # No nested dicts survived.
    assert all("/" not in k for k in flat)
