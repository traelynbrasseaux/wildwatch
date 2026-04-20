"""Metrics computation and confusion-matrix plotting.

Implemented with numpy directly (no ``scikit-learn`` dependency) so the
training container stays light. ``matplotlib`` is imported lazily inside the
plotting function so tests that only exercise numeric metrics don't pay for
it. All metrics are returned as a flat ``dict`` suitable for MLflow logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute an integer ``(num_classes, num_classes)`` confusion matrix.

    Rows are true labels, columns are predicted labels.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}"
        )
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


MetricsValue = float | dict[str, dict[str, float | int]]
MetricsDict = dict[str, MetricsValue]


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> MetricsDict:
    """Return accuracy + macro/per-class precision, recall, F1.

    Per-class metrics are keyed by class name under ``"per_class"``.
    """
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, num_classes)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denom,
        out=np.zeros_like(precision),
        where=f1_denom > 0,
    )

    accuracy = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(cm[i].sum()),
        }
        for i, name in enumerate(class_names)
    }

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": per_class,
    }


def flatten_metrics_for_mlflow(
    metrics: MetricsDict,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten the nested metrics dict into ``dict[str, float]`` for MLflow.

    Keys are sanitised: spaces and slashes become underscores. ``support``
    values are excluded (MLflow logs them anyway as metrics if needed).
    """
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                # per_class is the only nested block and is itself two-deep.
                if isinstance(sub_val, dict):
                    for metric_name, metric_val in sub_val.items():
                        flat[_sanitise(f"{prefix}{key}/{sub_key}/{metric_name}")] = float(
                            metric_val
                        )
                else:
                    flat[_sanitise(f"{prefix}{key}/{sub_key}")] = float(sub_val)
        else:
            flat[_sanitise(f"{prefix}{key}")] = float(value)
    return flat


def _sanitise(key: str) -> str:
    return key.replace(" ", "_").replace("/", "_")


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Render ``cm`` as a heatmap and save to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.7),) * 2)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[MetricsDict, np.ndarray]:
    """Run ``model`` over ``loader`` and return (metrics, confusion_matrix).

    The model is placed in eval mode; caller is responsible for restoring
    train mode afterwards if needed.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        images, targets = batch
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)

    metrics = classification_metrics(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    return metrics, cm
