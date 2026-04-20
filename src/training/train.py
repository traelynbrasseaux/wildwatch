"""Training entry-point: fine-tune the backbone and log to MLflow.

Reads config from ``params.yaml`` via :mod:`src.training.config`, builds the
data loaders over ``data/processed/{train,val}``, and runs a standard PyTorch
loop. Every run logs hyperparameters, per-epoch metrics, final evaluation
metrics, a confusion-matrix artifact, and the DVC data version tag to MLflow.

Outputs the best-validation-accuracy checkpoint to
``<models_dir>/<checkpoint_name>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
from pathlib import Path
from typing import cast

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.training.config import FullConfig, TrainingConfig, load_training_config
from src.training.evaluate import (
    evaluate_model,
    flatten_metrics_for_mlflow,
    save_confusion_matrix_plot,
)
from src.training.model import build_model, trainable_parameters

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _build_transforms(
    image_size: int,
    train_cfg: TrainingConfig,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_tf, eval_tf). Train applies augmentation; eval does not."""
    aug = train_cfg.augmentation

    train_ops: list = []
    if aug.horizontal_flip:
        train_ops.append(transforms.RandomHorizontalFlip())
    if aug.rotation_degrees > 0:
        train_ops.append(transforms.RandomRotation(degrees=aug.rotation_degrees))
    if aug.color_jitter > 0:
        train_ops.append(
            transforms.ColorJitter(
                brightness=aug.color_jitter,
                contrast=aug.color_jitter,
                saturation=aug.color_jitter,
            )
        )
    train_ops.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def _build_loaders(
    cfg: FullConfig,
    image_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test loaders from ``data/processed`` and return class names."""
    train_tf, eval_tf = _build_transforms(image_size, cfg.training)
    processed = cfg.paths.processed_dir

    train_ds = ImageFolder(str(processed / "train"), transform=train_tf)
    val_ds = ImageFolder(str(processed / "val"), transform=eval_tf)
    test_ds = ImageFolder(str(processed / "test"), transform=eval_tf)

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise RuntimeError(
            "Class lists differ across splits — re-run preprocess/validate."
        )

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader, test_loader, train_ds.classes


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch; return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += images.size(0)

    avg_loss = total_loss / max(total_seen, 1)
    accuracy = total_correct / max(total_seen, 1)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpointing & DVC tag
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    class_names: list[str],
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "config": config.model_dump(),
        },
        path,
    )


def _try_resume(
    resume_path: Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    """Load state_dicts from ``resume_path`` if present. Returns (start_epoch, best_val_acc)."""
    if resume_path is None or not resume_path.exists():
        return 0, 0.0
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best = float(ckpt.get("best_val_acc", 0.0))
    logger.info("Resumed from %s at epoch %d (best_val_acc=%.4f)", resume_path, start_epoch, best)
    return start_epoch, best


def _get_dvc_data_version() -> str | None:
    """Return a short hash identifying the current data version.

    Prefers the per-output hash recorded in ``dvc.lock`` for the ``preprocess``
    stage (this is what actually pins ``data/processed``). Falls back to
    git-hashing ``dvc.lock`` itself, then to ``None``.
    """
    lock_path = Path("dvc.lock")
    if lock_path.exists():
        try:
            import yaml

            lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
            outs = lock.get("stages", {}).get("preprocess", {}).get("outs", [])
            for out in outs:
                if out.get("path") == "data/processed" and out.get("md5"):
                    return str(out["md5"])[:12]
        except (OSError, yaml.YAMLError):
            pass

        try:
            result = subprocess.run(
                ["git", "hash-object", str(lock_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()[:12]
        except (FileNotFoundError, OSError):
            pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train(cfg: FullConfig) -> dict:
    """Run the full training loop and return a summary dict.

    Summary includes the MLflow run ID, final test metrics, and the checkpoint
    path. Written to ``<models_dir>/train_summary.json`` for the DVC stage.
    """
    _seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)

    # Loaders
    image_size = 224  # inference expects the same size as preprocess
    train_loader, val_loader, test_loader, class_names = _build_loaders(cfg, image_size)
    num_classes = len(class_names)
    logger.info("Loaded %d classes: %s", num_classes, class_names)

    # Model + optimiser
    model = build_model(
        backbone=cfg.training.backbone,
        num_classes=num_classes,
        freeze_backbone=cfg.training.freeze_backbone,
    ).to(device)

    optimizer = torch.optim.Adam(
        trainable_parameters(model),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Optional resume
    resume_path = (
        Path(cfg.training.resume_from) if cfg.training.resume_from else None
    )
    start_epoch, best_val_acc = _try_resume(resume_path, model, optimizer)

    # MLflow
    mlflow.set_tracking_uri(cfg.training.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.training.mlflow_experiment)

    checkpoint_path = cfg.paths.models_dir / cfg.training.checkpoint_name
    data_version = _get_dvc_data_version()

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "backbone": cfg.training.backbone,
                "learning_rate": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
                "batch_size": cfg.training.batch_size,
                "epochs": cfg.training.epochs,
                "freeze_backbone": cfg.training.freeze_backbone,
                "num_classes": num_classes,
                "seed": cfg.training.seed,
                "horizontal_flip": cfg.training.augmentation.horizontal_flip,
                "rotation_degrees": cfg.training.augmentation.rotation_degrees,
                "color_jitter": cfg.training.augmentation.color_jitter,
            }
        )
        mlflow.set_tag("class_names", ",".join(class_names))
        if data_version:
            mlflow.set_tag("dvc_data_version", data_version)

        for epoch in range(start_epoch, cfg.training.epochs):
            train_loss, train_acc = _train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_metrics, _ = evaluate_model(model, val_loader, device, class_names)
            val_acc = cast(float, val_metrics["accuracy"])

            logger.info(
                "epoch=%d train_loss=%.4f train_acc=%.4f val_acc=%.4f",
                epoch,
                train_loss,
                train_acc,
                val_acc,
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_f1_macro": cast(float, val_metrics["f1_macro"]),
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    best_val_acc,
                    class_names,
                    cfg.training,
                )
                logger.info("New best val_acc=%.4f; checkpoint saved", val_acc)

        # Final test-set evaluation using the best checkpoint.
        if checkpoint_path.exists():
            best_state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(best_state["model_state_dict"])

        test_metrics, cm = evaluate_model(model, test_loader, device, class_names)
        logger.info(
            "TEST  accuracy=%.4f f1_macro=%.4f",
            test_metrics["accuracy"],
            test_metrics["f1_macro"],
        )

        # Log flat test metrics (MLflow rejects nested dicts).
        mlflow.log_metrics(flatten_metrics_for_mlflow(test_metrics, prefix="test/"))

        # Confusion matrix artifact.
        cm_path = cfg.paths.reports_dir / "confusion_matrix.png"
        save_confusion_matrix_plot(cm, class_names, cm_path, title="Test set")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Full metrics blob (including per-class + support) as JSON.
        metrics_path = cfg.paths.reports_dir / "test_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, sort_keys=True)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        # Model artifact.
        mlflow.pytorch.log_model(model, artifact_path="model")

        summary = {
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment_id": run.info.experiment_id,
            "checkpoint_path": str(checkpoint_path),
            "best_val_acc": best_val_acc,
            "test_accuracy": cast(float, test_metrics["accuracy"]),
            "test_f1_macro": cast(float, test_metrics["f1_macro"]),
            "class_names": class_names,
            "dvc_data_version": data_version,
        }

    summary_path = cfg.paths.models_dir / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info("Wrote training summary to %s", summary_path)
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train the WildWatch classifier.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()

    cfg = load_training_config(args.params)
    train(cfg)


if __name__ == "__main__":
    main()
