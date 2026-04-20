"""Model factory for WildWatch classification.

Wraps ``timm`` so we get a pretrained backbone with a fresh classifier head
for ``num_classes``. Supports freezing the backbone for fast fine-tuning on
small datasets (the default regime for this project).
"""

from __future__ import annotations

import logging

import timm
import torch
from torch import nn

logger = logging.getLogger(__name__)


def build_model(
    backbone: str,
    num_classes: int,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ``timm`` model with a classifier head of size ``num_classes``.

    Passing ``num_classes`` to ``timm.create_model`` replaces the head with a
    correctly-shaped Linear layer and randomly initialises its weights.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if freeze_backbone:
        freeze_backbone_params(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built %s: classes=%d trainable=%d/%d (%.2f%%)",
        backbone,
        num_classes,
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    return model


def freeze_backbone_params(model: nn.Module) -> None:
    """Freeze every parameter that is not part of the classifier head.

    ``timm`` exposes the classifier via :func:`get_classifier`; we unfreeze only
    that module so the backbone stays in eval-forward mode during fine-tuning.
    """
    for p in model.parameters():
        p.requires_grad = False

    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None:
        raise RuntimeError(
            "Model does not expose .get_classifier(); cannot freeze backbone."
        )
    for p in classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Re-enable gradients on every parameter (used for full fine-tuning)."""
    for p in model.parameters():
        p.requires_grad = True


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    """Return the parameter list to hand to the optimiser."""
    return [p for p in model.parameters() if p.requires_grad]
