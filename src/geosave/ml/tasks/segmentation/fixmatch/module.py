from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from geosave.ml.tasks.segmentation import supervised

from .data import TrainingBatch


def pseudo_label_loss(
    logits: torch.Tensor,
    pseudo: torch.Tensor,
    confident: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Score logits against confident pseudo-labels, averaged over valid pixels.

    Unconfident pixels count in the average with zero loss, so the weight of
    the unlabeled term grows as the model grows confident.

    Args:
        logits: ``(B, classes, y, x)`` logits of a strong view.
        pseudo: ``(B, y, x)`` pseudo-labels.
        confident: ``(B, y, x)`` pixels whose pseudo-label is kept.
        valid: ``(B, y, x)`` pixels with real inputs.

    Returns:
        Scalar loss.
    """
    loss = F.cross_entropy(logits, pseudo, reduction="none")
    return (loss * confident).sum() / valid.sum().clamp(min=1)


class Module(supervised.Module):
    """Train segmentation with FixMatch: weak views label strong views.

    Confident weak-view classes supervise the strong view. Validation and
    testing are those of supervised training.

    Args:
        model_chain: Ordered stage construction specifications; the last
            stage states `num_classes`.
        ignore_index: Class index excluded from loss and metrics.
        criterion: Labeled loss class and arguments. Defaults to CrossEntropyLoss.
        optimizer: Optimizer class, arguments, and groups by stage.
        lr_scheduler: Scheduler class, arguments, and Lightning metadata.
        metrics: Metric names in dot notation (e.g. ``["iou.macro"]``).
        class_thresholds: Per-class confidence threshold, one per class.
        confidence: Weak-view probability a pseudo-label needs to be kept.
        unlabeled_weight: Weight of the unlabeled loss against the labeled one.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave.ml.tasks.segmentation.fixmatch.Module
          init_args:
            confidence: 0.95
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              decoder: {name: dpt}
              head: {name: segmentation, init_args: {classes: [background, palm]}}
    """

    def __init__(
        self,
        *,
        model_chain: dict[str, dict[str, Any]],
        ignore_index: int = 255,
        criterion: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        class_thresholds: list[float] | None = None,
        confidence: float = 0.95,
        unlabeled_weight: float = 1.0,
    ) -> None:
        super().__init__(
            model_chain=model_chain,
            ignore_index=ignore_index,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            class_thresholds=class_thresholds,
        )
        self.confidence = confidence
        self.unlabeled_weight = unlabeled_weight

    # Training reads labeled and unlabeled parts; evaluation keeps the supervised batch.
    def training_step(  # type: ignore[override]
        self, batch: TrainingBatch, batch_idx: int
    ) -> torch.Tensor:
        """Score labeled chips and train strong views on weak-view pseudo-labels.

        Args:
            batch: Labeled and unlabeled parts, as the FixMatch DataModule
                finishes them.
            batch_idx: Lightning batch number.

        Returns:
            Labeled loss plus the weighted unlabeled loss.
        """
        target = batch["labeled"]["target"]
        weak = batch["unlabeled"]["weak"]
        strong = batch["unlabeled"]["strong"]
        valid = batch["unlabeled"]["valid"]

        # Labeled loss.
        logits = self(**batch["labeled"]["inputs"])
        labeled_loss = self.criterion(logits, target)

        # Confident weak-view classes become the strong view's labels.
        with torch.no_grad():
            confidence, pseudo = self(**weak).softmax(dim=1).max(dim=1)
        confident = (confidence >= self.confidence) & valid
        unlabeled_loss = pseudo_label_loss(self(**strong), pseudo, confident, valid)

        loss = labeled_loss + self.unlabeled_weight * unlabeled_loss

        batch_size = target.shape[0]
        self.train_metrics.update(logits, target)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_unlabeled_loss",
            unlabeled_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_confident",
            confident.sum() / valid.sum().clamp(min=1),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log_dict(
            self.train_metrics, on_step=False, on_epoch=True, batch_size=batch_size
        )
        return loss
