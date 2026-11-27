from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn

from .module import Module as SupervisedModule

from .fixmatch_datamodule import TrainingBatch


def pseudo_label_loss(
    logits: torch.Tensor,
    pseudo: torch.Tensor,
    confident: torch.Tensor,
) -> torch.Tensor:
    """Score logits against confident pseudo-labels, averaged over all pixels.

    Unconfident pixels count in the average with zero loss, so the weight of
    the unlabeled term grows as the model grows confident.

    Args:
        logits: ``(B, classes, y, x)`` logits of a strong view.
        pseudo: ``(B, y, x)`` pseudo-labels.
        confident: ``(B, y, x)`` pixels whose pseudo-label is kept.

    Returns:
        Scalar loss.
    """
    loss = F.cross_entropy(logits, pseudo, reduction="none")
    return (loss * confident).mean()


class Module(SupervisedModule):
    """Train segmentation with FixMatch: weak views label strong views.

    Confident weak-view classes supervise the strong view. Validation and
    testing are those of supervised training.

    Args:
        model_chain: Ordered stage construction specifications; the last
            stage states `num_classes`.
        ignore_index: Class index excluded from loss and metrics.
        criterion: Labeled loss module. Defaults to CrossEntropyLoss ignoring
            `ignore_index`.
        optimizer: Optimizer class and arguments, built over the parameter
            groups. Defaults to AdamW with a learning rate of 1e-3.
        param_groups: Optimizer options per chain stage, such as an
            encoder's `lr`. None puts every parameter in one group.
        lr_scheduler: Scheduler class and arguments. None disables scheduling.
        lr_scheduler_config: Lightning's scheduler options, such as `interval`.
        metrics: Metric names, such as ``["iou_macro"]``.
        confidence: Weak-view probability a pseudo-label needs to be kept.
        unlabeled_weight: Weight of the unlabeled loss against the labeled one.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: training.fixmatch_module.Module
          init_args:
            confidence: 0.95
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              decoder: {name: dpt}
              head: {name: segmentation, init_args: {num_classes: 2}}
    """

    def __init__(
        self,
        *,
        model_chain: dict[str, dict[str, Any]],
        ignore_index: int = 255,
        criterion: nn.Module | None = None,
        optimizer: OptimizerCallable | None = None,
        param_groups: dict[str, dict[str, Any]] | None = None,
        lr_scheduler: LRSchedulerCallable | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        confidence: float = 0.95,
        unlabeled_weight: float = 1.0,
    ) -> None:
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if optimizer is None:
            optimizer = partial(torch.optim.AdamW, lr=1e-3)
        super().__init__(
            model_chain=model_chain,
            ignore_index=ignore_index,
            criterion=criterion,
            optimizer=optimizer,
            param_groups=param_groups,
            lr_scheduler=lr_scheduler,
            lr_scheduler_config=lr_scheduler_config,
            metrics=["iou_macro"] if metrics is None else metrics,
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

        # Labeled loss.
        labeled_loss = self.labeled_loss(batch["labeled"])

        # Confident weak-view classes become the strong view's labels.
        with torch.no_grad():
            confidence, pseudo = self(**weak).softmax(dim=1).max(dim=1)
        confident = confidence >= self.confidence
        unlabeled_loss = pseudo_label_loss(self(**strong), pseudo, confident)

        loss = labeled_loss + self.unlabeled_weight * unlabeled_loss

        batch_size = target.shape[0]
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
            confident.float().mean(),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log_dict(self.train_metrics, on_epoch=True, batch_size=batch_size)
        return loss
