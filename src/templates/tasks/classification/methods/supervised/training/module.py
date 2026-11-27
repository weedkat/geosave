from __future__ import annotations

from functools import partial
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from geosave.ml.torch.training.optim import configure_optimizers
from geosave.ml.torch.training.metrics import ClassificationMetrics
from geosave.ml.torch import ModelChain

from .datamodule import Sample


class Module(LightningModule):
    """Train and evaluate a classification model chain on chip batches.

    A batch is a collated `Sample` with one class code per chip. Every
    split scores chips one by one.

    Args:
        model_chain: Ordered stage construction specifications. The last
            stage states the class count as `num_classes`.
        criterion: Loss module. Defaults to CrossEntropyLoss.
        optimizer: Optimizer class and arguments, built over the parameter
            groups. Defaults to AdamW with a learning rate of 1e-3.
        param_groups: Optimizer options per chain stage, such as an
            encoder's `lr`. None puts every parameter in one group.
        lr_scheduler: Scheduler class and arguments. None disables scheduling.
        lr_scheduler_config: Lightning's scheduler options, such as `interval`.
        metrics: Metric names, such as ``["accuracy_macro"]``.

    Raises:
        TypeError: The last stage states no integer `num_classes`.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: training.Module
          init_args:
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              head: {name: classification, init_args: {num_classes: 3}}
            metrics: [accuracy_macro, f1_macro]
    """

    model: ModelChain
    num_classes: int

    def __init__(
        self,
        *,
        model_chain: dict[str, dict[str, Any]],
        criterion: nn.Module | None = None,
        optimizer: OptimizerCallable | None = None,
        param_groups: dict[str, dict[str, Any]] | None = None,
        lr_scheduler: LRSchedulerCallable | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        if not model_chain:
            raise ValueError("Supply at least one model stage")
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = partial(torch.optim.AdamW, lr=1e-3)
        self.save_hyperparameters()
        self.model_chain = model_chain
        self.optimizer = optimizer
        self.param_groups = param_groups
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_config = lr_scheduler_config
        self.metrics_config = metrics
        self.criterion = criterion

    def configure_model(self) -> None:
        """Construct the model, then the metrics its classes size.

        Raises:
            TypeError: The last stage states no integer `num_classes`.
        """
        if hasattr(self, "model"):
            return
        last = list(self.model_chain)[-1]
        self.model = ModelChain.from_config(self.model_chain)
        head = self.model.get_submodule(last)
        num_classes = getattr(head, "num_classes", None)
        if not isinstance(num_classes, int):
            raise TypeError(
                f"stage {last!r} ({type(head).__name__}) states no num_classes; "
                f"the last stage names how many classes it predicts"
            )
        self.num_classes = num_classes

        metrics = ClassificationMetrics(
            num_classes=num_classes, metrics=self.metrics_config
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build optimization for the configured chain's trainable parameters.

        Returns:
            The optimizer alone when scheduling is disabled, otherwise a
            Lightning dictionary with `optimizer` and `lr_scheduler` entries.
        """
        return configure_optimizers(
            self.model,
            self.optimizer,
            self.param_groups,
            self.lr_scheduler,
            self.lr_scheduler_config,
        )

    def forward(self, **model_inputs: Any) -> torch.Tensor:
        """Return raw ``(B, classes)`` logits for prepared model inputs.

        Args:
            **model_inputs: Named, batched tensors and context required by the
                configured chain. Pixel preparation is the caller's job.

        Returns:
            `(batch, num_classes)` raw class scores in the head's class order.

        Raises:
            TypeError: The model does not return one tensor.
        """
        logits = self.model(**model_inputs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Classification models must return logits as a tensor")
        return logits

    def _score(self, batch: Sample, split: str) -> torch.Tensor:
        """Score one batch, update the split's metrics, and log both."""
        target = batch["target"]
        logits = self(**batch["inputs"])
        loss = self.criterion(logits, target)
        metrics = getattr(self, f"{split}_metrics")
        metrics.update(logits, target)
        self.log(
            f"{split}_loss",
            loss,
            on_step=split == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(target),
        )
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=len(target))
        return loss

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        """Score one classification batch and log training loss and metrics.

        Args:
            batch: Collated `Sample` with named inputs and `(batch,)` int64
                class targets, already prepared by the DataModule.
            batch_idx: Lightning's batch position; unused by this task.

        Returns:
            Scalar criterion loss used by Lightning for backpropagation.
        """
        return self._score(batch, "train")

    def validation_step(self, batch: Sample, batch_idx: int) -> None:
        """Update and log classification validation loss and metrics.

        Args:
            batch: Prepared, collated `Sample` with one class target per chip.
            batch_idx: Lightning's batch position; unused by this task.
        """
        self._score(batch, "val")

    def test_step(self, batch: Sample, batch_idx: int) -> None:
        """Update and log classification test loss and metrics.

        Args:
            batch: Prepared, collated `Sample` with one class target per chip.
            batch_idx: Lightning's batch position; unused by this task.
        """
        self._score(batch, "test")
