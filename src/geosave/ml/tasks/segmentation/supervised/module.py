from copy import deepcopy
from typing import Any, cast

import torch
from torchmetrics import MetricCollection
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from geosave.ml.builders import build_criterion, build_optimizer, build_scheduler
from geosave.ml.metrics import MulticlassMetrics
from geosave.model.chain import ModelChain
from geosave.model.registry import build_model

from .data import Sample


class Module(LightningModule):
    """Train and evaluate a segmentation model chain on chip batches.

    Every split scores supplied class targets, excluding `ignore_index`.

    Args:
        model_chain: Ordered stage construction specifications. Each selects a
            registered name or class_path with optional init_args. The last
            stage states the class count as `num_classes`.
        ignore_index: Class index excluded from loss and metrics.
        criterion: Loss class and arguments. Defaults to CrossEntropyLoss with
            ignore_index; an explicit criterion ignore_index must match the task.
        optimizer: Optimizer class, arguments, and groups by stage. Defaults to AdamW with a learning rate of 1e-3.
        lr_scheduler: Scheduler class, arguments, and Lightning metadata.
            None disables scheduling.
        metrics: Metric names in dot notation (e.g. ``["iou.macro", "f1.macro"]``).
        class_thresholds: Per-class confidence threshold, one per class the
            last stage states. None initializes every class to 0.5. Checkpoint
            loading restores saved thresholds; an explicitly configured
            calibration callback may update them.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave.ml.tasks.segmentation.supervised.Module
          init_args:
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              decoder: {name: dpt}
              head:
                name: segmentation
                init_args:
                  num_classes: 2
    """

    model: ModelChain
    num_classes: int
    class_thresholds: torch.Tensor

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
    ) -> None:
        super().__init__()

        if not model_chain:
            raise ValueError("Supply at least one model stage")
        if criterion is None:
            criterion = {"name": "cross_entropy"}
        if optimizer is None:
            optimizer = {"name": "adamw", "init_args": {"lr": 1e-3}}
        self.save_hyperparameters()
        self.model_chain = model_chain

        self.ignore_index = ignore_index

        self.optimizer_spec = deepcopy(optimizer)
        self.scheduler_spec = deepcopy(lr_scheduler)
        self.metrics_config = metrics
        self._initial_class_thresholds = class_thresholds

        criterion_spec = deepcopy(criterion)
        configured_ignore = criterion_spec.get("init_args", {}).get(
            "ignore_index", ignore_index
        )
        if configured_ignore != ignore_index:
            raise ValueError("criterion ignore_index must match the task ignore_index")
        criterion_spec["init_args"] = {
            "ignore_index": ignore_index,
            **criterion_spec.get("init_args", {}),
        }
        self.criterion = build_criterion(criterion_spec)

    def configure_model(self):
        """Construct the model, then the thresholds and metrics its classes size.

        Returns:
            (None): Initializes module state in place, or leaves an already
                constructed model unchanged.

        Raises:
            TypeError: The last stage states no integer `num_classes`.
            ValueError: `class_thresholds` does not hold one entry per class.
        """
        if hasattr(self, "model"):
            return

        last = list(self.model_chain)[-1]
        self.model = build_model(deepcopy(self.model_chain))

        head = self.model.get_submodule(last)
        num_classes = getattr(head, "num_classes", None)
        if not isinstance(num_classes, int):
            raise TypeError(
                f"stage {last!r} ({type(head).__name__}) states no num_classes; "
                f"the last stage names how many classes it predicts"
            )
        self.num_classes = num_classes
        thresholds = self._initial_class_thresholds
        if thresholds is not None and len(thresholds) != num_classes:
            raise ValueError(
                f"class_thresholds must have {num_classes} entries, "
                f"got {len(thresholds)}"
            )
        self.register_buffer(
            "class_thresholds",
            torch.full((num_classes,), 0.5)
            if thresholds is None
            else torch.tensor(thresholds),
        )

        # Lightning runs setup before configure_model, so metrics are built here.
        metrics = MulticlassMetrics(
            num_classes=num_classes,
            ignore_index=self.ignore_index,
            metrics=self.metrics_config,
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
        optimizer = build_optimizer(self.optimizer_spec, self.model)
        if self.scheduler_spec is None:
            return optimizer
        return cast(
            OptimizerLRScheduler,
            {
                "optimizer": optimizer,
                "lr_scheduler": build_scheduler(self.scheduler_spec, optimizer),
            },
        )

    def forward(self, **model_inputs: Any) -> torch.Tensor:
        """Return raw per-pixel logits for prepared model inputs.

        Args:
            **model_inputs: Prepared named model tensors. ``image`` holds the
                selected encoder's input layout, including a time axis when
                required; contextual tensors are ordinary keys.

        Returns:
            Raw per-pixel logits.

        Raises:
            TypeError: If the configured model does not return one tensor.
        """
        logits = self.model(**model_inputs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "Semantic segmentation models must return logits as a tensor"
            )
        return logits

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        """Score segmentation targets and log training loss and class metrics.

        Args:
            batch: Prepared `Sample` with named inputs and `(batch, y, x)`
                int64 targets. Unknown targets hold the loss's ignore value.
            batch_idx: Lightning's batch position; unused by this task.

        Returns:
            Criterion loss on the class logits and supplied targets, used
            by Lightning for backpropagation.
        """
        target = batch["target"]
        logits = self(**batch["inputs"])
        loss = self._loss(logits, target)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=target.shape[0],
        )
        if target.ne(self.ignore_index).any():
            self.train_metrics.update(logits, target)
        self._log_metrics(
            self.train_metrics,
            target.ne(self.ignore_index).sum(),
            batch_size=target.shape[0],
        )

        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> dict[str, torch.Tensor]:
        """Score validation chips over their known labels.

        Args:
            batch: Prepared `Sample` with named inputs and int64 targets
                shaped `(batch, y, x)`.
            batch_idx: Lightning's batch position; unused by this task.

        Returns:
            `logits` shaped `(batch, num_classes, y, x)` and `label` shaped
            `(batch, y, x)`, retaining ignored labels, for
            validation callbacks. Loss and metrics are logged separately.
        """
        logits, target = self._evaluate(batch)
        self.log(
            "val_loss",
            self._loss(logits, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(target),
        )
        if target.ne(self.ignore_index).any():
            self.val_metrics.update(logits, target)
        self._log_metrics(
            self.val_metrics, target.ne(self.ignore_index).sum(), batch_size=len(target)
        )
        return {"logits": logits, "label": target}

    def test_step(self, batch: Sample, batch_idx: int) -> dict[str, torch.Tensor]:
        """Score test chips over their known labels.

        Args:
            batch: Prepared `Sample` with named inputs and int64 targets
                shaped `(batch, y, x)`.
            batch_idx: Lightning's batch position; unused by this task.

        Returns:
            `logits` shaped `(batch, num_classes, y, x)` and `label` shaped
            `(batch, y, x)`, retaining ignored labels, for
            test callbacks. Metrics are logged separately.
        """
        logits, target = self._evaluate(batch)
        if target.ne(self.ignore_index).any():
            self.test_metrics.update(logits, target)
        self._log_metrics(
            self.test_metrics,
            target.ne(self.ignore_index).sum(),
            batch_size=len(target),
        )
        return {"logits": logits, "label": target}

    def predict_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        """Return the logits of chips that may carry no label.

        Args:
            batch: Prepared `Sample`; a `target`, when present, is ignored.
            batch_idx: Lightning's batch position; unused by this task.

        Returns:
            Raw per-pixel logits shaped `(batch, num_classes, y, x)`.
        """
        return self(**batch["inputs"])

    def _evaluate(self, batch: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Return chip logits and the supplied class targets."""
        logits = self(**batch["inputs"])
        target = batch["target"]
        return logits, target

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Score known labels; an entirely ignored target gives zero gradient."""
        if not target.ne(self.ignore_index).any():
            return logits.sum() * 0
        return self.criterion(logits, target)

    def _log_metrics(
        self, metrics: MetricCollection, count: torch.Tensor, *, batch_size: int
    ) -> None:
        """Log on every rank when any rank has target entries to score."""
        if self._trainer is not None:
            count = self.trainer.strategy.reduce(count, reduce_op="sum")
        if count:
            self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch_size)
