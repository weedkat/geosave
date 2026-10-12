from copy import deepcopy
from typing import Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from geosave.ml.builders import build_criterion, build_optimizer, build_scheduler
from geosave.ml.metrics import MulticlassMetrics
from geosave.model.chain import ModelChain
from geosave.model.registry import build_model

from .data import Sample


class Module(LightningModule):
    """Train and evaluate a segmentation model chain on chip batches.

    Every split scores chips one by one; validation and test score only
    pixels whose inputs are valid.

    Args:
        model_chain: Ordered stage construction specifications. Each selects a
            registered name or class_path with optional init_args. The last
            stage states the class count as `num_classes`.
        ignore_index: Class index excluded from loss and metrics.
        criterion: Loss class and arguments. Defaults to CrossEntropyLoss with
            ignore_index; an explicit criterion ignore_index takes precedence.
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
                  classes: [background, oil_palm]
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
        criterion_spec["init_args"] = {
            "ignore_index": ignore_index,
            **criterion_spec.get("init_args", {}),
        }
        self.criterion = build_criterion(criterion_spec)

    def configure_model(self):
        """Construct the model, then the thresholds and metrics its classes size.

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
        target = batch["target"]
        logits = self(**batch["inputs"])
        loss = self.criterion(logits, target)

        self.train_metrics.update(logits, target)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=target.shape[0],
        )
        self.log_dict(
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=target.shape[0],
        )

        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> dict[str, torch.Tensor]:
        """Score validation chips over their valid pixels.

        Returns:
            Chip logits and the labels scored, for validation callbacks.
        """
        logits, target = self._evaluate(batch)
        self.log(
            "val_loss",
            self.criterion(logits, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(target),
        )
        self.val_metrics.update(logits, target)
        self.log_dict(
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(target),
        )
        return {"logits": logits, "label": target}

    def test_step(self, batch: Sample, batch_idx: int) -> dict[str, torch.Tensor]:
        """Score test chips over their valid pixels.

        Returns:
            Chip logits and the labels scored, for test callbacks.
        """
        logits, target = self._evaluate(batch)
        self.test_metrics.update(logits, target)
        self.log_dict(
            self.test_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(target),
        )
        return {"logits": logits, "label": target}

    def _evaluate(self, batch: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Return chip logits and labels ignoring pixels without valid inputs."""
        logits = self(**batch["inputs"])
        target = batch["target"].masked_fill(~batch["valid"], self.ignore_index)
        return logits, target
