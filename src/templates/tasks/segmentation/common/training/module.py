from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torchmetrics import MetricCollection

from geosave.ml.torch.training.optim import configure_optimizers
from geosave.ml.torch.training.metrics import SemanticSegmentationMetrics
from geosave.ml.torch import ModelChain

from .datamodule import InputSample, Sample


class Module(LightningModule):
    """Train on prepared segmentation chips.

    Validation and test are extension points pending the evaluation input contract.

    Args:
        model_chain: Ordered stage specifications. Each selects a registered
            name or class_path with init_args. The last stage states the class
            count as `num_classes`.
        ignore_index: Class index excluded from loss and metrics.
        criterion: Loss module. Its `ignore_index` must equal `ignore_index`.
        optimizer: Optimizer class and arguments, built over the parameter groups.
        param_groups: Optimizer options per chain stage, such as an encoder's
            `lr`. None puts every parameter in one group.
        lr_scheduler: Scheduler class and arguments. None keeps the learning
            rate constant.
        lr_scheduler_config: Lightning's scheduler options, such as `interval`.
        metrics: Metric names, such as `iou_macro`.

    Raises:
        ValueError: The criterion ignores another index than `ignore_index`.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: training.module.Module
          init_args:
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              decoder: {name: dpt}
              head: {name: segmentation, init_args: {num_classes: 2}}
            ignore_index: 255
            criterion: {class_path: torch.nn.CrossEntropyLoss, init_args: {ignore_index: 255}}
            optimizer: {class_path: torch.optim.AdamW, init_args: {lr: 0.001}}
            param_groups: null
            lr_scheduler: null
            lr_scheduler_config: null
            metrics: [iou_macro, f1_macro]
    """

    model: ModelChain
    num_classes: int
    train_metrics: MetricCollection
    val_metrics: MetricCollection
    test_metrics: MetricCollection

    def __init__(
        self,
        *,
        model_chain: dict[str, dict[str, Any]],
        ignore_index: int,
        criterion: nn.Module,
        optimizer: OptimizerCallable,
        param_groups: dict[str, dict[str, Any]] | None,
        lr_scheduler: LRSchedulerCallable | None,
        lr_scheduler_config: dict[str, Any] | None,
        metrics: list[str],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if getattr(criterion, "ignore_index", None) != ignore_index:
            raise ValueError("criterion.ignore_index must equal ignore_index")

        self.model_chain = model_chain
        self.ignore_index = ignore_index
        self.criterion = criterion
        self.optimizer = optimizer
        self.param_groups = param_groups
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_config = lr_scheduler_config
        self.metric_names = metrics

    def configure_model(self) -> None:
        """Build the chain, then the metrics its class count sizes.

        Raises:
            TypeError: The last stage states no integer `num_classes`.
        """
        if hasattr(self, "model"):
            return

        self.model = ModelChain.from_config(self.model_chain)

        last = list(self.model_chain)[-1]
        head = self.model.get_submodule(last)
        num_classes = getattr(head, "num_classes", None)
        if not isinstance(num_classes, int):
            raise TypeError(f"the last stage {last!r} states no integer num_classes")
        self.num_classes = num_classes

        # Lightning runs setup before configure_model, so metrics are built here.
        metrics = SemanticSegmentationMetrics(
            num_classes=num_classes,
            ignore_index=self.ignore_index,
            metrics=self.metric_names,
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self) -> Any:
        """Return the optimizer, with its scheduler when one is configured."""
        return configure_optimizers(
            self.model,
            self.optimizer,
            self.param_groups,
            self.lr_scheduler,
            self.lr_scheduler_config,
        )

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Return raw per-pixel logits shaped `(batch, class, y, x)`.

        Args:
            **inputs: Named tensors prepared for the chain, such as `image`
                and, for a model that uses them, `centroid`, `year` and
                `day_of_year`.
        """
        return self.model(**inputs)

    def labeled_loss(self, batch: Sample) -> torch.Tensor:
        """Score known labels and update their metrics, including all-ignored batches."""
        target = batch["target"]
        logits = self(**batch["inputs"])
        if target.ne(self.ignore_index).any():
            self.train_metrics.update(logits, target)
            return self.criterion(logits, target)
        return logits.sum() * 0

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        """Return the chip loss and log it with the class metrics."""
        loss = self.labeled_loss(batch)
        batch_size = len(batch["target"])
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log_dict(self.train_metrics, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> dict[str, torch.Tensor]:
        """Evaluate validation data once the evaluation input contract is defined."""
        raise NotImplementedError(
            "The segmentation validation input contract is pending"
        )

    def test_step(self, batch: Sample, batch_idx: int) -> dict[str, torch.Tensor]:
        """Evaluate test data once the evaluation input contract is defined."""
        raise NotImplementedError("The segmentation test input contract is pending")

    def predict_step(self, batch: InputSample, batch_idx: int) -> torch.Tensor:
        """Return the logits of chips that may carry no label."""
        return self(**batch["inputs"])
