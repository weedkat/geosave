from functools import partial
from typing import Any

import torch
from torchmetrics import MetricCollection
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from geosave.ml.torch.training.optim import configure_optimizers
from geosave.ml.torch.training.metrics import RegressionMetrics
from geosave.ml.torch import ModelChain

from .datamodule import Sample


class Module(LightningModule):
    """Train and evaluate a pixelwise regression model chain on chip batches.

    Every split scores pixels with known target values.

    Args:
        model_chain: Ordered stage construction specifications. The last
            stage names its predicted `variables`, one channel each.
        criterion: Loss module, applied to known pixels. Defaults to MSELoss.
        optimizer: Optimizer class and arguments, built over the parameter
            groups. Defaults to AdamW with a learning rate of 1e-3.
        param_groups: Optimizer options per chain stage, such as an
            encoder's `lr`. None puts every parameter in one group.
        lr_scheduler: Scheduler class and arguments. None disables scheduling.
        lr_scheduler_config: Lightning's scheduler options, such as `interval`.
        metrics: Names among ``mae``, ``rmse`` and ``r2``. None selects all.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: training.Module
          init_args:
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              decoder: {name: dpt}
              head: {name: regression, init_args: {variables: [biomass]}}
    """

    model: ModelChain
    variables: list[str]

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
            criterion = nn.MSELoss()
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
        """Construct the model, then the metrics.

        Raises:
            TypeError: The last stage names no `variables`.
        """
        if hasattr(self, "model"):
            return

        last = list(self.model_chain)[-1]
        self.model = ModelChain.from_config(self.model_chain)

        head = self.model.get_submodule(last)
        variables = getattr(head, "variables", None)
        if not isinstance(variables, list):
            raise TypeError(
                f"stage {last!r} ({type(head).__name__}) names no variables; "
                f"the last stage names the quantities it predicts"
            )
        self.variables = variables

        # Lightning runs setup before configure_model, so metrics are built here.
        metrics = RegressionMetrics(self.metrics_config)
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
        """Return ``(B, variable, y, x)`` values for prepared model inputs.

        Args:
            **model_inputs: Named, batched tensors and context required by the
                configured chain. Pixel preparation is the caller's job.

        Returns:
            `(batch, variable, y, x)` predictions in the head's variable order.

        Raises:
            TypeError: If the configured model does not return one tensor.
        """
        values = self.model(**model_inputs)
        if not isinstance(values, torch.Tensor):
            raise TypeError("Regression models must return values as a tensor")
        return values

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        """Score known target values and log regression training metrics.

        Args:
            batch: Prepared `Sample` with named inputs, floating targets
                `(batch, variable, y, x)`. Non-finite targets are excluded.
            batch_idx: Lightning's batch position; unused by this task.

        Returns:
            Criterion loss on the finite target entries, used by Lightning
            for backpropagation.
        """
        batch_size = len(batch["target"])
        predictions, target = self._evaluate(batch)
        loss = self._loss(predictions, target)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        if target.numel():
            self.train_metrics.update(predictions.detach(), target)
        self._log_metrics(
            self.train_metrics,
            target.new_tensor(target.numel(), dtype=torch.long),
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> None:
        """Log validation loss and metrics over finite target values.

        Args:
            batch: Prepared `Sample` with named inputs, floating targets
                `(batch, variable, y, x)`.
            batch_idx: Lightning's batch position; unused by this task.
        """
        predictions, target = self._evaluate(batch)
        self.log(
            "val_loss",
            self._loss(predictions, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["target"]),
        )
        if target.numel():
            self.val_metrics.update(predictions, target)
        self._log_metrics(
            self.val_metrics,
            target.new_tensor(target.numel(), dtype=torch.long),
            batch_size=len(batch["target"]),
        )

    def test_step(self, batch: Sample, batch_idx: int) -> None:
        """Log test metrics over finite target values.

        Args:
            batch: Prepared `Sample` with named inputs, floating targets
                `(batch, variable, y, x)`.
            batch_idx: Lightning's batch position; unused by this task.
        """
        predictions, target = self._evaluate(batch)
        if target.numel():
            self.test_metrics.update(predictions, target)
        self._log_metrics(
            self.test_metrics,
            target.new_tensor(target.numel(), dtype=torch.long),
            batch_size=len(batch["target"]),
        )

    def _evaluate(self, batch: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted and true values at known target entries."""
        predictions = self(**batch["inputs"])
        target = batch["target"]
        scored = torch.isfinite(target)
        return predictions[scored], target[scored]

    def _loss(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Score selected target entries; an empty selection gives zero gradient."""
        if not target.numel():
            return predictions.sum() * 0
        return self.criterion(predictions, target)

    def _log_metrics(
        self, metrics: MetricCollection, count: torch.Tensor, *, batch_size: int
    ) -> None:
        """Log on every rank when any rank has target entries to score."""
        if self._trainer is not None:
            count = self.trainer.strategy.reduce(count, reduce_op="sum")
        if count:
            self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch_size)
