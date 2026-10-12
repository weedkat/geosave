from copy import deepcopy
from typing import Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from geosave_engine.ml.builders import build_criterion, build_optimizer, build_scheduler
from geosave_engine.ml.metrics import RegressionMetrics
from geosave_engine.model.chain import ModelChain
from geosave_engine.model.registry import build_model

from .data import Sample


class Module(LightningModule):
    """Train and evaluate a pixelwise regression model chain on chip batches.

    Every split scores chips one by one, over pixels with a known value;
    validation and test also skip pixels whose inputs are not valid.

    Args:
        model_chain: Ordered stage construction specifications. The last
            stage names its predicted `variables`, one channel each.
        criterion: Loss class and arguments, applied to known pixels.
            Defaults to MSELoss.
        optimizer: Optimizer class, arguments, and groups by stage. Defaults
            to AdamW with a learning rate of 1e-3.
        lr_scheduler: Scheduler class, arguments, and Lightning metadata.
            None disables scheduling.
        metrics: Names among ``mae``, ``rmse`` and ``r2``. None selects all.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave_engine.ml.tasks.regression.supervised.Module
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
        criterion: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()

        if not model_chain:
            raise ValueError("Supply at least one model stage")
        if criterion is None:
            criterion = {"class_path": "torch.nn.MSELoss"}
        if optimizer is None:
            optimizer = {"name": "adamw", "init_args": {"lr": 1e-3}}
        self.save_hyperparameters()
        self.model_chain = model_chain

        self.optimizer_spec = deepcopy(optimizer)
        self.scheduler_spec = deepcopy(lr_scheduler)
        self.metrics_config = metrics
        self.criterion = build_criterion(deepcopy(criterion))

    def configure_model(self) -> None:
        """Construct the model, then the metrics.

        Raises:
            TypeError: The last stage names no `variables`.
        """
        if hasattr(self, "model"):
            return

        last = list(self.model_chain)[-1]
        self.model = build_model(deepcopy(self.model_chain))

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
        """Return ``(B, variable, y, x)`` values for prepared model inputs.

        Raises:
            TypeError: If the configured model does not return one tensor.
        """
        values = self.model(**model_inputs)
        if not isinstance(values, torch.Tensor):
            raise TypeError("Regression models must return values as a tensor")
        return values

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        target = batch["target"]
        predictions = self(**batch["inputs"])
        known = torch.isfinite(target)
        loss = self.criterion(predictions[known], target[known])

        self.train_metrics.update(predictions[known].detach(), target[known])
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

    def validation_step(self, batch: Sample, batch_idx: int) -> None:
        """Score validation chips over their known, valid pixels."""
        predictions, target = self._evaluate(batch)
        self.log(
            "val_loss",
            self.criterion(predictions, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["target"]),
        )
        self.val_metrics.update(predictions, target)
        self.log_dict(
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch["target"]),
        )

    def test_step(self, batch: Sample, batch_idx: int) -> None:
        """Score test chips over their known, valid pixels."""
        predictions, target = self._evaluate(batch)
        self.test_metrics.update(predictions, target)
        self.log_dict(
            self.test_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch["target"]),
        )

    def _evaluate(self, batch: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted and true values at known pixels with valid inputs."""
        predictions = self(**batch["inputs"])
        target = batch["target"]
        scored = torch.isfinite(target) & batch["valid"].unsqueeze(1)
        return predictions[scored], target[scored]
