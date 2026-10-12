from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch

from geosave.ml.tasks.regression import supervised
from geosave.model.chain import ModelChain

from .data import TrainingBatch


class Module(supervised.Module):
    """Train pixelwise regression with Mean Teacher on labeled and unlabeled chips.

    An EMA teacher's weak-view values supervise the student's strong view
    under the labeled criterion, weighted by ``exp(-5 (1 - t)^2)`` over
    `rampup_steps` (Tarvainen and Valpola, 2017).

    Args:
        model_chain: Ordered stage construction specifications. The last
            stage predicts one channel per target variable.
        criterion: Loss for labeled and consistency terms. Defaults to MSELoss.
        optimizer: Optimizer class, arguments, and groups by stage.
        lr_scheduler: Scheduler class, arguments, and Lightning metadata.
        metrics: Names among ``mae``, ``rmse`` and ``r2``. None selects all.
        unlabeled_weight: Weight of the consistency loss once ramped up.
        rampup_steps: Optimizer steps over which the weight ramps up; 0
            starts at full weight.
        ema: Share of the teacher kept at each step.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave.ml.tasks.regression.mean_teacher.Module
          init_args:
            rampup_steps: 2000
            model_chain:
              encoder: {name: prithvi}
              decoder: {name: unet}
              head: {name: regression, init_args: {variables: [biomass]}}
    """

    teacher: ModelChain

    def __init__(
        self,
        *,
        model_chain: dict[str, dict[str, Any]],
        criterion: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        unlabeled_weight: float = 1.0,
        rampup_steps: int = 1000,
        ema: float = 0.99,
    ) -> None:
        super().__init__(
            model_chain=model_chain,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
        )
        self.unlabeled_weight = unlabeled_weight
        self.rampup_steps = rampup_steps
        self.ema = ema

    def configure_model(self) -> None:
        """Construct the student and its teacher."""
        if hasattr(self, "teacher"):
            return
        super().configure_model()
        self.teacher = deepcopy(self.model).requires_grad_(False)

    def consistency_weight(self) -> float:
        """Return the consistency weight at the current step."""
        if not self.rampup_steps:
            return self.unlabeled_weight
        progress = min(self.global_step / self.rampup_steps, 1.0)
        return self.unlabeled_weight * math.exp(-5.0 * (1.0 - progress) ** 2)

    # Training reads labeled and unlabeled parts; evaluation keeps the supervised batch.
    def training_step(  # type: ignore[override]
        self, batch: TrainingBatch, batch_idx: int
    ) -> torch.Tensor:
        target = batch["labeled"]["target"]
        weak = batch["unlabeled"]["weak"]
        strong = batch["unlabeled"]["strong"]
        valid = batch["unlabeled"]["valid"]

        # Labeled loss on pixels with a known value.
        predictions = self(**batch["labeled"]["inputs"])
        known = torch.isfinite(target)
        labeled_loss = self.criterion(predictions[known], target[known])

        # Consistency: the student's strong view against the teacher's weak view.
        self.teacher.eval()
        with torch.no_grad():
            taught = self.teacher(**weak)
        scored = valid.unsqueeze(1).expand_as(taught)
        unlabeled_loss = self.criterion(self(**strong)[scored], taught[scored])

        loss = labeled_loss + self.consistency_weight() * unlabeled_loss

        batch_size = len(target)
        self.train_metrics.update(predictions[known].detach(), target[known])
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "train_unlabeled_loss", unlabeled_loss, on_epoch=True, batch_size=batch_size
        )
        self.log_dict(
            self.train_metrics, on_step=False, on_epoch=True, batch_size=batch_size
        )
        return loss

    @torch.no_grad()
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Move the teacher's weights toward the student's; copy its buffers."""
        for kept, trained in zip(
            self.teacher.parameters(), self.model.parameters(), strict=True
        ):
            kept.lerp_(trained, 1 - self.ema)
        for kept, trained in zip(
            self.teacher.buffers(), self.model.buffers(), strict=True
        ):
            kept.copy_(trained)
