from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_warn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from tiler import Merger

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS
from geosave_engine.geodata.warnings import GeoSaveWarning
from geosave_engine.ml.builders import build_criterion, build_optimizer, build_scheduler
from geosave_engine.ml.metrics import RegressionMetrics
from geosave_engine.model.chain import ModelChain
from geosave_engine.model.registry import build_model

from .data import Batch, Dataset


def _dataset(loaders: Any, dataloader_idx: int) -> Any:
    """Return the dataset of the loader Lightning is reading."""
    if isinstance(loaders, (list, tuple)):
        loaders = loaders[dataloader_idx]
    return loaders.dataset


@dataclass
class _Evaluation:
    """Chips one loader has still to deliver, and the rasters left open."""

    remaining: set[int]
    mergers: dict[int, Merger] = field(default_factory=dict)
    halos: dict[int, list[tuple[int, int]]] = field(default_factory=dict)


class Module(LightningModule):
    """Train and evaluate a pixelwise regression model chain on tile batches.

    Training scores each tile; validation and test merge tile values and
    score each raster whole, on a single device. Only pixels with a known
    value are scored.

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
          class_path: geosave_engine.ml.regression.supervised.Module
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

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        model_inputs, target, _, _ = batch

        predictions = self(**model_inputs)
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

    def on_validation_epoch_start(self) -> None:
        """Start merging afresh, dropping rasters a cut-short run left open."""
        self._start_evaluation()

    def on_test_epoch_start(self) -> None:
        """Start merging afresh, dropping rasters a cut-short run left open."""
        self._start_evaluation()

    def _start_evaluation(self) -> None:
        """Require local scene completion before initializing merge state."""
        if self._trainer is not None and self.trainer.world_size > 1:
            raise ValueError(
                "scene evaluation requires a single device so each rank can "
                "complete scenes; distributed scene partitioning is not supported"
            )
        self._evaluation: dict[int, _Evaluation] = {}
        self._raster_count = 0

    def on_validation_epoch_end(self) -> None:
        """Say so when a run cut short scored nothing, so no `val_loss` exists."""
        if (
            self._evaluation
            and not self._raster_count
            and not self.trainer.sanity_checking
        ):
            rank_zero_warn(
                "validation ended before any raster was whole, so nothing was "
                "scored and val_loss was not logged; a raster is scored once "
                "all its tiles have been read, so let validation read at least "
                "one whole raster (raise limit_val_batches)",
                category=GeoSaveWarning,
            )

    def validation_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Merge validation tiles and score each raster once it is whole."""
        model_inputs, _, ids, valid = batch
        predictions = self(**model_inputs)

        dataset = _dataset(self.trainer.val_dataloaders, dataloader_idx)
        for values, target in self._merge(
            predictions, ids, valid, dataset, dataloader_idx
        ):
            known = torch.isfinite(target)
            self.log(
                "val_loss",
                self.criterion(values[known], target[known]),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
            )
            self.val_metrics.update(values[known], target[known])
            self.log_dict(
                self.val_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )

    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Merge test tiles and score each raster once it is whole."""
        model_inputs, _, ids, valid = batch
        predictions = self(**model_inputs)

        dataset = _dataset(self.trainer.test_dataloaders, dataloader_idx)
        for values, target in self._merge(
            predictions, ids, valid, dataset, dataloader_idx
        ):
            known = torch.isfinite(target)
            self.test_metrics.update(values[known], target[known])
            self.log_dict(
                self.test_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )

    def _merge(
        self,
        predictions: torch.Tensor,
        ids: Sequence[int],
        valid: torch.Tensor,
        dataset: Dataset,
        dataloader_idx: int = 0,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Add chip values to their rasters and return the rasters they complete.

        Chips may arrive in any order. A raster is released once every chip
        cut from it has been added, paired with its original values, so
        evaluation scores whole rasters and never blends chip targets.

        Args:
            predictions: ``(B, variable, y, x)`` chip values.
            ids: Each chip's row in `dataset.windows`.
            valid: ``(B, y, x)`` pixels whose inputs were valid.
            dataset: Dataset the chips were read from.
            dataloader_idx: Loader the chips came from; loaders merge apart.

        Returns:
            ``(values, target)`` per completed raster holding a known pixel,
            both shaped ``(1, variable, y, x)``; target is NaN where unknown
            or no valid chip covers it.

        Raises:
            ValueError: A chip was added twice, or the target grid differs
                from the prediction grid.
        """
        if dataloader_idx not in self._evaluation:
            self._evaluation[dataloader_idx] = _Evaluation(
                set(range(len(dataset.windows)))
            )
        state = self._evaluation[dataloader_idx]
        chips = dataset.spec.chips
        assert chips is not None, "a dataset is cut from a spec declaring chips"
        positions = [int(index) for index in ids]
        rows = dataset.windows.iloc[positions]
        pixels = predictions.detach().float().cpu().numpy()
        masks = valid.cpu().numpy() & np.isfinite(pixels).all(axis=1)

        # Accumulate every chip into the merger of its raster.
        for sample_id, (_, row), values, mask in zip(
            positions, rows.iterrows(), pixels, masks, strict=True
        ):
            if sample_id not in state.remaining:
                raise ValueError(f"prediction {sample_id!r} was already accumulated")
            parent_id = int(row.parent)
            if parent_id not in state.mergers:
                # The layout a cut used is rebuilt from the same shape and settings.
                shape = dataset.frames.iloc[parent_id]["proj:shape"]
                height, width = (int(edge) for edge in shape)
                tiler, state.halos[parent_id] = chips.layout((height, width))
                state.mergers[parent_id] = Merger(
                    tiler,
                    logits=len(self.variables) + 1,
                    window=chips.window,
                    save_visits=False,
                    data_dtype=np.float64,
                    weights_dtype=np.float64,
                )
            # Coverage uses the same taper as values, excluding invalid pixels.
            contribution = np.concatenate([np.where(mask, values, 0), mask[None]])
            state.mergers[parent_id].add(int(row.chip), contribution)
            state.remaining.remove(sample_id)

        # Release every raster whose chips have all arrived.
        completed = []
        for parent_id in (int(parent) for parent in rows.parent.unique()):
            expected = dataset.windows.index[dataset.windows["parent"] == parent_id]
            if state.remaining.intersection(int(index) for index in expected):
                continue
            raster = self._complete(state, parent_id, dataset, predictions.device)
            if raster is not None:
                completed.append(raster)
        return completed

    def _complete(
        self,
        state: _Evaluation,
        parent_id: int,
        dataset: Dataset,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Divide a raster's value sums by coverage and pair them with its values."""
        sums = state.mergers.pop(parent_id).merge(
            extra_padding=state.halos.pop(parent_id),
            normalize_by_weights=False,
        )
        coverage = sums[-1]
        merged = np.full(sums[:-1].shape, np.nan, dtype=np.float32)
        np.divide(sums[:-1], coverage, out=merged, where=coverage > 0)

        # The original values are read whole, never stitched from chip targets.
        parent = dataset.frames.iloc[parent_id].gs.to_stack()
        image = next(iter(parent.gs.rasters.values()))
        truth = parent[dataset.target].dataset
        image_shape = tuple(image.sizes[d] for d in SPATIAL_DIMENSIONS)
        truth_shape = tuple(truth.sizes[d] for d in SPATIAL_DIMENSIONS)
        if image.gs.geobox != truth.gs.geobox or image_shape != truth_shape:
            raise ValueError(f"target grid differs from prediction for {parent_id!r}")
        target = truth.gs.mask_and_scale().gs.to_tensor(dtype="float32").to(device)
        values = torch.from_numpy(merged).to(device)
        if target.shape != values.shape:
            raise ValueError(f"target shape differs from prediction for {parent_id!r}")
        target = target.masked_fill(~torch.isfinite(values), torch.nan)
        self._raster_count += 1

        if not torch.isfinite(target).any():
            return None
        return torch.nan_to_num(values).unsqueeze(0), target.unsqueeze(0)

    def predict_step(
        self,
        batch: tuple[dict[str, Any], Sequence[int]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, Sequence[int]]:
        """Return chip values beside their window positions.

        Args:
            batch: ``(model_inputs, ids)`` from a method-owned prediction Dataset.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning prediction loader number.

        Returns:
            ``(values, ids)``. IDs are the chips' rows in the dataset's `windows`.
        """
        model_inputs, ids = batch
        return self(**model_inputs), ids
