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
from geosave_engine.ml.metrics import MulticlassMetrics
from geosave_engine.ml.builders import build_criterion, build_optimizer, build_scheduler
from geosave_engine.model.chain import ModelChain
from geosave_engine.model.registry import build_model
from geosave_engine.geodata.warnings import GeoSaveWarning

from .data import Dataset, _read_target


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
    """Train and evaluate a segmentation model chain on tile batches.

    Training scores each tile; validation and test merge tile logits and
    score each raster whole, on a single device.

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
          class_path: geosave_engine.ml.segmentation.supervised.Module
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

    def training_step(
        self,
        batch: tuple[dict[str, Any], torch.Tensor, Sequence[int], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        model_inputs, target, _, _ = batch

        logits = self(**model_inputs)
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
        # Only continuous logits are assembled; categorical targets stay native.
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
        self,
        batch: tuple[dict[str, Any], torch.Tensor, Sequence[int], torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Merge validation tiles and score each raster once it is whole.

        Args:
            batch: ``(model_inputs, target, ids, valid)`` prepared for the model chain.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning validation loader number.

        Returns:
            Tile logits and labels for validation callbacks.
        """
        model_inputs, target, ids, valid = batch
        logits = self(**model_inputs)

        dataset = _dataset(self.trainer.val_dataloaders, dataloader_idx)
        for raster_logits, raster_target in self._merge(
            logits, ids, valid, dataset, dataloader_idx
        ):
            self.log(
                "val_loss",
                self.criterion(raster_logits, raster_target),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
            )
            self.val_metrics.update(raster_logits, raster_target)
            self.log_dict(
                self.val_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )
        return {"logits": logits, "label": target}

    def test_step(
        self,
        batch: tuple[dict[str, Any], torch.Tensor, Sequence[int], torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Merge test tiles and score each raster once it is whole.

        Args:
            batch: ``(model_inputs, target, ids, valid)`` prepared for the model chain.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning test loader number.

        Returns:
            Tile logits and labels for test callbacks.
        """
        model_inputs, target, ids, valid = batch
        logits = self(**model_inputs)

        dataset = _dataset(self.trainer.test_dataloaders, dataloader_idx)
        for raster_logits, raster_target in self._merge(
            logits, ids, valid, dataset, dataloader_idx
        ):
            self.test_metrics.update(raster_logits, raster_target)
            self.log_dict(
                self.test_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )
        return {"logits": logits, "label": target}

    def _merge(
        self,
        logits: torch.Tensor,
        ids: Sequence[int],
        valid: torch.Tensor,
        dataset: Dataset,
        dataloader_idx: int = 0,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Add chip logits to their rasters and return the rasters they complete.

        Chips may arrive in any order. A raster is released once every chip
        cut from it has been added, paired with its original labels, so
        evaluation scores whole rasters and never blends chip targets.

        Args:
            logits: ``(B, classes, y, x)`` chip logits.
            ids: Each chip's row in `dataset.windows`.
            valid: ``(B, y, x)`` pixels whose inputs were valid.
            dataset: Dataset the chips were read from.
            dataloader_idx: Loader the chips came from; loaders merge apart.

        Returns:
            ``(logits, target)`` per completed raster holding a scored pixel,
            shaped ``(1, classes, y, x)`` and ``(1, y, x)``.

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
        pixels = logits.detach().float().cpu().numpy()
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
                    logits=self.num_classes + 1,
                    window=chips.window,
                    save_visits=False,
                    data_dtype=np.float64,
                    weights_dtype=np.float64,
                )
            # Coverage uses the same taper as logits, excluding invalid pixels.
            contribution = np.concatenate([np.where(mask, values, 0), mask[None]])
            state.mergers[parent_id].add(int(row.chip), contribution)
            state.remaining.remove(sample_id)

        # Release every raster whose chips have all arrived.
        completed = []
        for parent_id in (int(parent) for parent in rows.parent.unique()):
            expected = dataset.windows.index[dataset.windows["parent"] == parent_id]
            if state.remaining.intersection(int(index) for index in expected):
                continue
            raster = self._complete(state, parent_id, dataset, logits.device)
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
        """Divide a raster's logit sums by coverage and pair them with its labels."""
        sums = state.mergers.pop(parent_id).merge(
            extra_padding=state.halos.pop(parent_id),
            normalize_by_weights=False,
        )
        coverage = sums[-1]
        values = np.full(sums[:-1].shape, np.nan, dtype=np.float32)
        np.divide(sums[:-1], coverage, out=values, where=coverage > 0)

        # The original labels are read whole, never stitched from chip targets.
        parent = dataset.frames.iloc[parent_id].gs.to_stack()
        image = next(iter(parent.gs.rasters.values()))
        labels = parent[dataset.target].dataset
        image_shape = tuple(image.sizes[d] for d in SPATIAL_DIMENSIONS)
        label_shape = tuple(labels.sizes[d] for d in SPATIAL_DIMENSIONS)
        if image.gs.geobox != labels.gs.geobox or image_shape != label_shape:
            raise ValueError(f"target grid differs from prediction for {parent_id!r}")
        target = _read_target(labels, self.ignore_index).to(device)
        logits = torch.from_numpy(values).to(device)
        covered = torch.isfinite(logits).all(dim=0)
        if target.shape != covered.shape:
            raise ValueError(f"target shape differs from prediction for {parent_id!r}")
        target = target.masked_fill(~covered, self.ignore_index)
        self._raster_count += 1

        if not (target != self.ignore_index).any():
            return None
        return torch.nan_to_num(logits).unsqueeze(0), target.unsqueeze(0)

    def predict_step(
        self,
        batch: tuple[dict[str, Any], Sequence[int]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, Sequence[int]]:
        """Return raw chip logits beside their window positions.

        Args:
            batch: ``(model_inputs, ids)`` from a method-owned prediction Dataset.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning prediction loader number.

        Returns:
            ``(logits, ids)``. IDs are the chips' rows in the dataset's `windows`.
        """
        model_inputs, ids = batch
        return self(**model_inputs), ids
