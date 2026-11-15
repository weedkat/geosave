from copy import deepcopy
from typing import Any

import torch
from lightning import LightningModule
from torch.utils.data import default_collate
from torchmetrics import MetricCollection

from geosave.geodata.sampling import TensorChipper, TensorMerger
from geosave.ml.builders import build_criterion, build_optimizer, build_scheduler
from geosave.ml.metrics import MulticlassMetrics
from geosave.ml.chain import ModelChain
from geosave.ml.components.build import build_model
from geosave.ml.factory import BuildSpec
from geosave.ml.processing.predictors import ChipPredictor
from geosave.ml.processing.roles import PREDICTORS

from .data import Sample


class Module(LightningModule):
    """Train on chips; validate and test on whole scenes, predicted chip by chip.

    Validation and test batches hold whole scenes. `TensorChipper` cuts them
    into chips and `TensorMerger` merges the logits back, with the chip
    settings of the release's `ChipPredictor`, so the steps score what a
    release predicts, every scene pixel once. Every argument is required, so
    the YAML config states the whole setup.

    Args:
        model_chain: Ordered stage specifications. Each selects a registered
            name or class_path with init_args. The last stage states the class
            count as `num_classes`.
        ignore_index: Class index excluded from loss and metrics.
        criterion: Loss name or class_path with init_args. Its
            `init_args.ignore_index` must equal `ignore_index`.
        optimizer: Optimizer name or class_path, init_args, and groups by stage.
        lr_scheduler: Scheduler name or class_path, init_args, and Lightning
            metadata such as `interval`. None keeps the learning rate constant.
        metrics: Metric names in dot notation, such as `iou.macro`.
        predictor: The release's chip predictor by name or class_path with
            init_args, such as `{name: chips, init_args: {size: 224}}`.
            Validation and test chip and merge with its settings.

    Raises:
        ValueError: The criterion ignores another index than `ignore_index`.
        TypeError: The predictor is not a `ChipPredictor`.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: modules.segmentation.supervised.Module
          init_args:
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              decoder: {name: dpt}
              head: {name: segmentation, init_args: {num_classes: 2}}
            ignore_index: 255
            criterion: {name: cross_entropy, init_args: {ignore_index: 255}}
            optimizer: {name: adamw, init_args: {lr: 0.001}}
            lr_scheduler: null
            metrics: [iou.macro, f1.macro]
            predictor: {name: chips, init_args: {size: 224, stride: 192}}
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
        criterion: dict[str, Any],
        optimizer: dict[str, Any],
        lr_scheduler: dict[str, Any] | None,
        metrics: list[str],
        predictor: dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if criterion.get("init_args", {}).get("ignore_index") != ignore_index:
            raise ValueError("criterion init_args.ignore_index must equal ignore_index")

        self.model_chain = model_chain
        self.ignore_index = ignore_index
        self.criterion = build_criterion(criterion)
        self.optimizer_spec = optimizer
        self.scheduler_spec = lr_scheduler
        self.metric_names = metrics
        built = PREDICTORS.build(BuildSpec.model_validate(predictor))
        if not isinstance(built, ChipPredictor):
            raise TypeError("validation chips with a ChipPredictor's settings")
        self.predictor = built

    def configure_model(self) -> None:
        """Build the chain, then the metrics its class count sizes.

        Raises:
            TypeError: The last stage states no integer `num_classes`.
        """
        if hasattr(self, "model"):
            return

        self.model = build_model(deepcopy(self.model_chain))

        last = list(self.model_chain)[-1]
        head = self.model.get_submodule(last)
        num_classes = getattr(head, "num_classes", None)
        if not isinstance(num_classes, int):
            raise TypeError(f"the last stage {last!r} states no integer num_classes")
        self.num_classes = num_classes

        # Lightning runs setup before configure_model, so metrics are built here.
        metrics = MulticlassMetrics(
            num_classes=num_classes,
            ignore_index=self.ignore_index,
            metrics=self.metric_names,
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self) -> Any:
        """Return the optimizer, with its scheduler when one is configured."""
        optimizer = build_optimizer(self.optimizer_spec, self.model)
        if self.scheduler_spec is None:
            return optimizer

        scheduler = build_scheduler(self.scheduler_spec, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, **inputs: Any) -> torch.Tensor:
        """Return raw per-pixel logits shaped `(batch, class, y, x)`.

        Args:
            **inputs: Named pixel tensors, such as `image`, and the grid
                entries `transform`, `crs` and `time`. Each stage takes the
                names it declares; an encoder pretrained on location or date
                derives them from the grid entries.
        """
        return self.model(**inputs)

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        """Return the loss of a chip batch and log it with the class metrics."""
        target = batch["target"]
        logits = self(
            **batch["inputs"],
            transform=batch["transform"],
            crs=batch["crs"],
            time=batch.get("time"),
        )

        if target.ne(self.ignore_index).any():
            loss = self.criterion(logits, target)
            self.train_metrics.update(logits, target)
        else:
            # A batch without known labels still backpropagates, as zero,
            # so every distributed rank runs the same steps.
            loss = logits.sum() * 0

        batch_size = len(target)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log_dict(self.train_metrics, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> dict[str, torch.Tensor]:
        """Score whole scenes and return their `logits` and `label` for callbacks."""
        target = batch["target"]

        # Every model input is a pixel tensor on the scene grid; chip them all.
        pixels = []
        for name in batch["inputs"]:
            pixels.append(f"inputs/{name}")

        predictor = self.predictor
        chipper = TensorChipper(
            batch,
            pixels=pixels,
            size=predictor.size,
            stride=predictor.stride,
            mode=predictor.mode,
        )
        merger = TensorMerger(chipper.layout, window=predictor.window)
        for start in range(0, len(chipper), predictor.batch_size):
            indices = range(start, min(start + predictor.batch_size, len(chipper)))
            chips = []
            for index in indices:
                chips.append(chipper[index])
            chips = default_collate(chips)

            chip_logits = self(
                **chips["inputs"],
                transform=chips["transform"],
                crs=chips["crs"],
                time=chips.get("time"),
            )
            for index, logits in zip(indices, chip_logits, strict=True):
                merger.add(index, logits)
        logits = merger.merge()

        # Pixels no chip reached, as "drop" leaves the edges, are not scored.
        unpredicted = logits.isnan().any(dim=1)
        target = target.masked_fill(unpredicted, self.ignore_index)
        logits = logits.nan_to_num()

        if target.ne(self.ignore_index).any():
            loss = self.criterion(logits, target)
            self.val_metrics.update(logits, target)
        else:
            loss = logits.sum() * 0

        batch_size = len(target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(self.val_metrics, on_epoch=True, batch_size=batch_size)

        return {"logits": logits, "label": target}

    def test_step(self, batch: Sample, batch_idx: int) -> dict[str, torch.Tensor]:
        """Score whole scenes and return their `logits` and `label` for callbacks."""
        target = batch["target"]

        # Every model input is a pixel tensor on the scene grid; chip them all.
        pixels = []
        for name in batch["inputs"]:
            pixels.append(f"inputs/{name}")

        predictor = self.predictor
        chipper = TensorChipper(
            batch,
            pixels=pixels,
            size=predictor.size,
            stride=predictor.stride,
            mode=predictor.mode,
        )
        merger = TensorMerger(chipper.layout, window=predictor.window)
        for start in range(0, len(chipper), predictor.batch_size):
            indices = range(start, min(start + predictor.batch_size, len(chipper)))
            chips = []
            for index in indices:
                chips.append(chipper[index])
            chips = default_collate(chips)

            chip_logits = self(
                **chips["inputs"],
                transform=chips["transform"],
                crs=chips["crs"],
                time=chips.get("time"),
            )
            for index, logits in zip(indices, chip_logits, strict=True):
                merger.add(index, logits)
        logits = merger.merge()

        # Pixels no chip reached, as "drop" leaves the edges, are not scored.
        unpredicted = logits.isnan().any(dim=1)
        target = target.masked_fill(unpredicted, self.ignore_index)
        logits = logits.nan_to_num()

        if target.ne(self.ignore_index).any():
            loss = self.criterion(logits, target)
            self.test_metrics.update(logits, target)
        else:
            loss = logits.sum() * 0

        batch_size = len(target)
        self.log("test_loss", loss, on_epoch=True, batch_size=batch_size)
        self.log_dict(self.test_metrics, on_epoch=True, batch_size=batch_size)

        return {"logits": logits, "label": target}

    def predict_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        """Return the logits of chips that may carry no label."""
        return self(
            **batch["inputs"],
            transform=batch["transform"],
            crs=batch["crs"],
            time=batch.get("time"),
        )
