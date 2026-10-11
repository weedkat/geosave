from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from geosave_engine.geodata import cuts, open_vector
from geosave_engine.ml.inputs import model_inputs
from geosave_engine.ml.segmentation import supervised
from geosave_engine.ml.transforms import ImageAugmenter
from geosave_engine.model.spec import ModelSpec

if TYPE_CHECKING:
    import geopandas as gpd

type Inputs = dict[str, torch.Tensor]


class TrainingBatch(TypedDict):
    """One training step's batch, as the FixMatch DataModule finishes it.

    Examples:
        >>> batch["labeled"]
        (model_inputs, target, ids, valid)
        >>> batch["unlabeled"]
        (weak_inputs, strong_inputs, valid)
    """

    labeled: tuple[Inputs, torch.Tensor, Sequence[int], torch.Tensor]
    unlabeled: tuple[Inputs, Inputs, torch.Tensor]


class UnlabeledDataset(TorchDataset[tuple[Inputs, int, torch.Tensor]]):
    """Read a prepared unlabeled catalog's chips as model inputs only.

    Args:
        catalog: Prepared item table holding the spec's input rasters.
        spec: Model spec declaring `chips`, `inputs`, and optionally `frames`.
    """

    def __init__(self, catalog: gpd.GeoDataFrame, spec: ModelSpec) -> None:
        assert spec.chips is not None, "the model spec declares no chips"
        self.spec = spec
        # A window per item of the first raster the inputs read.
        windows = cuts.stacks(catalog, on=spec.input_rasters[0])
        self.frames = windows if spec.frames is None else spec.frames.cut(windows)
        self.windows = spec.chips.cut(self.frames)

    def __len__(self) -> int:
        """Count the chips of every frame of every sample."""
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[Inputs, int, torch.Tensor]:
        """Return ``(model_inputs, index, valid)`` for the chip this number names."""
        rasters = self.windows.iloc[index].gs.to_stack().gs.rasters
        inputs = model_inputs(self.spec, rasters)
        assert self.spec.chips is not None
        shape = self.spec.chips.shape
        valid = torch.ones(shape, dtype=torch.bool)
        for name in self.spec.pixel_inputs:
            valid &= torch.isfinite(inputs[name]).reshape(-1, *shape).all(dim=0)
        return inputs, index, valid


class DataModule(supervised.DataModule):
    """Feed weak-to-strong semi-supervised segmentation.

    An unlabeled chip gives a weak view and a strong view perturbed further;
    strong augmentations must keep geometry, so weak-view pseudo-labels fit.

    Args:
        spec: Model spec file declaring `chips`, `inputs`, and `transforms`.
        train: Prepared labeled training item table (`.parquet`).
        val: Prepared validation item table (`.parquet`).
        unlabeled: Prepared unlabeled item table (`.parquet`); only the spec's input
            collections are read.
        test: Prepared test item table (`.parquet`). None leaves the test split unavailable.
        target: Layer holding the labels.
        augmentations: Kornia augmentations run on labeled batches with their
            target and on unlabeled batches with their valid pixels.
        strong_augmentations: Kornia photometric augmentations drawn anew for
            the strong view, such as ``RandomGaussianNoise``.
        cutmix: Probability a labeled sample gets a box of its batch neighbour.
        ignore_index: Label of pixels a raster or an augmentation lacks.
        batch_size: Samples per batch, labeled and unlabeled alike.
        num_workers: DataLoader worker processes per loader.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: geosave_engine.ml.segmentation.fixmatch.DataModule
          init_args:
            spec: configs/model_spec.yaml
            train: data/train
            val: data/val
            unlabeled: data/unlabeled
            augmentations:
              - {name: RandomHorizontalFlip, init_args: {p: 0.5}}
            strong_augmentations:
              - {name: RandomGaussianNoise, init_args: {std: 0.1, p: 0.8}}
    """

    def __init__(
        self,
        spec: str | Path,
        train: str | Path,
        val: str | Path,
        unlabeled: str | Path,
        *,
        test: str | Path | None = None,
        target: str = "label",
        augmentations: list[dict[str, Any]] | None = None,
        strong_augmentations: list[dict[str, Any]] | None = None,
        cutmix: float = 0.0,
        ignore_index: int = 255,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__(
            spec,
            train,
            val,
            test=test,
            target=target,
            augmentations=augmentations,
            cutmix=cutmix,
            ignore_index=ignore_index,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.catalogs["unlabeled"] = unlabeled
        self.strong_augmentations = strong_augmentations or []

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets the stage reads, the unlabeled one while fitting.

        Raises:
            ValueError: The stage needs a catalog that was not given.
        """
        super().setup(stage)
        if stage in (None, "fit"):
            self.unlabeled_dataset = UnlabeledDataset(
                open_vector(self.catalogs["unlabeled"]), self.spec
            )
        assert self.spec.chips is not None
        self._strong = ImageAugmenter(self.strong_augmentations, self.spec.chips.shape)

    def train_dataloader(self) -> dict[str, DataLoader]:
        """Return shuffled labeled and unlabeled batches, read side by side."""
        return {
            "labeled": super().train_dataloader(),
            "unlabeled": self._loader(self.unlabeled_dataset, shuffle=True),
        }

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Finish a batch on its device.

        Args:
            batch: A `TrainingBatch` while fitting, otherwise
                ``(model_inputs, target, ids, valid)``.
            dataloader_idx: Lightning loader number.

        Returns:
            Evaluation batches as supervised training finishes them; a
            training batch with its unlabeled part as ``(weak, strong, valid)``,
            `valid` marking weak-view pixels no augmentation invented.
        """
        if not isinstance(batch, dict):
            return super().on_after_batch_transfer(batch, dataloader_idx)
        labeled = super().on_after_batch_transfer(batch["labeled"], dataloader_idx)

        # Weak view: the labeled augmentations, valid pixels moving with the image.
        model_inputs, _, valid = batch["unlabeled"]
        images = [model_inputs[name] for name in self.spec.pixel_inputs]
        images, moved = self._augment(images, valid.float())
        weak, valid = self._finish(model_inputs, images, moved > 0.5)

        # Strong view: photometric augmentations on top; every instant alike.
        perturbed = [
            self._strong(image.flatten(1, -3)).unflatten(1, image.shape[1:-2])
            for image in images
        ]
        strong, _ = self._finish(model_inputs, perturbed, valid)
        return {"labeled": labeled, "unlabeled": (weak, strong, valid)}
