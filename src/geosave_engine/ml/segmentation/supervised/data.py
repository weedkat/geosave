from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import xarray as xr
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from geosave_engine.geodata import cuts, open_vector
from geosave_engine.ml.inputs import model_inputs
from geosave_engine.ml.transforms import DataKey, ImageAugmenter, cutmix_boxes
from geosave_engine.model.spec import ModelSpec

if TYPE_CHECKING:
    import geopandas as gpd

type Batch = tuple[dict[str, torch.Tensor], torch.Tensor, Sequence[int], torch.Tensor]


def _read_target(data: xr.Dataset, ignore_index: int) -> torch.Tensor:
    """Read categorical labels using one nodata-to-ignore conversion."""
    target = data.gs.to_tensor().squeeze(0)
    if target.is_floating_point():
        target = torch.where(torch.isfinite(target), target, ignore_index)
    return target.long()


class Dataset(
    TorchDataset[tuple[dict[str, torch.Tensor], torch.Tensor, int, torch.Tensor]]
):
    """Read a prepared catalog's samples as model inputs and targets, one chip each.

    One window per target item, frames along time, then chips in space; every
    chip of every frame is one sample.

    Args:
        catalog: Prepared item table.
        spec: Model spec declaring `chips`, `inputs`, and optionally `frames`.
        target: Collection holding the labels, read raw.
        ignore_index: Label a target pixel takes where its raster has none.

    Examples:
        >>> dataset = Dataset(open_vector("data/train.parquet"), spec)
        >>> model_inputs, target, index, valid = dataset[0]
        >>> model_inputs["image"].shape, target.shape
        (torch.Size([4, 224, 224]), torch.Size([224, 224]))
    """

    def __init__(
        self,
        catalog: gpd.GeoDataFrame,
        spec: ModelSpec,
        *,
        target: str = "label",
        ignore_index: int = 255,
    ) -> None:
        assert spec.chips is not None, "the model spec declares no chips"
        self.spec = spec
        self.target = target
        self.ignore_index = ignore_index

        # Cut from the table alone: a window per label, its frames, then chips.
        windows = cuts.stacks(catalog, on=target)
        self.frames = windows if spec.frames is None else spec.frames.cut(windows)
        self.windows = spec.chips.cut(self.frames)

    def __len__(self) -> int:
        """Count the chips of every frame of every sample."""
        return len(self.windows)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, torch.Tensor]:
        """Read the chip this number names.

        Args:
            index: Chip number in `range(len(self))`, its row in `windows`.

        Returns:
            ``(model_inputs, target, index, valid)``: the inputs the spec
            declares, int64 labels shaped ``(y, x)``, and the ``(y, x)``
            pixels whose inputs are finite in every band and instant.
        """
        rasters = self.windows.iloc[index].gs.to_stack().gs.rasters
        inputs = model_inputs(self.spec, rasters)
        target = _read_target(rasters[self.target], self.ignore_index)

        valid = torch.ones(target.shape, dtype=torch.bool)
        for name in self.spec.pixel_inputs:
            pixels = inputs[name]
            valid &= torch.isfinite(pixels).reshape(-1, *target.shape).all(dim=0)
        return inputs, target, index, valid


class DataModule(LightningDataModule):
    """Feed supervised segmentation from prepared catalogs and one model spec.

    Batches are finished on the device: augmentations while training, the
    spec's transforms in every split, and NaN pixels set to zero.

    Args:
        spec: Model spec file declaring `chips`, `inputs`, and `transforms`.
        train: Prepared training item table (`.parquet`).
        val: Prepared validation item table (`.parquet`).
        test: Prepared test item table (`.parquet`). None leaves the test split unavailable.
        target: Layer holding the labels.
        augmentations: Kornia augmentations by ``name`` and ``init_args``, run
            on the pixel inputs and the target together while training.
        cutmix: Probability a training sample gets a box of its batch
            neighbour pasted in, pixels and labels alike, after augmentation.
        ignore_index: Label a target pixel takes where its raster has none or
            an augmentation invented it. Match the module's `ignore_index`.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: geosave_engine.ml.segmentation.supervised.DataModule
          init_args:
            spec: configs/model_spec.yaml
            train: data/train
            val: data/val
            augmentations:
              - {name: RandomHorizontalFlip, init_args: {p: 0.5}}
    """

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(
        self,
        spec: str | Path,
        train: str | Path,
        val: str | Path,
        *,
        test: str | Path | None = None,
        target: str = "label",
        augmentations: list[dict[str, Any]] | None = None,
        cutmix: float = 0.0,
        ignore_index: int = 255,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.spec = ModelSpec.load(spec)
        self.catalogs = {"train": train, "val": val, "test": test}
        self.target = target
        self.augmentations = augmentations or []
        self.cutmix = cutmix
        self.ignore_index = ignore_index
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _dataset(self, split: str) -> Dataset:
        """Read one split's catalog as a dataset."""
        folder = self.catalogs[split]
        if folder is None:
            raise ValueError(f"no {split} catalog; pass {split}= to the datamodule")
        return Dataset(
            open_vector(folder),
            self.spec,
            target=self.target,
            ignore_index=self.ignore_index,
        )

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets the stage reads and the batch pipelines.

        Raises:
            ValueError: The stage needs a catalog that was not given.
        """
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train")
        if stage in (None, "fit", "validate"):
            self.val_dataset = self._dataset("val")
        if stage == "test":
            self.test_dataset = self._dataset("test")
        assert self.spec.chips is not None
        size = self.spec.chips.shape
        self._augmenter = ImageAugmenter(self.augmentations, size)
        self._transforms = {
            name: ImageAugmenter([step.model_dump() for step in steps], size)
            for name, steps in self.spec.transforms.items()
        }

    def _loader(self, dataset: TorchDataset, *, shuffle: bool) -> DataLoader:
        # A forked worker inherits open rasters and reader threads it cannot use.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            multiprocessing_context="forkserver" if self.num_workers else None,
            persistent_workers=bool(self.num_workers),
        )

    def train_dataloader(self) -> Any:
        """Return shuffled training batches."""
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Any:
        """Return validation batches in chip order."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Any:
        """Return test batches in chip order."""
        return self._loader(self.test_dataset, shuffle=False)

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        """Finish a batch on its device.

        Args:
            batch: ``(model_inputs, target, ids, valid)`` as the loaders collate it.
            dataloader_idx: Lightning loader number.

        Returns:
            The batch with its pixel inputs augmented and mixed while
            training, transformed, and free of NaN. IDs and context inputs
            pass through. Validity is intersected with transformed finite
            pixels; while training it does not follow the augmented geometry
            and the training module ignores it.
        """
        model_inputs, target, ids, valid = batch
        images = [model_inputs[name] for name in self.spec.pixel_inputs]
        if self.trainer is not None and self.trainer.training:
            # Labels shift up by one, leaving zero for pixels an augmentation invents.
            images, shifted = self._augment(images, (target + 1).float())
            target = shifted.round().long() - 1
            target[target < 0] = self.ignore_index
            if self.cutmix:
                images, target = self._mix(images, target)

        model_inputs, valid = self._finish(model_inputs, images, valid)
        return model_inputs, target, ids, valid

    def _augment(
        self, images: list[torch.Tensor], mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the augmentations on every pixel input and a mask together.

        Args:
            images: Pixel inputs, ``(B, ..., band, y, x)``.
            mask: ``(B, y, x)`` values that move with nearest-neighbour
                sampling; pixels an augmentation invents read zero.

        Returns:
            The moved images and mask, in the shapes given.
        """
        # Time joins the bands, so every instant of a sample moves the same way.
        shapes = [image.shape[1:-2] for image in images]
        keys: list[DataKey] = ["input" for _ in images]
        keys.append("mask")
        *images, mask = self._augmenter(
            *(image.flatten(1, -3) for image in images),
            mask.unsqueeze(1),
            data_keys=keys,
        )
        unflattened = [
            image.unflatten(1, shape)
            for image, shape in zip(images, shapes, strict=True)
        ]
        return unflattened, mask.squeeze(1)

    def _finish(
        self,
        model_inputs: dict[str, torch.Tensor],
        images: list[torch.Tensor],
        valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Transform the pixel inputs and set their NaN pixels to zero.

        Args:
            model_inputs: Batch inputs; context inputs pass through.
            images: Pixel inputs in `spec.pixel_inputs` order, replacing theirs.
            valid: ``(B, y, x)`` pixels with real inputs.

        Returns:
            The finished inputs, and `valid` without pixels a transform left
            non-finite in any band or instant.
        """
        finished = dict(model_inputs)
        for name, image in zip(self.spec.pixel_inputs, images, strict=True):
            if name in self._transforms:
                # Per-band statistics apply to each instant, so time joins the batch.
                leading = image.shape[:-3]
                image = self._transforms[name](image.flatten(0, -4)).unflatten(
                    0, leading
                )
            valid = valid & torch.isfinite(image).reshape(
                image.shape[0], -1, *image.shape[-2:]
            ).all(dim=1)
            finished[name] = torch.nan_to_num(image, nan=0.0)
        return finished, valid

    def _mix(
        self, images: list[torch.Tensor], target: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Paste a box of the previous sample in the batch into each sample."""
        height, width = target.shape[-2:]
        box = cutmix_boxes(
            len(target), (height, width), p=self.cutmix, device=target.device
        )
        pasted = []
        for image in images:
            # One box spans every instant and band of a sample.
            spread = box.reshape(len(box), *([1] * (image.ndim - 3)), height, width)
            pasted.append(torch.where(spread, image.roll(1, 0), image))
        return pasted, torch.where(box, target.roll(1, 0), target)
