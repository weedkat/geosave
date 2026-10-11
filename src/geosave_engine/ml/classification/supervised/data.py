from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from geosave_engine.geodata import cuts, open_vector
from geosave_engine.ml.inputs import model_inputs
from geosave_engine.ml.transforms import DataKey, ImageAugmenter
from geosave_engine.model.spec import ModelSpec

if TYPE_CHECKING:
    import geopandas as gpd

type Batch = tuple[dict[str, torch.Tensor], torch.Tensor, Sequence[int], torch.Tensor]


class Dataset(
    TorchDataset[tuple[dict[str, torch.Tensor], torch.Tensor, int, torch.Tensor]]
):
    """Read a prepared catalog's chips as model inputs and one class each.

    A chip's class is the `label` property of the item its window was cut
    from, as a class code: the class's position in the head's `classes`.

    Args:
        catalog: Prepared item table holding the spec's input rasters.
        spec: Model spec declaring `chips`, `inputs`, and optionally `frames`.
        label: Item property holding the class code.

    Examples:
        >>> dataset = Dataset(open_vector("data/train.parquet"), spec, label="class_id")
        >>> model_inputs, target, index, valid = dataset[0]
        >>> target
        tensor(3)
    """

    def __init__(
        self, catalog: gpd.GeoDataFrame, spec: ModelSpec, *, label: str = "class_id"
    ) -> None:
        assert spec.chips is not None, "the model spec declares no chips"
        self.spec = spec

        # A window per item of the first raster the inputs read, then chips.
        windows = cuts.stacks(catalog, on=spec.input_rasters[0])
        self.frames = windows if spec.frames is None else spec.frames.cut(windows)
        self.windows = spec.chips.cut(self.frames)
        self.labels = torch.as_tensor(self.windows[label].to_numpy(dtype="int64"))

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
            declares, the int64 class code, and the ``(y, x)`` pixels with
            finite inputs.
        """
        rasters = self.windows.iloc[index].gs.to_stack().gs.rasters
        inputs = model_inputs(self.spec, rasters)
        assert self.spec.chips is not None
        shape = self.spec.chips.shape
        valid = torch.ones(shape, dtype=torch.bool)
        for name in self.spec.pixel_inputs:
            valid &= torch.isfinite(inputs[name]).reshape(-1, *shape).all(dim=0)
        return inputs, self.labels[index], index, valid


class DataModule(LightningDataModule):
    """Feed supervised classification from prepared catalogs and one model spec.

    Batches are finished on the device: augmentations while training, the
    spec's transforms in every split, and NaN pixels set to zero.

    Args:
        spec: Model spec file declaring `chips`, `inputs`, and `transforms`.
        train: Prepared training item table (`.parquet`).
        val: Prepared validation item table (`.parquet`).
        test: Prepared test item table (`.parquet`). None leaves the test split unavailable.
        label: Item property holding the class code.
        augmentations: Kornia augmentations by ``name`` and ``init_args``, run
            on the pixel inputs while training.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: geosave_engine.ml.classification.supervised.DataModule
          init_args:
            spec: configs/model_spec.yaml
            train: data/train
            val: data/val
            label: class_id
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
        label: str = "class_id",
        augmentations: list[dict[str, Any]] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.spec = ModelSpec.load(spec)
        self.catalogs = {"train": train, "val": val, "test": test}
        self.label = label
        self.augmentations = augmentations or []
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _dataset(self, split: str) -> Dataset:
        """Read one split's catalog as a dataset."""
        folder = self.catalogs[split]
        if folder is None:
            raise ValueError(f"no {split} catalog; pass {split}= to the datamodule")
        return Dataset(open_vector(folder), self.spec, label=self.label)

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
            The batch with its pixel inputs augmented while training,
            transformed, and free of NaN; `valid` moves with the pixels.
        """
        model_inputs, target, ids, valid = batch
        names = self.spec.pixel_inputs
        images = [model_inputs[name] for name in names]

        if self.trainer is not None and self.trainer.training:
            # Time joins the bands, so every instant of a sample moves the same way.
            shapes = [image.shape[1:-2] for image in images]
            keys: list[DataKey] = ["input" for _ in images]
            keys.append("mask")
            *images, moved = self._augmenter(
                *(image.flatten(1, -3) for image in images),
                valid.unsqueeze(1).float(),
                data_keys=keys,
            )
            images = [
                image.unflatten(1, shape)
                for image, shape in zip(images, shapes, strict=True)
            ]
            valid = moved.squeeze(1) > 0.5

        finished = dict(model_inputs)
        for name, image in zip(names, images, strict=True):
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
        return finished, target, ids, valid
