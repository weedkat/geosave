from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from geosave_engine.ml.transforms import DataKey, ImageAugmenter
from geosave_engine.model.spec import ModelSpec


class Sample(TypedDict):
    """One chip, as every dataset this DataModule reads returns it.

    A collated batch holds the same keys with a leading batch axis.

    Attributes:
        inputs: Model inputs by name; pixel inputs end in ``(band, y, x)``.
        target: int64 class code: the class's position in the head's `classes`.
        valid: ``(y, x)`` pixels whose inputs are finite in every band and instant.
    """

    inputs: dict[str, torch.Tensor]
    target: torch.Tensor
    valid: torch.Tensor


class DataModule(LightningDataModule):
    """Feed supervised classification from datasets returning a `Sample`.

    Batches are finished on the device: augmentations while training, the
    spec's transforms in every split, and NaN pixels set to zero.

    Args:
        spec: Model spec file declaring `chips`, `inputs`, and `transforms`.
        train: Training dataset.
        val: Validation dataset.
        test: Test dataset. None leaves the test split unavailable.
        augmentations: Kornia augmentations by ``name`` and ``init_args``, run
            on the pixel inputs while training.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: geosave_engine.ml.tasks.classification.supervised.DataModule
          init_args:
            spec: configs/model_spec.yaml
            train: {class_path: my_project.LandCover, init_args: {split: train}}
            val: {class_path: my_project.LandCover, init_args: {split: val}}
    """

    def __init__(
        self,
        spec: str | Path,
        train: Dataset,
        val: Dataset,
        *,
        test: Dataset | None = None,
        augmentations: list[dict[str, Any]] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.spec = ModelSpec.load(spec)
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert self.spec.chips is not None, "the model spec declares no chips"
        size = self.spec.chips.shape
        self._augmenter = ImageAugmenter(augmentations or [], size)
        self._transforms = {
            name: ImageAugmenter([step.model_dump() for step in steps], size)
            for name, steps in self.spec.transforms.items()
        }

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
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
        """Return test batches in chip order.

        Raises:
            ValueError: No test dataset was given.
        """
        if self.test_dataset is None:
            raise ValueError("no test dataset; pass test= to the datamodule")
        return self._loader(self.test_dataset, shuffle=False)

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Finish a batch on its device.

        Args:
            batch: A collated `Sample`.
            dataloader_idx: Lightning loader number.

        Returns:
            The batch with its pixel inputs augmented while training,
            transformed, and free of NaN; `valid` moves with the pixels.
        """
        augment = self.trainer is not None and self.trainer.training
        inputs, valid = self._finish(batch["inputs"], batch["valid"], augment=augment)
        return {"inputs": inputs, "target": batch["target"], "valid": valid}

    def _finish(
        self, model_inputs: dict[str, torch.Tensor], valid: torch.Tensor, *, augment: bool
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Augment the pixel inputs if asked, transform them, and zero their NaN."""
        names = self.spec.pixel_inputs
        images = [model_inputs[name] for name in names]

        if augment:
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
        return finished, valid
