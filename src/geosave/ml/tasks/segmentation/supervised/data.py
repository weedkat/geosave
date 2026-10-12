from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from geosave.ml.transforms import DataKey, ImageAugmenter, cutmix_boxes
from geosave.model.spec import ModelSpec


class Sample(TypedDict):
    """One chip, as every dataset this DataModule reads returns it.

    A collated batch holds the same keys with a leading batch axis.

    Attributes:
        inputs: Model inputs by name; pixel inputs end in ``(band, y, x)``.
        target: ``(y, x)`` int64 labels, `ignore_index` where unknown.
        valid: ``(y, x)`` pixels whose inputs are finite in every band and instant.

    """

    inputs: dict[str, torch.Tensor]
    target: torch.Tensor
    valid: torch.Tensor


class DataModule(LightningDataModule):
    """Feed supervised segmentation from datasets returning a `Sample`.

    Batches are finished on the device: augmentations while training, the
    spec's transforms in every split, and NaN pixels set to zero.

    Args:
        spec: Model spec file declaring `chips`, `inputs`, and `transforms`.
        train: Training dataset.
        val: Validation dataset.
        test: Test dataset. None leaves the test split unavailable.
        augmentations: Kornia augmentations by ``name`` and ``init_args``, run
            on the pixel inputs and the target together while training.
        cutmix: Probability a training sample gets a box of its batch
            neighbour pasted in, pixels and labels alike, after augmentation.
        ignore_index: Label a target pixel takes where an augmentation
            invented it. Match the datasets' and the module's `ignore_index`.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: geosave.ml.tasks.segmentation.supervised.DataModule
          init_args:
            spec: configs/model_spec.yaml
            train: {class_path: my_project.Plantations, init_args: {split: train}}
            val: {class_path: my_project.Plantations, init_args: {split: val}}
            augmentations:
              - {name: RandomHorizontalFlip, init_args: {p: 0.5}}
    """

    def __init__(
        self,
        spec: str | Path,
        train: Dataset,
        val: Dataset,
        *,
        test: Dataset | None = None,
        augmentations: list[dict[str, Any]] | None = None,
        cutmix: float = 0.0,
        ignore_index: int = 255,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.spec = ModelSpec.load(spec)
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.cutmix = cutmix
        self.ignore_index = ignore_index
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
            The batch with its pixel inputs augmented and mixed while
            training, transformed, and free of NaN. Context inputs pass
            through. Validity is intersected with transformed finite pixels;
            while training it does not follow the augmented geometry and the
            training module ignores it.
        """
        inputs, target = batch["inputs"], batch["target"]
        images = [inputs[name] for name in self.spec.pixel_inputs]
        if self.trainer is not None and self.trainer.training:
            # Labels shift up by one, leaving zero for pixels an augmentation invents.
            images, shifted = self._augment(images, (target + 1).float())
            target = shifted.round().long() - 1
            target[target < 0] = self.ignore_index
            if self.cutmix:
                images, target = self._mix(images, target)

        inputs, valid = self._finish(inputs, images, batch["valid"])
        return {"inputs": inputs, "target": target, "valid": valid}

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
