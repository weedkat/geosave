from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from geosave.ml.transforms import CutMix, DataKey, ImageAugmenter, InputTransform
from geosave.model.spec import ModelSpec


class Sample(TypedDict):
    """One chip, as every dataset this DataModule reads returns it.

    A collated batch holds the same keys with a leading batch axis.

    Attributes:
        inputs: Named model tensors without a batch axis. Pixel inputs end in
            `(band, y, x)`; temporal axes and separate context tensors follow
            the selected model's contract.
        target: ``(y, x)`` int64 labels, `ignore_index` where unknown.

    """

    inputs: dict[str, torch.Tensor]
    target: torch.Tensor


@runtime_checkable
class SampleDataset(Protocol):
    """A map-style dataset returning this method's model-ready Sample."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, index: int) -> Sample:
        """Return one sample satisfying the selected model's input contract."""
        ...


class DataModule(LightningDataModule):
    """Feed supervised segmentation from datasets returning a `Sample`.

    Batches are finished on the device: augmentations while training, the
    spec's transforms in every split, without changing input availability.

    Args:
        spec: Model spec, or its file, declaring `chips`, `inputs`, and
            `transforms`.
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
          class_path: modules.segmentation.supervised.DataModule
          init_args:
            spec: configs/segmentation-supervised/model_spec.yaml
            train: {class_path: my_project.Plantations, init_args: {split: train}}
            val: {class_path: my_project.Plantations, init_args: {split: val}}
            augmentations:
              - {name: RandomHorizontalFlip, init_args: {p: 0.5}}
    """

    def __init__(
        self,
        spec: str | Path | ModelSpec,
        train: SampleDataset,
        val: SampleDataset,
        *,
        test: SampleDataset | None = None,
        augmentations: list[dict[str, Any]] | None = None,
        cutmix: float = 0.0,
        ignore_index: int = 255,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.spec = spec if isinstance(spec, ModelSpec) else ModelSpec.load(spec)
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.cutmix = CutMix(cutmix)
        self.ignore_index = ignore_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.spec.chips is None:
            raise ValueError("the model spec declares no chips")
        size = self.spec.chips.shape
        self.augmenter = ImageAugmenter(augmentations or [], size)
        self.input_transform = InputTransform(self.spec.transforms, size)

    def _loader(self, dataset: SampleDataset, *, shuffle: bool) -> DataLoader[Sample]:
        # A forked worker inherits open rasters and reader threads it cannot use.
        return DataLoader(
            cast(Dataset[Sample], dataset),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            multiprocessing_context="forkserver" if self.num_workers else None,
            persistent_workers=bool(self.num_workers),
        )

    def train_dataloader(self) -> Any:
        """Return shuffled training batches.

        Returns:
            DataLoader collating `Sample` dictionaries with a leading batch
            axis; targets become `(batch, y, x)`.
        """
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Sample]:
        """Return validation batches in chip order.

        Returns:
            DataLoader collating `Sample` dictionaries without shuffling.
        """
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[Sample]:
        """Return test batches in chip order.

        Returns:
            DataLoader collating test `Sample` dictionaries without shuffling.

        Raises:
            ValueError: No test dataset was given.
        """
        if self.test_dataset is None:
            raise ValueError("no test dataset; pass test= to the datamodule")
        return self._loader(self.test_dataset, shuffle=False)

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Apply training augmentation, then configured input transforms.

        Args:
            batch: Collated model-ready samples.
            dataloader_idx: Lightning loader number.

        Returns:
            Prepared samples with context inputs preserved. Input availability
            remains the Dataset/model contract; values are not repaired.
        """
        inputs = dict(batch["inputs"])
        names = self.spec.pixel_inputs
        target = batch["target"]
        if self.trainer is not None and self.trainer.training:
            # Zero is reserved for unknown labels and augmentation padding.
            shifted = (target + 1).masked_fill(target == self.ignore_index, 0).float()
            keys: list[DataKey] = ["input" for _ in names]
            keys.append("mask")
            *images, shifted = self.augmenter(
                *(inputs[name] for name in names), shifted, data_keys=keys
            )
            target = shifted.round().long() - 1
            target = target.masked_fill(target < 0, self.ignore_index)
            *images, target = self.cutmix(*images, target)
            inputs.update(zip(names, images, strict=True))
        return {"inputs": self.input_transform(inputs), "target": target}
