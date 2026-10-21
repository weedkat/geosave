from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from geosave.ml.transforms import DataKey, ImageAugmenter, InputTransform
from geosave.model.spec import ModelSpec


class Sample(TypedDict):
    """One unlabeled chip, as every dataset this DataModule reads returns it.

    A collated batch holds the same keys with a leading batch axis.

    Attributes:
        inputs: Named model tensors without a batch axis. Pixel inputs end in
            `(band, y, x)`; temporal axes and separate context tensors follow
            the selected model's contract. Reconstruction targets are derived
            from the selected pixel input by the Module.
    """

    inputs: dict[str, torch.Tensor]


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
    """Feed masked image modelling from unlabeled datasets returning a `Sample`.

    Batches are finished on the device: augmentations while training, the
    spec's transforms in every split, without changing input availability.

    Args:
        spec: Model spec, or its file, declaring `chips`, `inputs`, and
            `transforms`.
        train: Training dataset.
        val: Validation dataset.
        augmentations: Kornia augmentations by ``name`` and ``init_args``,
            run jointly on pixel inputs while
            training.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: modules.pretraining.mim.DataModule
          init_args:
            spec: configs/pretraining-mim/model_spec.yaml
            train: {class_path: my_project.Scenes, init_args: {split: train}}
            val: {class_path: my_project.Scenes, init_args: {split: val}}
    """

    def __init__(
        self,
        spec: str | Path | ModelSpec,
        train: SampleDataset,
        val: SampleDataset,
        *,
        augmentations: list[dict[str, Any]] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.spec = spec if isinstance(spec, ModelSpec) else ModelSpec.load(spec)
        self.train_dataset = train
        self.val_dataset = val
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
            DataLoader collating unlabeled `Sample` dictionaries with a
            leading batch axis on inputs.
        """
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Sample]:
        """Return validation batches in chip order.

        Returns:
            DataLoader collating unlabeled `Sample` dictionaries without
            shuffling.
        """
        return self._loader(self.val_dataset, shuffle=False)

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
        if self.trainer is not None and self.trainer.training:
            keys: list[DataKey] = ["input"] * len(names)
            images = self.augmenter(*(inputs[name] for name in names), data_keys=keys)
            tensors = (images,) if isinstance(images, torch.Tensor) else images
            inputs.update(zip(names, tensors, strict=True))
        return {"inputs": self.input_transform(inputs)}
