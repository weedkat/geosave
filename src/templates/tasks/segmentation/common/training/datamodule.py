from __future__ import annotations

from typing import Any, Protocol, TypedDict, cast, runtime_checkable

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from geosave.ml.torch.training.transforms import CutMix, DataKey, ImageAugmenter


class InputSample(TypedDict):
    """Named tensor arguments for one model call."""

    inputs: dict[str, torch.Tensor]


class Sample(InputSample):
    """One labeled chip or scene. Collation adds a batch axis to each tensor.

    Attributes:
        inputs: Named pixel layers, such as `image`, and any context tensors
            the model uses. Pixel tensors end in `(band, y, x)`.
        target: Class IDs on the pixel grid, an int64 tensor shaped `(y, x)`
            with `ignore_index` wherever a label is unknown.
    """

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

    Training augments all configured input layers and their targets together.
    Datasets provide prepared inputs for every split.

    Args:
        train: Training dataset.
        val: Optional validation dataset.
        test: Test dataset. None leaves the test split unavailable.
        layers: Pixel names in `sample["inputs"]` to augment with the target.
            Defaults to `["image"]`.
        augmentations: Kornia augmentations by ``name`` and ``init_args``, run
            on every layer and the target together while training.
        cutmix: Probability a training sample gets a box of its batch
            neighbour pasted in, pixels and labels alike, after augmentation.
        ignore_index: Label a target pixel takes where an augmentation
            invented it. Match the datasets' and the module's `ignore_index`.
        batch_size: Training samples per batch.
        eval_batch_size: Validation and test samples per batch. Default collation
            requires equal spatial shapes within a batch.
        num_workers: DataLoader worker processes.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: training.datamodule.DataModule
          init_args:
            train: {class_path: my_project.Plantations, init_args: {split: train}}
            val: {class_path: my_project.Plantations, init_args: {split: val}}
            augmentations:
              - {name: RandomHorizontalFlip, init_args: {p: 0.5}}
    """

    def __init__(
        self,
        train: SampleDataset,
        val: SampleDataset | None = None,
        *,
        test: SampleDataset | None = None,
        layers: list[str] | None = None,
        augmentations: list[dict[str, Any]] | None = None,
        cutmix: float = 0.0,
        ignore_index: int = 255,
        batch_size: int = 8,
        eval_batch_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.layers = layers or ["image"]
        self.cutmix = CutMix(cutmix)
        self.ignore_index = ignore_index
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.augmenter = ImageAugmenter(augmentations or [])

    def _loader(
        self, dataset: SampleDataset, *, batch_size: int, shuffle: bool
    ) -> DataLoader[Sample]:
        # A forked worker inherits open rasters and reader threads it cannot use.
        return DataLoader(
            cast(Dataset[Sample], dataset),
            batch_size=batch_size,
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
        return self._loader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader[Sample]:
        """Return validation scenes in order.

        Returns:
            DataLoader collating `Sample` dictionaries without shuffling.
        """
        if self.val_dataset is None:
            raise ValueError("pass val= to use a validation dataset")
        return self._loader(
            self.val_dataset, batch_size=self.eval_batch_size, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Sample]:
        """Return test scenes in order.

        Returns:
            DataLoader collating test `Sample` dictionaries without shuffling.

        Raises:
            ValueError: No test dataset was given.
        """
        if self.test_dataset is None:
            raise ValueError("no test dataset; pass test= to the datamodule")
        return self._loader(
            self.test_dataset, batch_size=self.eval_batch_size, shuffle=False
        )

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Augment every pixel layer with the target while training.

        Args:
            batch: Collated samples.
            dataloader_idx: Lightning loader number.

        Returns:
            Samples with jointly transformed pixel layers and labels. Other
            inputs retain the context of the original observation.
        """
        if self.trainer is None or not self.trainer.training:
            return batch

        inputs = batch["inputs"].copy()
        target = batch["target"]
        # Reserve zero for unknown labels and the padding added by spatial transforms.
        shifted = (target + 1).masked_fill(target == self.ignore_index, 0).float()
        keys: list[DataKey] = ["input" for _ in self.layers]
        keys.append("mask")
        *images, shifted = self.augmenter(
            *(inputs[name] for name in self.layers), shifted, data_keys=keys
        )
        target = shifted.round().long() - 1
        target = target.masked_fill(target < 0, self.ignore_index)
        *images, target = self.cutmix(*images, target)
        inputs.update(zip(self.layers, images, strict=True))
        return {"inputs": inputs, "target": target}
