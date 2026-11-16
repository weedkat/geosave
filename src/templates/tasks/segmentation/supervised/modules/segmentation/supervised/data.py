from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

import torch
from typing_extensions import TypedDict
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from geosave.ml.transforms import CutMix, DataKey, ImageAugmenter
from geosave.ml.factory import BuildSpec
from geosave.ml.processing.raster.preprocessing.base import PREPROCESSORS


class Sample(TypedDict, extra_items=torch.Tensor):
    """One chip or scene, as every dataset this DataModule reads returns it.

    Every entry is a tensor, named as the model chain takes it. A collated
    batch holds the same keys with a leading batch axis.

    Attributes:
        image: Pixels on the sample's grid, ending in `(band, y, x)`, with a
            time axis before `band` when the model takes one. Further pixel
            layers, such as `dem`, sit beside it under their own names.
        target: ``(y, x)`` int64 labels, `ignore_index` where unknown.
        transform: float64 affine coefficients placing the pixels, as
            `DataTree.gs.to_tensor(grid=True)` writes them.
        crs: int64 EPSG code of the grid.

    Further entries are the model's context, such as `centroid`, as
    `DataTree.gs.to_tensor(context=...)` writes them.
    """

    image: torch.Tensor
    target: torch.Tensor
    transform: torch.Tensor
    crs: torch.Tensor


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

    Batches are finished on the device: augmentations of every input and the
    target together while training, then the preprocessor's normalization of
    `image` in every split, as a release's processor normalizes it.

    Args:
        preprocessor: Preprocessor by registered `name` or `class_path`, with
            `init_args`; its `normalize` runs on every batch.
        train: Training dataset.
        val: Validation dataset.
        test: Test dataset. None leaves the test split unavailable.
        layers: Names of the pixel tensors in each sample, augmented with the
            target while training and chipped with it in evaluation. None
            takes `["image"]`.
        augmentations: Kornia augmentations by ``name`` and ``init_args``, run
            on every layer and the target together while training.
        cutmix: Probability a training sample gets a box of its batch
            neighbour pasted in, pixels and labels alike, after augmentation.
        ignore_index: Label a target pixel takes where an augmentation
            invented it. Match the datasets' and the module's `ignore_index`.
        size: Spatial size used by training augmentation.
        batch_size: Training samples per batch.
        eval_batch_size: Validation and test samples per batch. These are
            whole scenes that the Module chips; scenes of different sizes
            batch only one at a time.
        num_workers: DataLoader worker processes.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: modules.segmentation.supervised.DataModule
          init_args:
            preprocessor: {class_path: modules.segmentation.supervised.processing.Reflectance}
            train: {class_path: my_project.Plantations, init_args: {split: train}}
            val: {class_path: my_project.Plantations, init_args: {split: val}}
            augmentations:
              - {name: RandomHorizontalFlip, init_args: {p: 0.5}}
    """

    def __init__(
        self,
        preprocessor: dict[str, Any],
        train: SampleDataset,
        val: SampleDataset,
        *,
        test: SampleDataset | None = None,
        layers: list[str] | None = None,
        augmentations: list[dict[str, Any]] | None = None,
        cutmix: float = 0.0,
        ignore_index: int = 255,
        size: int | tuple[int, int] = 224,
        batch_size: int = 8,
        eval_batch_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.preprocessor = PREPROCESSORS.build(BuildSpec.model_validate(preprocessor))
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.layers = layers or ["image"]
        self.cutmix = CutMix(cutmix)
        self.ignore_index = ignore_index
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.augmenter = ImageAugmenter(augmentations or [], size)

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
        """Augment every layer with the target while training, then normalize `image`.

        Args:
            batch: Collated samples.
            dataloader_idx: Lightning loader number.

        Returns:
            Prepared samples with their grid and context entries. Values are
            not repaired.
        """
        prepared = batch.copy()
        if self.trainer is not None and self.trainer.training:
            target = batch["target"]
            # Zero is reserved for unknown labels and augmentation padding.
            shifted = (target + 1).masked_fill(target == self.ignore_index, 0).float()
            keys: list[DataKey] = ["input" for _ in self.layers]
            keys.append("mask")
            *images, shifted = self.augmenter(
                *(batch[name] for name in self.layers), shifted, data_keys=keys
            )
            target = shifted.round().long() - 1
            target = target.masked_fill(target < 0, self.ignore_index)
            *images, target = self.cutmix(*images, target)
            for name, image in zip(self.layers, images, strict=True):
                prepared[name] = image
            prepared["target"] = target
        prepared["image"] = self.preprocessor.normalize(prepared["image"])

        # Augmentation moves pixels without moving `transform`; the grid is
        # exact for validation, test and prediction batches only.
        return prepared
