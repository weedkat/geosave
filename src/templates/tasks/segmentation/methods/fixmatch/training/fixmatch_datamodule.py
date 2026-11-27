from __future__ import annotations

from typing import Any, Protocol, TypedDict, cast, runtime_checkable

import torch
from torch.utils.data import DataLoader, Dataset

from . import datamodule as supervised
from .datamodule import InputSample as Unlabeled
from geosave.ml.torch.training.transforms import DataKey, ImageAugmenter

type Inputs = dict[str, torch.Tensor]


class Views(TypedDict):
    """An unlabeled batch finished as two views of the same pixels.

    Attributes:
        weak: Batched named tensors after the shared spatial augmentation. Pixel inputs end in `(band, y, x)` with a batch axis.
        strong: Batched named tensors with additional photometric perturbations
            of the same pixel geometry. Context tensors pass through unchanged.
    """

    weak: Inputs
    strong: Inputs


class TrainingBatch(TypedDict):
    """One training step's batch, as the FixMatch DataModule finishes it.

    Attributes:
        labeled: Collated, prepared segmentation `Sample`, with named inputs,
            int64 `(batch, y, x)` targets.
        unlabeled: `Views` with weak/strong named tensor inputs on matching
            pixel geometry.
    """

    labeled: supervised.Sample
    unlabeled: Views


@runtime_checkable
class UnlabeledDataset(Protocol):
    """A map-style dataset returning model-ready unlabeled inputs."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, index: int) -> Unlabeled:
        """Return named tensors satisfying the selected model's contract."""
        ...


class DataModule(supervised.DataModule):
    """Feed weak-to-strong semi-supervised segmentation.

    An unlabeled chip gives a weak view and a strong view perturbed further;
    strong augmentations must keep geometry, so weak-view pseudo-labels fit.

    Args:
        layers: Names of the pixel tensors augmented together.
        train: Labeled training dataset returning a `supervised.Sample`.
        val: Validation dataset returning a `supervised.Sample`.
        unlabeled: Unlabeled dataset returning an `Unlabeled` chip.
        test: Test dataset. None leaves the test split unavailable.
        augmentations: Kornia augmentations run on labeled batches with their
            target and on unlabeled pixel inputs.
        strong_augmentations: Kornia photometric augmentations drawn anew for
            the strong view, such as ``RandomGaussianNoise``.
        cutmix: Probability a labeled sample gets a box of its batch neighbour.
        ignore_index: Label of pixels an augmentation invented.
        batch_size: Samples per batch, labeled and unlabeled alike.
        num_workers: DataLoader worker processes per loader.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: training.fixmatch_datamodule.DataModule
          init_args:
            train: {class_path: my_project.Plantations, init_args: {split: train}}
            val: {class_path: my_project.Plantations, init_args: {split: val}}
            unlabeled: {class_path: my_project.Scenes}
            strong_augmentations:
              - {name: RandomGaussianNoise, init_args: {std: 0.1, p: 0.8}}
    """

    def __init__(
        self,
        train: supervised.SampleDataset,
        val: supervised.SampleDataset | None = None,
        *,
        unlabeled: UnlabeledDataset,
        test: supervised.SampleDataset | None = None,
        augmentations: list[dict[str, Any]] | None = None,
        strong_augmentations: list[dict[str, Any]] | None = None,
        cutmix: float = 0.0,
        ignore_index: int = 255,
        batch_size: int = 8,
        num_workers: int = 4,
        layers: list[str] | None = None,
        eval_batch_size: int = 1,
    ) -> None:
        super().__init__(
            train,
            val,
            test=test,
            augmentations=augmentations,
            cutmix=cutmix,
            ignore_index=ignore_index,
            batch_size=batch_size,
            num_workers=num_workers,
            layers=layers,
            eval_batch_size=eval_batch_size,
        )
        self.unlabeled_dataset = unlabeled
        self.strong_augmenter = ImageAugmenter(
            strong_augmentations or [], photometric=True
        )

    def train_dataloader(self) -> dict[str, DataLoader]:
        """Return shuffled labeled and unlabeled batches, read side by side.

        Returns:
            DataLoaders keyed by `labeled` and `unlabeled`. Lightning combines
            their collated samples under those keys for the training hook.
        """
        return {
            "labeled": super().train_dataloader(),
            "unlabeled": DataLoader(
                cast(Dataset[Unlabeled], self.unlabeled_dataset),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                multiprocessing_context="forkserver" if self.num_workers else None,
                persistent_workers=bool(self.num_workers),
            ),
        }

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Finish a batch on its device.

        Args:
            batch: Labeled and `Unlabeled` batches side by side while fitting,
                otherwise a collated `supervised.Sample`.
            dataloader_idx: Lightning loader number.

        Returns:
            Evaluation batches as supervised training finishes them; a
            `TrainingBatch` while fitting.
        """
        if "labeled" not in batch:
            return super().on_after_batch_transfer(batch, dataloader_idx)
        labeled = super().on_after_batch_transfer(batch["labeled"], dataloader_idx)

        inputs = dict(batch["unlabeled"]["inputs"])
        names = self.layers
        keys: list[DataKey] = ["input"] * len(names)

        # Draw shared spatial augmentation once before splitting the views.
        images = self.augmenter(*(inputs[name] for name in names), data_keys=keys)
        tensors = (images,) if isinstance(images, torch.Tensor) else images
        inputs.update(zip(names, tensors, strict=True))
        weak = inputs.copy()

        images = self.strong_augmenter(
            *(inputs[name] for name in names), data_keys=keys
        )
        tensors = (images,) if isinstance(images, torch.Tensor) else images
        inputs.update(zip(names, tensors, strict=True))
        strong = inputs
        return {"labeled": labeled, "unlabeled": {"weak": weak, "strong": strong}}
