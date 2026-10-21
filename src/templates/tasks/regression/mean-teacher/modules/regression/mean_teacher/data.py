from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

import torch
from torch.utils.data import DataLoader, Dataset

from . import supervised
from geosave.ml.transforms import DataKey, ImageAugmenter
from geosave.model.spec import ModelSpec

type Inputs = dict[str, torch.Tensor]


class Unlabeled(TypedDict):
    """One unlabeled chip, as the `unlabeled` dataset returns it.

    Attributes:
        inputs: Named model tensor inputs for one sample, without a batch
            axis. Pixels end in `(band, y, x)`; leading temporal axes and
            separate context tensors follow the selected model's contract.
    """

    inputs: Inputs


class Views(TypedDict):
    """An unlabeled batch finished as two views of the same pixels.

    Attributes:
        weak: Batched named tensors after the labeled augmentations and spec
            transforms. Pixel inputs end in `(band, y, x)` with a batch axis.
        strong: Batched named tensors with additional photometric perturbations
            of the same pixel geometry. Context tensors pass through unchanged.
    """

    weak: Inputs
    strong: Inputs


class TrainingBatch(TypedDict):
    """One training step's batch, as the Mean Teacher DataModule finishes it.

    Attributes:
        labeled: Collated, prepared regression `Sample`, with named inputs,
            floating `(batch, variable, y, x)` targets.
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
    """Feed Mean Teacher regression: labeled chips beside unlabeled weak and strong views.

    The strong view adds photometric augmentations to the weak one, so both
    keep the same pixels in place.

    Args:
        spec: Model spec, or its file, declaring `chips`, `inputs`, and
            `transforms`.
        train: Labeled training dataset returning a `supervised.Sample`.
        val: Validation dataset returning a `supervised.Sample`.
        unlabeled: Unlabeled dataset returning an `Unlabeled` chip.
        test: Test dataset. None leaves the test split unavailable.
        augmentations: Kornia augmentations run on labeled and unlabeled chips.
        strong_augmentations: Kornia photometric augmentations of the strong
            view, such as ``RandomGaussianNoise``.
        batch_size: Samples per batch, labeled and unlabeled alike.
        num_workers: DataLoader worker processes per loader.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: modules.regression.mean_teacher.DataModule
          init_args:
            spec: configs/regression-mean-teacher/model_spec.yaml
            train: {class_path: my_project.Biomass, init_args: {split: train}}
            val: {class_path: my_project.Biomass, init_args: {split: val}}
            unlabeled: {class_path: my_project.Scenes}
            strong_augmentations:
              - {name: RandomGaussianNoise, init_args: {std: 0.1, p: 0.8}}
    """

    def __init__(
        self,
        spec: str | Path | ModelSpec,
        train: supervised.SampleDataset,
        val: supervised.SampleDataset,
        unlabeled: UnlabeledDataset,
        *,
        test: supervised.SampleDataset | None = None,
        augmentations: list[dict[str, Any]] | None = None,
        strong_augmentations: list[dict[str, Any]] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__(
            spec,
            train,
            val,
            test=test,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.unlabeled_dataset = unlabeled
        assert self.spec.chips is not None
        self.strong_augmenter = ImageAugmenter(
            strong_augmentations or [], self.spec.chips.shape, photometric=True
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
        names = self.spec.pixel_inputs
        keys: list[DataKey] = ["input"] * len(names)

        # Draw shared spatial augmentation once before splitting the views.
        images = self.augmenter(*(inputs[name] for name in names), data_keys=keys)
        tensors = (images,) if isinstance(images, torch.Tensor) else images
        inputs.update(zip(names, tensors, strict=True))
        weak = self.input_transform(inputs)

        images = self.strong_augmenter(
            *(inputs[name] for name in names), data_keys=keys
        )
        tensors = (images,) if isinstance(images, torch.Tensor) else images
        inputs.update(zip(names, tensors, strict=True))
        strong = self.input_transform(inputs)
        return {"labeled": labeled, "unlabeled": {"weak": weak, "strong": strong}}
