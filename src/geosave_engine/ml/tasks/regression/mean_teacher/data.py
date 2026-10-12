from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset

from geosave_engine.ml.tasks.regression import supervised
from geosave_engine.ml.transforms import ImageAugmenter

type Inputs = dict[str, torch.Tensor]


class Unlabeled(TypedDict):
    """One unlabeled chip, as the `unlabeled` dataset returns it.

    Attributes:
        inputs: Model inputs by name; pixel inputs end in ``(band, y, x)``.
        valid: ``(y, x)`` pixels whose inputs are finite in every band and instant.
    """

    inputs: Inputs
    valid: torch.Tensor


class Views(TypedDict):
    """An unlabeled batch finished as two views of the same pixels.

    Attributes:
        weak: Inputs under the labeled augmentations.
        strong: The weak view perturbed further, pixels in place.
        valid: ``(B, y, x)`` weak-view pixels no augmentation invented.
    """

    weak: Inputs
    strong: Inputs
    valid: torch.Tensor


class TrainingBatch(TypedDict):
    """One training step's batch, as the Mean Teacher DataModule finishes it."""

    labeled: supervised.Sample
    unlabeled: Views


class DataModule(supervised.DataModule):
    """Feed Mean Teacher regression: labeled chips beside unlabeled weak and strong views.

    The strong view adds photometric augmentations to the weak one, so both
    keep the same pixels in place.

    Args:
        spec: Model spec file declaring `chips`, `inputs`, and `transforms`.
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
          class_path: geosave_engine.ml.tasks.regression.mean_teacher.DataModule
          init_args:
            spec: configs/model_spec.yaml
            train: {class_path: my_project.Biomass, init_args: {split: train}}
            val: {class_path: my_project.Biomass, init_args: {split: val}}
            unlabeled: {class_path: my_project.Scenes}
            strong_augmentations:
              - {name: RandomGaussianNoise, init_args: {std: 0.1, p: 0.8}}
    """

    def __init__(
        self,
        spec: str | Path,
        train: Dataset,
        val: Dataset,
        unlabeled: Dataset,
        *,
        test: Dataset | None = None,
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
        self._strong = ImageAugmenter(strong_augmentations or [], self.spec.chips.shape)

    def train_dataloader(self) -> dict[str, DataLoader]:
        """Return shuffled labeled and unlabeled batches, read side by side."""
        return {
            "labeled": super().train_dataloader(),
            "unlabeled": self._loader(self.unlabeled_dataset, shuffle=True),
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

        # Weak view: the labeled augmentations, valid pixels moving with the image.
        inputs, valid = batch["unlabeled"]["inputs"], batch["unlabeled"]["valid"]
        images = [inputs[name] for name in self.spec.pixel_inputs]
        images, moved = self._augment(images, valid.unsqueeze(1).float())
        weak, valid = self._finish(inputs, images, moved.squeeze(1) > 0.5)

        # Strong view: photometric augmentations on top; every instant alike.
        perturbed = [
            self._strong(image.flatten(1, -3)).unflatten(1, image.shape[1:-2])
            for image in images
        ]
        strong, _ = self._finish(inputs, perturbed, valid)
        return {
            "labeled": labeled,
            "unlabeled": {"weak": weak, "strong": strong, "valid": valid},
        }
