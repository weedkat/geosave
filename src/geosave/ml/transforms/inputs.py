"""Apply configured transforms to named model inputs."""

from typing import cast

import torch
from torch import nn

from geosave.model.spec.model import TransformSpec

from .augmenter import ImageAugmenter


class InputTransform(nn.Module):
    """Transform named pixel inputs per instant, preserving context tensors.

    Args:
        transforms: Model spec's transforms, keyed by pixel input name.
        size: Default spatial size for constructing the transforms.

    Raises:
        ValueError: A transform changes pixel geometry.
    """

    def __init__(
        self, transforms: dict[str, list[TransformSpec]], size: tuple[int, int] | int
    ) -> None:
        super().__init__()
        self.transforms = nn.ModuleDict(
            {
                name: ImageAugmenter(
                    [step.model_dump() for step in steps], size, photometric=True
                )
                for name, steps in transforms.items()
            }
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return transformed inputs without inferring validity or repairing values.

        Args:
            inputs: Batched named tensors; pixels end in `(band, y, x)`.

        Returns:
            Named tensors with configured pixel transforms applied. Other
            tensors pass through; the original mapping is unchanged.
        """
        result = dict(inputs)
        for name, transform in self.transforms.items():
            image = inputs[name]
            # Band statistics apply separately to each instant of each sample.
            leading = image.shape[:-3]
            transformed = cast(torch.Tensor, transform(image.flatten(0, -4)))
            result[name] = transformed.unflatten(0, leading)
        return result
