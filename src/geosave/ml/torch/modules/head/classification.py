from __future__ import annotations

import torch
import torch.nn as nn

from geosave.ml.torch.chain.registry import MODULES
from geosave.ml.torch.typing import FeatureMap


@MODULES.register("classification")
class ClassificationHead(nn.Module):
    """Map an encoder's deepest features to one class logit vector per sample.

    Averages the deepest pyramid level over its spatial axes, then projects it
    linearly.

    Args:
        num_classes: Number of classes. Class names belong to the labels,
            not to the model.
        pyramid_channels: Channel width of each pyramid level. Wired from the
            encoder during chain construction.
        dropout: Dropout probability before the projection.

    Raises:
        ValueError: `num_classes` is below one.

    Examples:
        >>> head = ClassificationHead(num_classes=2, pyramid_channels=[768] * 4)
        >>> head.num_classes
        2
    """

    returns = ("logits",)

    def __init__(
        self,
        num_classes: int,
        pyramid_channels: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(
                f"a classification head needs at least one class, got {num_classes}"
            )
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(pyramid_channels[-1], num_classes)

    def forward(self, pyramid: list[FeatureMap]) -> torch.Tensor:
        """Project the pooled deepest level to class logits.

        Args:
            pyramid: Per-level (B, C, H, W) features from the encoder.

        Returns:
            (B, num_classes) logits tensor.
        """
        pooled = pyramid[-1].flatten(2).mean(dim=-1)
        return self.projection(self.dropout(pooled))
