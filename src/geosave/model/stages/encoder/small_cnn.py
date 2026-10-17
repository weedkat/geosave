from __future__ import annotations

import torch
import torch.nn as nn

from geosave.model.chain import Published, chain_step
from geosave.model.registry import register_model


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Halve the grid, then refine it: two 3x3 conv-BN-ReLU layers."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


@register_model("encoder", "small_cnn")
class SmallCNN(nn.Module):
    """A small untrained CNN encoder, for tests, examples and CPU baselines.

    Returns a four-level pyramid at strides 4, 8, 16 and 32, the layout the
    `unet` decoder reads.

    Args:
        in_channels: Input band count.
        width: Channels of the first level; each deeper level doubles them.
        input_size: Input spatial size the head resizes its logits to.

    Examples:
        >>> model = build_model({
        ...     "encoder": {"name": "small_cnn", "init_args": {"in_channels": 4, "input_size": 128}},
        ...     "decoder": {"name": "unet"},
        ...     "head": {"name": "segmentation", "init_args": {"num_classes": 10}},
        ... })
        >>> model(image=torch.rand(2, 4, 128, 128)).shape
        torch.Size([2, 10, 128, 128])
    """

    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]
    input_size: Published[int | tuple[int, int]]

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 16,
        input_size: int | tuple[int, int] = 224,
    ) -> None:
        super().__init__()
        self.pyramid_channels = [width * 2**level for level in range(4)]
        self.pyramid_strides = [4, 8, 16, 32]
        self.input_size = input_size
        self.stem = _block(in_channels, width // 2)
        widths = [width // 2, *self.pyramid_channels]
        self.levels = nn.ModuleList(
            _block(widths[level], widths[level + 1]) for level in range(4)
        )

    @chain_step()
    def forward_pyramid(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Encode an image into four ever coarser feature maps.

        Args:
            image: (B, in_channels, H, W) prepared pixels.

        Returns:
            Per-level (B, channels, H / stride, W / stride) features.
        """
        x = self.stem(image)  # stride 2
        pyramid = []
        for level in self.levels:
            x = level(x)
            pyramid.append(x)
        return pyramid
