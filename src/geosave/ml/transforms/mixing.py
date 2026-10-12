"""Draw the boxes CutMix pastes between samples."""

from __future__ import annotations

import math

import torch


def cutmix_boxes(
    batch: int,
    shape: tuple[int, int],
    *,
    p: float = 1.0,
    scale: tuple[float, float] = (0.02, 0.4),
    ratio: tuple[float, float] = (0.3, 1 / 0.3),
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Draw one random box per sample, as CutMix pastes it.

    Pasting is left to the caller, so images, labels and pseudo-labels move
    with the same box: ``torch.where(box, other, sample)``.

    Args:
        batch: Number of boxes.
        shape: Height and width of the samples.
        p: Probability a sample gets a box; the others get an empty mask.
        scale: Range of the box area, as a fraction of the sample area.
        ratio: Range of the box aspect ratio (height over width), drawn
            log-uniformly.
        device: Device of the returned masks.

    Returns:
        Boolean ``(batch, height, width)`` masks, true inside each box.

    Examples:
        >>> box = cutmix_boxes(len(image), image.shape[-2:], p=0.5)
        >>> mixed = torch.where(box[:, None], image.roll(1, 0), image)
    """
    height, width = shape
    area = torch.empty(batch, device=device).uniform_(*scale) * height * width
    log_ratio = torch.empty(batch, device=device).uniform_(
        math.log(ratio[0]), math.log(ratio[1])
    )
    aspect = log_ratio.exp()

    # Box sides from area and aspect, kept inside the sample.
    box_height = (area * aspect).sqrt().round().clamp(1, height)
    box_width = (area / aspect).sqrt().round().clamp(1, width)
    top = (torch.rand(batch, device=device) * (height - box_height + 1)).floor()
    left = (torch.rand(batch, device=device) * (width - box_width + 1)).floor()

    rows = torch.arange(height, device=device)[None, :, None]
    columns = torch.arange(width, device=device)[None, None, :]
    inside_rows = (rows >= top[:, None, None]) & (
        rows < (top + box_height)[:, None, None]
    )
    inside_columns = (columns >= left[:, None, None]) & (
        columns < (left + box_width)[:, None, None]
    )
    drawn = torch.rand(batch, device=device) < p
    return inside_rows & inside_columns & drawn[:, None, None]
