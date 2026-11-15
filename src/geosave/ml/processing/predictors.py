"""Predict a raster larger than a model's input, chip by chip."""

from __future__ import annotations

from collections.abc import Callable

import torch
import xarray as xr

from geosave.geodata.sampling import ChipMode, MergeWindow, RasterChipper, RasterMerger
from geosave.ml.processing.roles import PREDICTORS


@PREDICTORS.register("chips")
class ChipPredictor:
    """Cut a raster into chips, predict them in batches and merge the outputs.

    A LightningModule simulates this on tensor batches with `TensorChipper`
    and `TensorMerger`, which share its layout and weights.

    Args:
        size: Chip height and width in pixels; the model's input size.
        stride: Pixels between chip starts; None takes `size`.
        mode: How chips meet the raster's edge, as `RasterChipper` takes it.
        window: Taper weighting each chip where chips overlap.
        batch_size: Chips predicted at once.

    Examples:
        >>> predictor = ChipPredictor(size=256, stride=192)
        >>> predictor.predict(forward, image).dims
        ('band', 'y', 'x')
    """

    def __init__(
        self,
        size: int,
        stride: int | None = None,
        mode: ChipMode = "reflect",
        window: MergeWindow = "hann",
        batch_size: int = 8,
    ) -> None:
        self.size = size
        self.stride = stride
        self.mode: ChipMode = mode
        self.window: MergeWindow = window
        self.batch_size = batch_size

    def predict(
        self, forward: Callable[[torch.Tensor], torch.Tensor], image: xr.Dataset
    ) -> xr.DataArray:
        """Return `forward`'s merged output over the image, reading it chip by chip.

        Args:
            forward: Maps a `(batch, band, size, size)` image batch to a
                `(batch, band, size, size)` output.
            image: Preprocessed raster, lazy.

        Returns:
            float32 `(band, y, x)` output on the image's grid.
        """
        chipper = RasterChipper(
            [xr.DataTree.from_dict({"image": image})],
            size=self.size,
            stride=self.stride,
            mode=self.mode,
        )
        merger = RasterMerger(chipper.layout, window=self.window)

        for start in range(0, len(chipper), self.batch_size):
            indexes = range(start, min(start + self.batch_size, len(chipper)))
            images = []
            for index in indexes:
                images.append(chipper[index].gs.to_tensor()["image"])

            outputs = forward(torch.stack(images)).float().cpu().numpy()

            for index, output in zip(indexes, outputs):
                merger.add(index, output)

        return merger.merge(0)
