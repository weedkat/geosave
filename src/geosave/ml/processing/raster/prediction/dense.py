"""Predict a sample larger than a dense model's input, chip by chip."""

from __future__ import annotations

import torch
import xarray as xr
from torch import nn
from torch.utils.data import default_collate

from geosave.geodata.sampling import ChipMode, MergeWindow, RasterChipper, RasterMerger
from geosave.ml.chain import chain_step
from geosave.ml.processing.raster.prediction.base import PREDICTORS, RasterPredictor


@PREDICTORS.register("dense")
class DensePredictor(RasterPredictor):
    """Cut a sample into chips, predict them in batches and merge the outputs.

    A LightningModule simulates this on tensor batches with `TensorChipper`
    and `TensorMerger`, which share its layout and weights.

    Chips are read one batch at a time, but the merged output is held whole:
    `band * y * x` float32 values, twice while merging.
    `RasterPipeline.predict_to_zarr` bounds that by predicting chunk by chunk.

    Args:
        size: Chip height and width in pixels; the model's input size.
        stride: Pixels between chip starts; None takes `size`.
        mode: How chips meet the sample's edge, as `RasterChipper` takes it.
        window: Taper weighting each chip where chips overlap.
        batch_size: Chips predicted at once.
        context: Names in `TENSOR_CONTEXT` the model takes beside its pixel
            inputs, such as `"centroid"`; each is read from the chip's grid and time.
        nodes: Sample nodes the model takes as pixel inputs; None takes every
            node. Name them when the sample also holds a node for the
            postprocessor, such as a mask.

    Examples:
        >>> predictor = DensePredictor(size=256, stride=192)
        >>> predictor(model, sample).dims
        ('band', 'y', 'x')
    """

    def __init__(
        self,
        size: int,
        stride: int | None = None,
        mode: ChipMode = "reflect",
        window: MergeWindow = "hann",
        batch_size: int = 8,
        context: list[str] | None = None,
        nodes: list[str] | None = None,
    ) -> None:
        self.size = size
        self.stride = stride
        self.mode: ChipMode = mode
        self.window: MergeWindow = window
        self.batch_size = batch_size
        self.context = context if context is not None else []
        self.nodes = nodes

    def measure_chips(self) -> tuple[int, int]:
        """Return the chip size and the stride between chip starts, in pixels."""
        if self.stride is None:
            return self.size, self.size
        return self.size, self.stride

    @chain_step()
    def __call__(self, model: nn.Module, sample: xr.DataTree) -> xr.DataArray:
        """Return the model's merged output over the sample, reading it chip by chip.

        Each batch holds the pixel inputs, shaped `(batch, band, size, size)`,
        beside the context entries, and is passed as `model(**inputs)` on the
        model's device, without gradients.

        Args:
            model: Module returning a `(batch, band, size, size)` output.
            sample: Model-ready rasters by name, lazy, on one grid.

        Returns:
            `output`: the model's values as it returned them, float32
            `(band, y, x)` on the sample's grid, one band per output channel
            (logits for segmentation). Overlapping chips are averaged by
            `window`; a pixel no chip covers is NaN. It carries the grid's
            coordinates only: no mask is applied and no scalar coordinate of
            the sample, such as `time`, is copied.

        Raises:
            KeyError: The sample lacks one of `nodes`.
        """
        pixels = sample
        if self.nodes is not None:
            selected = {}
            for name in self.nodes:
                selected[name] = sample[name].to_dataset()
            pixels = xr.DataTree.from_dict(selected)

        chipper = RasterChipper(
            [pixels], size=self.size, stride=self.stride, mode=self.mode
        )
        merger = RasterMerger(chipper.layout, window=self.window)
        device = next(model.parameters()).device

        for start in range(0, len(chipper), self.batch_size):
            indexes = range(start, min(start + self.batch_size, len(chipper)))
            chips = []
            for index in indexes:
                pixels, context = chipper[index].gs.to_tensor(context=self.context)
                chips.append({**pixels, **context})

            inputs = {}
            for name, value in default_collate(chips).items():
                inputs[name] = value.to(device)
            with torch.inference_mode():
                outputs = model(**inputs)

            for index, chip_output in zip(indexes, outputs.float().cpu().numpy()):
                merger.add(index, chip_output)

        output = merger.merge(0)
        # The weighted sums are as large as the output; free them before decoding.
        merger.discard(0)
        return output
