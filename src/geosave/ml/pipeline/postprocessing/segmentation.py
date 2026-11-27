"""Class maps from per-class chip scores."""

import numpy as np
import xarray as xr
from upath import UPath

from geosave.geodata.sampling import MergeWindow, RasterMerger
from geosave.ml.pipeline.prediction import PredictOutput

from .base import PostprocessOutput, Postprocessor, stack_windows


class SegmentationPostprocessor(Postprocessor[np.ndarray, xr.Dataset]):
    """Merge per-class chip scores and keep each pixel's highest-scoring class.

    Overlapping chips are averaged with the window's weights before the class
    is chosen. A pixel no chip reaches takes `fill`. Each window's scores are
    held in memory as float32, or, given a store, accumulated in
    `<store>/<window index>.zarr`.

    Args:
        window: Taper weighting overlapping chips; `"boxcar"` averages evenly.
        fill: Class index of pixels without a prediction, written as nodata.
        name: Variable name of the class map.

    Examples:
        >>> postprocessor = SegmentationPostprocessor(window="hann")
        >>> classes = postprocessor(predictor(preprocessed)).product
        >>> classes.class_id.dims
        ('time', 'y', 'x')
    """

    def __init__(
        self, *, window: MergeWindow = "boxcar", fill: int = -1, name: str = "class_id"
    ) -> None:
        self.window: MergeWindow = window
        self.fill = fill
        self.name = name

    def __call__(
        self, predicted: PredictOutput[np.ndarray], store: UPath | None = None
    ) -> PostprocessOutput[xr.Dataset]:
        """Return each window's class map on its grid, along `time`.

        Args:
            predicted: Each chip's location with its `(class, y, x)` scores.
            store: Folder to stage the merged scores in; None holds them in
                memory.

        Returns:
            One int32 variable of class indices, with `fill` as its nodata,
            dated by each window as `stack_windows` dates it.
        """
        merger = RasterMerger(predicted.chips, window=self.window, store=store)
        for location, output in predicted.outputs:
            merger.add(location, output)

        products = []
        timespans = []
        for scores, source in zip(merger.merge(), predicted.chips.sources, strict=True):
            products.append(self.classify(scores))
            timespans.append(source.timespan)
        return PostprocessOutput(stack_windows(products, timespans))

    def classify(self, scores: xr.DataArray) -> xr.Dataset:
        """Return one window's class map from its merged `(class, y, x)` scores."""
        # The merger leaves NaN, never infinity, where no chip predicted.
        valid = scores.notnull().all("band")
        classes = scores.argmax("band", skipna=False).where(valid, self.fill)
        classes = classes.astype("int32").gs.write_nodata(self.fill)
        return classes.to_dataset(name=self.name)
