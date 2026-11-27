"""Value rasters from per-variable chip outputs."""

import numpy as np
import xarray as xr
from upath import UPath

from geosave.geodata.sampling import MergeWindow, RasterMerger
from geosave.ml.pipeline.prediction import PredictOutput

from .base import PostprocessOutput, Postprocessor, stack_windows


class RegressionPostprocessor(Postprocessor[np.ndarray, xr.Dataset]):
    """Merge per-variable chip values into one raster variable each.

    Overlapping chips are averaged with the window's weights. A pixel no chip
    reaches stays NaN. Each window's values are held in memory as float32,
    or, given a store, accumulated in `<store>/<window index>.zarr`.

    Args:
        variables: Names of the predicted quantities, one per output band, in
            the order the model's regression head returns them.
        window: Taper weighting overlapping chips; `"boxcar"` averages evenly.

    Examples:
        >>> postprocessor = RegressionPostprocessor(["biomass"], window="hann")
        >>> values = postprocessor(predictor(preprocessed)).product
        >>> list(values.data_vars)
        ['biomass']
    """

    def __init__(self, variables: list[str], *, window: MergeWindow = "boxcar") -> None:
        self.variables = variables
        self.window: MergeWindow = window

    def __call__(
        self, predicted: PredictOutput[np.ndarray], store: UPath | None = None
    ) -> PostprocessOutput[xr.Dataset]:
        """Return each window's values on its grid, along `time`.

        Args:
            predicted: Each chip's location with its `(variable, y, x)` values.
            store: Folder to stage the merged values in; None holds them in
                memory.

        Returns:
            One float32 variable per name in `variables`, dated by each
            window as `stack_windows` dates it.

        Raises:
            ValueError: The outputs hold another number of bands than `variables`.
        """
        merger = RasterMerger(predicted.chips, window=self.window, store=store)
        for location, output in predicted.outputs:
            merger.add(location, output)

        products = []
        timespans = []
        for values, source in zip(merger.merge(), predicted.chips.sources, strict=True):
            products.append(self.split(values))
            timespans.append(source.timespan)
        return PostprocessOutput(stack_windows(products, timespans))

    def split(self, merged: xr.DataArray) -> xr.Dataset:
        """Return one window's merged bands as one variable per name.

        Raises:
            ValueError: The bands do not match `variables`.
        """
        if merged.sizes["band"] != len(self.variables):
            raise ValueError(
                f"the model returned {merged.sizes['band']} bands for "
                f"{len(self.variables)} variables {self.variables}"
            )
        values = {}
        for index, name in enumerate(self.variables):
            values[name] = merged.isel(band=index, drop=True)
        return xr.Dataset(values)
