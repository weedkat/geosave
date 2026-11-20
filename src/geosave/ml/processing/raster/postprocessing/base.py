"""The postprocessor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr


class RasterPostprocessor[Output, Product: (xr.Dataset, gpd.GeoDataFrame)](ABC):
    """Turn a predictor's output into the product a release publishes.

    `Output` is what the task's predictor returns; `Product` is a raster
    Dataset for a per-pixel task, or a GeoDataFrame for a task whose results
    are shapes, such as detection.

    Examples:
        >>> class LandCoverMap(RasterPostprocessor[xr.DataArray, xr.Dataset]):
        ...     def __call__(self, output: xr.DataArray, sample: xr.DataTree) -> xr.Dataset:
        ...         ...
    """

    @abstractmethod
    def __call__(self, output: Output, sample: xr.DataTree) -> Product:
        """Decode what the task's predictor returned over a sample.

        Decoding, masking and stating the product's metadata all happen
        here; the predictor's output arrives as the model gave it.

        Args:
            output: The predictor's output.
            sample: The sample the output was predicted over; it holds what
                the preprocessor left for this step, such as a mask, and the
                acquisition's scalar coordinates.

        Returns:
            For a per-pixel task, a raster Dataset on the output's grid, each
            variable stating its nodata, with the sample's scalar coordinates;
            for a task whose results are shapes, a GeoDataFrame.
        """


POSTPROCESSORS = Registry(
    RasterPostprocessor, imports=("geosave.ml.processing.raster.postprocessing.dense",)
)
