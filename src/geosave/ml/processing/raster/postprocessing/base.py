"""The postprocessor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import xarray as xr


class RasterPostprocessor(ABC):
    """Turn a model's output into the product a release publishes.

    A subclass declares its `__call__` with `@chain_step()`, takes the
    predictor's output under the name and type the predictor returns it by,
    and returns a variable named `product`.
    """

    @abstractmethod
    def __call__(self, output: Any, sample: xr.DataTree) -> Any:
        """Decode what the task's predictor returned over a sample.

        Decoding, masking and stating the product's metadata all happen
        here; the predictor's output arrives as the model gave it.

        Args:
            output: The predictor's output: float32 `(band, y, x)` model
                values for a per-pixel task.
            sample: The sample the output was predicted over; it holds what
                the preprocessor left for this step, such as a mask, and the
                acquisition's scalar coordinates.

        Returns:
            `product`: a raster Dataset on the sample's grid for a per-pixel
            task, each variable stating its nodata, with the sample's scalar
            coordinates; a task whose results are shapes, such as detection,
            returns a GeoDataFrame.
        """


POSTPROCESSORS = Registry(
    RasterPostprocessor, imports=("geosave.ml.processing.raster.postprocessing.dense",)
)
