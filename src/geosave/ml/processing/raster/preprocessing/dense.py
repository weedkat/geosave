"""Prepare one raster for a dense model: select bands, decode, fill, standardize."""

import xarray as xr

from geosave.ml.chain import chain_step
from geosave.ml.processing.raster.preprocessing.base import (
    PREPROCESSORS,
    RasterPreprocessor,
)


@PREPROCESSORS.register("dense")
class DensePreprocessor(RasterPreprocessor):
    """Build one model input from some bands of one raster.

    Every step but band selection is switched by an argument, so one class
    serves stored digital numbers and rasters already in model units alike.
    A model that needs more, such as cloud masking or a second sensor, gets
    its own `RasterPreprocessor`.

    Args:
        bands: Variables of the source raster, in model order.
        decode: Apply the raster's own nodata and packing
            (`scale_factor`, `add_offset`); False takes stored values as they are.
        fill: Value given to missing pixels; None leaves them NaN.
        mean: Mean of each prepared band; None leaves the bands unstandardized.
        std: Standard deviation of each prepared band.
        name: Node the raster is read from, and the model input it becomes.

    Raises:
        ValueError: Only one of `mean` and `std` is given.

    Examples:
        >>> preprocessor = DensePreprocessor(bands=["B04", "B08"])
        >>> preprocessor(xr.DataTree.from_dict({"image": raster}))
        <xarray.DataTree> ...
    """

    def __init__(
        self,
        bands: list[str],
        decode: bool = True,
        fill: float | None = 0.0,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        name: str = "image",
    ):
        if (mean is None) != (std is None):
            raise ValueError("Supply both mean and std for normalization")

        self.bands = bands
        self.decode = decode
        self.fill = fill
        self.mean = mean
        self.std = std
        self.name = name

    @chain_step()
    def __call__(self, rasters: xr.DataTree) -> xr.DataTree:
        """Prepare the named raster's bands as standardized float32, lazily.

        Runs before cropping or chipping, so every sample of a raster is
        prepared alike.

        Args:
            rasters: Tree holding the raster under `name`.

        Returns:
            `sample`: one node, `name`, whose variables are `bands` in model
            order as float32 on the raster's grid: decoded, filled and
            standardized as the arguments say, with the raster's scalar
            coordinates.

        Raises:
            KeyError: `rasters` has no node of this name, or it lacks a band.
        """
        image = rasters[self.name].to_dataset()[self.bands]
        if self.decode:
            image = image.gs.mask_and_scale()
        if self.fill is not None:
            image = image.fillna(self.fill)
        image = image.astype("float32")

        if self.mean is not None and self.std is not None:
            for band, mean, std in zip(self.bands, self.mean, self.std, strict=True):
                image[band] = (image[band] - mean) / std

        sample = xr.DataTree.from_dict({self.name: image})
        return sample
