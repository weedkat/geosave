"""Preprocessors the library registers: turn a stored raster into model input."""

import torch
import xarray as xr

from geosave.ml.processing.roles import PREPROCESSORS


@PREPROCESSORS.register("sentinel2_reflectance")
class Reflectance:
    """Decode Sentinel-2 digital numbers to reflectance, and standardize batches.

    Args:
        bands: Sentinel-2 bands of the image, in model order.
        mean: Reflectance mean of each band; None leaves batches unnormalized.
        std: Reflectance standard deviation of each band.

    Raises:
        ValueError: Only one of `mean` and `std` is given.
    """

    def __init__(
        self,
        bands: list[str],
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ):
        if (mean is None) != (std is None):
            raise ValueError("Supply both mean and std for normalization")

        self.bands = bands
        self.mean = mean
        self.std = std

    def preprocess(self, raster: xr.Dataset) -> xr.Dataset:
        """Decode a whole raster's digital numbers to reflectance, lazily.

        Missing pixels become 0. Runs before cropping or chipping, so every
        sample of a raster is decoded alike.
        """
        reflectance = raster[self.bands].gs.mask_and_scale()

        return reflectance.fillna(0).astype("float32")

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Standardize a `(..., band, y, x)` reflectance batch; applied after augmentation."""
        if self.mean is None or self.std is None:
            return image

        mean = image.new_tensor(self.mean)[:, None, None]
        std = image.new_tensor(self.std)[:, None, None]
        return (image - mean) / std
