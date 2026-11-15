"""This project's band selection and reflectance preparation.

`Reflectance` is a `geosave.ml.processing.roles.Preprocessor`: datasets call
`preprocess` on each raster and the DataModule `normalize` on each batch, so
a release's processor prepares inference input exactly as training did.
"""

import torch
import xarray as xr


class Reflectance:
    """Select Sentinel-2 bands in model order and decode stored reflectance."""

    def preprocess(self, raster: xr.Dataset) -> xr.Dataset:
        """Select the bands in model order and decode stored reflectance lazily."""
        selected = raster[["B02", "B03", "B04", "B08"]]
        for variable in selected.data_vars.values():
            if variable.dims != ("y", "x"):
                raise ValueError("Select one observation with the expected dimensions")
        return selected.gs.mask_and_scale()

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Pass decoded reflectance through unchanged, keeping its gradients."""
        return image
