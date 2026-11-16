"""This project's band selection and reflectance preparation.

`Reflectance` is a `geosave.ml.processing.raster.preprocessing.base.RasterPreprocessor`: datasets call
it on each raster and the DataModule `normalize` on each batch, so
a release's pipeline prepares inference input exactly as training did.
"""

import torch
import xarray as xr

from geosave.ml.chain import chain_step
from geosave.ml.processing.raster.preprocessing.base import RasterPreprocessor


class Reflectance(RasterPreprocessor):
    """Select Sentinel-2 bands in model order and decode stored reflectance."""

    @chain_step()
    def __call__(self, rasters: xr.DataTree) -> xr.DataTree:
        """Select the `image` bands in model order and decode stored reflectance lazily.

        Returns:
            `sample`: one node, `image`, holding B02, B03, B04 and B08 as
            decoded reflectance on the raster's grid.
        """
        selected = rasters["image"].to_dataset()[["B02", "B03", "B04", "B08"]]
        for variable in selected.data_vars.values():
            if variable.dims != ("y", "x"):
                raise ValueError("Select one observation with the expected dimensions")
        sample = xr.DataTree.from_dict({"image": selected.gs.mask_and_scale()})
        return sample

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Pass decoded reflectance through unchanged, keeping its gradients."""
        return image
