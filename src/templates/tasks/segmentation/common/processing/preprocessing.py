"""Model input preparation shared by the Dataset and prediction pipeline."""

import xarray as xr

import geosave  # noqa: F401 - register the raster accessors


class Reflectance:
    """Select Sentinel-2 bands in model order and decode stored reflectance."""

    def preprocess(self, rasters: xr.DataTree) -> xr.DataTree:
        """Select the `image` bands in model order and decode stored reflectance lazily.

        Returns:
            One node, `image`, holding B02, B03, B04 and B08 as
            decoded reflectance on the raster's grid.
        """
        selected = rasters["image"].gs.to_dataset()[["B02", "B03", "B04", "B08"]]
        for variable in selected.data_vars.values():
            if variable.dims != ("y", "x"):
                raise ValueError("Select one observation with the expected dimensions")
        return xr.DataTree.from_dict({"image": selected.gs.mask_and_scale()})
