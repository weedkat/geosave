"""Prepare the four Sentinel-2 reflectance bands used by the example model."""

import xarray as xr

import geosave  # noqa: F401  registers xarray accessors

BANDS = ["B02", "B03", "B04", "B08"]


def preprocess(sources: xr.DataTree) -> xr.DataTree:
    """Return model-ordered reflectance on its source grid and observation dates."""
    reflectance = sources["s2/reflectance"].gs.to_dataset()[BANDS]
    image = reflectance.gs.mask_and_scale()
    return xr.DataTree.from_dict({"image": image})
