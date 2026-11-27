"""Turn scene logits into a class-index raster."""

import numpy as np
import xarray as xr

import geosave  # noqa: F401 - register the raster accessors


def postprocess(logits: xr.DataArray) -> xr.Dataset:
    """Return the highest-scoring class on each pixel, using -1 for missing logits."""
    valid = np.isfinite(logits).all("band")
    classes = logits.argmax("band", skipna=False).where(valid, -1).astype("int32")
    return classes.gs.write_nodata(-1).to_dataset(name="class_id")
