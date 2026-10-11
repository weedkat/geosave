"""Turn raw loads into model-ready layers.

Only the operations `odc.stac.load` and xarray leave undone. Nothing is
aligned, promoted, reprojected, or resampled implicitly.

Examples:
    One preprocess, end to end::

        clear = nodata.mask(raw, raw.scl.isin(CLEAR_CLASSES))
        filled = combine.mosaic([clear, last_week])
        monthly = temporal.reduce(temporal.resample(filled, "MS"), "median")
        dem = warp.reproject(srtm, monthly, resampling="bilinear")
        prepared = monthly.assign(elevation=dem.elevation)
"""

from . import color, combine, crop, nodata, temporal, vector, warp

__all__ = ["color", "combine", "crop", "nodata", "temporal", "vector", "warp"]
