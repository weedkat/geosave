"""A small dated scene with a static label, shared by the sampling tests."""

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

# June holds three observations, July two, August none, September one, October two.
DATES = pd.DatetimeIndex(
    [
        "2025-06-03",
        "2025-06-13",
        "2025-06-28",
        "2025-07-08",
        "2025-07-23",
        "2025-09-11",
        "2025-10-01",
        "2025-10-21",
    ]
)
GRID = GeoBox.from_bbox(
    (500_000.0, 9_000_000.0, 500_100.0, 9_000_100.0),
    crs="EPSG:32749",
    shape=(10, 10),
    tight=True,
)


def build_scene(grid: GeoBox = GRID, times: pd.DatetimeIndex | None = DATES) -> xr.Dataset:
    """Build lazy `red` and `nir` bands on `grid`, over `times` or timeless."""
    coords = dict(xr_coords(grid, always_yx=True))
    dims = ("y", "x")
    shape = tuple(grid.shape)
    if times is not None:
        coords["time"] = ("time", times)
        dims = ("time", "y", "x")
        shape = (len(times), *shape)
    pixels = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    scene = xr.Dataset({"red": (dims, pixels), "nir": (dims, pixels + 0.5)}, coords=coords)
    return scene.chunk()


def build_tree(grid: GeoBox = GRID, times: pd.DatetimeIndex | None = DATES) -> xr.DataTree:
    """Build an `image` node of two bands and a timeless `label` node on `grid`."""
    cover = np.arange(np.prod(grid.shape), dtype="uint8").reshape(tuple(grid.shape))
    coords = dict(xr_coords(grid, always_yx=True))
    label = xr.Dataset({"cover": (("y", "x"), cover)}, coords=coords).chunk()
    return xr.DataTree.from_dict({"image": build_scene(grid, times), "label": label})


def format_times(raster) -> list[str]:
    """Show a raster's time labels as month-day text."""
    return raster.time.dt.strftime("%m-%d").values.tolist()
