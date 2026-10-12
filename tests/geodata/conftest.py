import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.geodata.core.constructors import raster as build
from geosave.geodata.core.stack import stack as build_stack


def build_raster(
    *,
    crs: str = "EPSG:32749",
    times: int = 0,
    packed: bool = False,
) -> xr.Dataset:
    """Build one raster the way an adapter would.

    Args:
        crs: CRS of the two-by-two grid.
        times: Length of the time axis, or 0 for a timeless raster.
        packed: Give each variable CF packing over stored digital numbers.

    Returns:
        Raster Dataset holding `red` and `nir` as uint16.

    Examples:
        >>> build_raster(times=2).red.dims
        ('time', 'y', 'x')
    """
    geobox = GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_020.0, 9_000_020.0),
        crs=crs,
        shape=(2, 2),
        tight=True,
    )
    coords = dict(xr_coords(geobox, always_yx=True))
    spatial_dims = ("y", "x")
    values = np.array([[1000, 2000], [3000, 0]], dtype="uint16")
    if times:
        coords["time"] = xr.DataArray(
            np.array(
                [f"2025-06-{day + 1:02d}" for day in range(times)],
                dtype="datetime64[ns]",
            ),
            dims="time",
        )
        block = np.broadcast_to(values, (times, 2, 2)).copy()
        dims = ("time", *spatial_dims)
    else:
        block = values
        dims = spatial_dims

    raw = xr.Dataset({"red": (dims, block), "nir": (dims, block)}, coords=coords)
    if packed:
        for name in raw.data_vars:
            raw[name].attrs.update(
                {
                    "scale_factor": np.float32(1e-4),
                    "add_offset": np.float32(0.0),
                    "_FillValue": np.uint16(0),
                }
            )
    return raw.gs.write_crs()


@pytest.fixture
def raster() -> xr.Dataset:
    return build_raster()


@pytest.fixture
def stack(raster: xr.Dataset) -> xr.DataTree:
    return build_stack({"optical": raster[["red"]], "infrared": raster[["nir"]]})


@pytest.fixture
def grid() -> GeoBox:
    return GeoBox.from_bbox(
        (300000, 5000000, 300640, 5000640), "EPSG:32633", resolution=10
    )


@pytest.fixture
def scene(grid):
    """Two dates of two uint16 bands with a fill value."""
    times = pd.to_datetime(["2025-06-01", "2025-06-11"])
    cube = np.arange(2 * 64 * 64, dtype="uint16").reshape(2, 64, 64)
    data = build(
        {"red": (("time", "y", "x"), cube), "nir": (("time", "y", "x"), cube + 1)},
        grid,
        time=times,
    )
    for name in data.data_vars:
        data[name].attrs["_FillValue"] = np.uint16(0)
    return data


@pytest.fixture
def label(grid):
    """One timeless uint8 class plane on the same grid."""
    return build({"label": (("y", "x"), np.ones((64, 64), dtype="uint8"))}, grid)


def cog_template(root, name: str, *, dated: bool) -> str:
    """Spell, as a `to_cog` template, the layout one raster's files are expected in.

    A dated raster is a folder named after it holding one file per instant; an
    undated one is one file.
    """
    if dated:
        return f"{root}/{name}/{name}_{{time:%Y%m%dT%H%M%S}}.tif"
    return f"{root}/{name}.tif"


def stack_templates(sample, root) -> dict[str, str]:
    """Give every group of a stack its `cog_template` under `root`."""
    return {
        name: cog_template(root, name, dated="time" in raster.dims)
        for name, raster in sample.gs.rasters.items()
    }


def read_items(rows):
    """Read an item table's files as one raster, as a window reads one collection."""
    from geosave.geodata.io.raster import cogs
    from geosave.geodata.stac.item import cog_files
    from geosave.geodata.io.vector.stac_geoparquet import to_items

    return cogs.read([f for item in to_items(rows) for f in cog_files(item)])


def titled(obj, title: str = "scene"):
    """Return a copy of a raster or stack whose root names its title."""
    copied = obj.copy()
    copied.attrs = {**obj.attrs, "title": title}
    return copied


def geobox(*, crs: str = "EPSG:32633", shape: tuple[int, int] = (8, 8)) -> GeoBox:
    """Build one grid for a raster-construction test.

    Args:
        crs: CRS the grid sits in.
        shape: Grid height and width in pixels.

    Returns:
        Grid of `shape` at ten-unit pixels.
    """
    left, bottom = 300_000.0, 5_000_000.0
    return GeoBox.from_bbox(
        (left, bottom, left + shape[1] * 10, bottom + shape[0] * 10),
        crs=crs,
        shape=shape,
        tight=True,
    )
