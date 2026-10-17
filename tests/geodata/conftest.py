import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.geodata.core.raster.create import raster as build
from geosave.geodata.core.raster.create import tree as build_tree


def flat(scenes):
    """Return the files of every scene a writer returned, in order."""
    return [file for scene in scenes for file in scene.written.values()]


def item_table(scenes, *, id=None, prefix=None, **options):
    """Catalog the scenes a writer returned as an item table, one Item each.

    Each Item is named `id`, else its first file's stem, under `prefix/` where
    one is given. An undated scene takes the span the dated ones cover unless
    `datetime` is given.
    """
    from datetime import UTC

    from geosave.geodata import stac

    scenes = list(scenes)
    spans = [f.header.gs.timespan for f in flat(scenes) if f.header.gs.timespan]
    if spans and "datetime" not in options:
        start = min(start for start, _ in spans).replace(tzinfo=UTC)
        options["datetime"] = (start, max(end for _, end in spans).replace(tzinfo=UTC))
    items = []
    for scene in scenes:
        name = id or next(iter(scene.written.values())).path.stem
        name = name if prefix is None else f"{prefix}/{name}"
        items.append(next(stac.create_items(scene, id_format=name, **options)))
    return stac.create_catalog(
        items,
        collections=[
            stac.create_collection(key, description=key)
            for key in dict.fromkeys(item.collection_id for item in items)
            if key is not None
        ],
    )


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
def tree(raster: xr.Dataset) -> xr.DataTree:
    return build_tree({"optical": raster[["red"]], "infrared": raster[["nir"]]})


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


def tree_templates(sample, root) -> dict[str, str]:
    """Give every group of a tree its `cog_template` under `root`."""
    return {
        name: cog_template(root, name, dated="time" in raster.dims)
        for name, raster in sample.gs.rasters.items()
    }


def read_items(rows):
    """Read an item table's files as one raster, as a window reads one collection."""
    from geosave.geodata.stac.translate.table import to_items
    from geosave.geodata.io.raster.stac import open_raster

    return open_raster(to_items(rows))


def titled(obj, title: str = "scene"):
    """Return a copy of a raster or tree whose root names its title."""
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


@pytest.fixture
def rows(scene, label, tmp_path):
    sample = build_tree({"optical": scene, "label": label})
    saved = sample.gs.to_cog(tree_templates(sample, tmp_path / "s0"))
    return item_table(saved, prefix="s0")
