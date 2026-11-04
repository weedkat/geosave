"""Item tables of real COGs for the catalog accessors and samples."""

import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pystac
import pytest
from odc.geo.geobox import GeoBox
from shapely.geometry import box, mapping

from geosave.geodata import open_vector
from tests.conftest import build_dataset, build_tree
from geosave.geodata.core import GeoCatalog
from geosave.geodata.stac.catalog import from_items
from tests.geodata.conftest import item_table, write_collection_cogs
from tests.geodata.conftest import create_file_items, create_test_collection

OPTICAL_DATES = ["2024-01-10", "2024-03-08", "2024-03-28", "2024-04-12", "2024-07-03"]
RADAR_DATES = ["2024-03-05", "2024-04-04", "2024-05-20", "2024-06-15", "2024-07-01"]


@pytest.fixture(scope="module")
def grid() -> GeoBox:
    """50 rows by 70 columns at 10 m."""
    return GeoBox.from_bbox(
        (300000, 5000000, 300700, 5000500), "EPSG:32633", resolution=10
    )


@pytest.fixture(scope="module")
def wide_grid() -> GeoBox:
    """The same footprint at 20 m, as radar delivers it."""
    return GeoBox.from_bbox(
        (300000, 5000000, 300700, 5000500), "EPSG:32633", resolution=20
    )


def create_series(variable: str, dates: list[str], grid: GeoBox):
    """Return random uint16 pixels of one variable on each date."""
    times = pd.to_datetime(dates)
    rng = np.random.default_rng(len(variable))
    pixels = rng.integers(1, 9000, (len(times), *grid.shape)).astype("uint16")
    return build_dataset({variable: (("time", "y", "x"), pixels)}, grid, time=times)


@pytest.fixture(scope="module")
def optical_tree(grid):
    """Optical on its dates, a DEM and a label, all on one 10 m grid."""
    rng = np.random.default_rng(0)
    heights = rng.random(grid.shape).astype("float32")
    classes = rng.integers(0, 3, grid.shape).astype("uint8")
    return build_tree(
        {
            "s2": create_series("red", OPTICAL_DATES, grid),
            "dem": build_dataset({"height": (("y", "x"), heights)}, grid),
            "label": build_dataset({"label": (("y", "x"), classes)}, grid),
        }
    )


@pytest.fixture(scope="module")
def radar_tree(wide_grid):
    """Radar on its own dates at 20 m."""
    return build_tree({"s1": create_series("vv", RADAR_DATES, wide_grid)})


@pytest.fixture(scope="module")
def items(optical_tree, radar_tree, tmp_path_factory):
    """Both trees saved as COGs, catalogued, written and opened back."""
    root = tmp_path_factory.mktemp("raw")
    optical_files = write_collection_cogs(optical_tree, root)
    radar_files = write_collection_cogs(radar_tree, root)
    optical_rows = item_table(optical_files, prefix="a")
    radar_rows = item_table(radar_files, prefix="b")
    path = GeoCatalog.concat([optical_rows, radar_rows]).gs.to_geoparquet(
        root / "items.parquet"
    )
    return open_vector(path)


# Footprints in degrees: place A holds S2 on two dates, a label, and S1 whose
# coarser grid snaps a little wider; place B, apart from A, holds S2 alone.
FOOTPRINTS = {
    "s2_A_jan": ("s2", box(0, 0, 1, 1)),
    "s2_A_feb": ("s2", box(0, 0, 1, 1)),
    "s1_A": ("s1", box(0, 0, 1.02, 1)),
    "label_A": ("label", box(0, 0, 1, 1)),
    "s2_B_jan": ("s2", box(3, 0, 4, 1)),
}


def create_item(item_id: str, collection: str, footprint) -> pystac.Item:
    """Return an Item with one data asset on `footprint`."""
    item = pystac.Item(
        id=item_id,
        geometry=mapping(footprint),
        bbox=list(footprint.bounds),
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
        collection=collection,
    )
    item.add_asset("data", pystac.Asset(href=f"{item_id}.tif", roles=["data"]))
    return item


@pytest.fixture
def catalog():
    entries = []
    for item_id, (collection, footprint) in FOOTPRINTS.items():
        entries.append(create_item(item_id, collection, footprint))
    collections = []
    for collection in ("s2", "s1", "label"):
        collections.append(create_test_collection(collection, description=collection))
    return from_items(entries, collections=collections)


def read_sample_ids(samples) -> list[list[str]]:
    """Return each sample's Item ids, in sample order."""
    sample_ids = []
    for rows in samples:
        sample_ids.append(rows["id"].tolist())
    return sample_ids


def create_scene_items(raster, root, collection=None):
    """Build native Items from files written through the public raster API."""

    template = "{time:%Y%m%d}_{band}.tif" if "time" in raster.dims else "label.tif"
    scenes = raster.gs.to_cog(root, path_format=template)
    return list(
        create_file_items(
            scenes,
            collection=collection,
            id_format="forest_{start:%Y%m%dT%H%M%S}",
            datetime=datetime(2025, 6, 1, tzinfo=UTC),
        )
    )


TILE_CRS = "EPSG:32633"
# Tile A: s2 scenes, s1 within two days of the first three, a timeless DEM.
S2_DATES = ["2024-01-05", "2024-01-20", "2024-02-04", "2024-03-08", "2024-04-07"]
S1_DATES = ["2024-01-06", "2024-02-03", "2024-03-09", "2024-04-20"]


def write_items(root, collection, variable, left, resolution, dates):
    """Write a 640 m square COG per date, or one timeless COG, and return their Items."""
    grid = GeoBox.from_bbox(
        (left, 5000000, left + 640, 5000640), TILE_CRS, resolution=resolution
    )
    rng = np.random.default_rng(left + resolution)
    if dates:
        pixels = rng.random((len(dates), *grid.shape), dtype="float32")
        raster = build_dataset(
            {variable: (("time", "y", "x"), pixels)}, grid, time=pd.to_datetime(dates)
        )
        template = f"{collection}_{left}_{{time:%Y%m%d}}.tif"
        options = {}
    else:
        raster = build_dataset(
            {variable: (("y", "x"), rng.random(grid.shape, dtype="float32"))}, grid
        )
        template = f"{collection}_{left}.tif"
        span = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC))
        options = {"datetime": span}
    scenes = raster.gs.write_nodata(np.nan).gs.to_cog(root, path_format=template)
    id_format = f"{collection}_{left}_{{start:%Y%m%d}}"
    return list(
        create_file_items(scenes, collection=collection, id_format=id_format, **options)
    )


@pytest.fixture(scope="module")
def tiles(tmp_path_factory):
    """Tile A with s2, s1 at 20 m and a DEM; tile B, apart from A, with one s2 scene."""
    root = tmp_path_factory.mktemp("tiles")
    entries = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entries += write_items(root, "s2", "red", 300000, 10, S2_DATES)
        entries += write_items(root, "s1", "vv", 300000, 20, S1_DATES)
        entries += write_items(root, "dem", "height", 300000, 10, [])
        entries += write_items(root, "s2", "red", 320000, 10, ["2024-03-08"])
    collections = []
    for name in ("s2", "s1", "dem"):
        collections.append(create_test_collection(name, description=name))
    return from_items(entries, collections=collections)
