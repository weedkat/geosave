import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

from geosave.geodata import open_vector, raster, tree
from geosave.geodata.core import GeoCatalog
from geosave.geodata.transform import temporal
from tests.geodata.conftest import item_table, tree_templates

S2_DATES = ["2024-01-10", "2024-03-08", "2024-03-28", "2024-04-12", "2024-07-03"]
S1_DATES = ["2024-03-05", "2024-04-04", "2024-05-20", "2024-06-15", "2024-07-01"]


@pytest.fixture(scope="module")
def grid() -> GeoBox:
    # 50 rows by 70 columns at 10 m
    return GeoBox.from_bbox(
        (300000, 5000000, 300700, 5000500), "EPSG:32633", resolution=10
    )


@pytest.fixture(scope="module")
def wide_grid() -> GeoBox:
    # Radar has the same footprint at 20 m.
    return GeoBox.from_bbox(
        (300000, 5000000, 300700, 5000500), "EPSG:32633", resolution=20
    )


def _series(name: str, dates: list[str], grid: GeoBox):
    times = pd.to_datetime(dates)
    rng = np.random.default_rng(len(name))
    values = rng.integers(1, 9000, (len(times), *grid.shape)).astype("uint16")
    return raster({name: (("time", "y", "x"), values)}, grid, time=times)


@pytest.fixture(scope="module")
def sample(grid):
    """Optical on its dates, a DEM and a label spanning them, all on one grid."""
    rng = np.random.default_rng(0)
    return tree(
        {
            "s2": _series("red", S2_DATES, grid),
            "dem": raster(
                {"height": (("y", "x"), rng.random(grid.shape).astype("float32"))}, grid
            ),
            "label": raster(
                {"label": (("y", "x"), rng.integers(0, 3, grid.shape).astype("uint8"))},
                grid,
            ),
        }
    )


@pytest.fixture(scope="module")
def radar(wide_grid):
    return tree({"s1": _series("vv", S1_DATES, wide_grid)})


@pytest.fixture(scope="module")
def items(sample, radar, tmp_path_factory):
    """Raw scenes catalogued and read back: s2, dem, label on one grid, s1 on its own."""
    root = tmp_path_factory.mktemp("raw")
    optical_rows = item_table(
        sample.gs.to_cog(tree_templates(sample, root)), prefix="a"
    )
    radar_rows = item_table(radar.gs.to_cog(tree_templates(radar, root)), prefix="b")
    path = GeoCatalog.concat([optical_rows, radar_rows]).gs.to_geoparquet(
        root / "items.parquet"
    )
    return open_vector(path)


def _composite(items, collection: str, root):
    """Monthly median composites of one collection from March to July, gaps filled."""
    from geosave.geodata.io.raster.stac import open_raster

    observed = open_raster(items[items["collection"] == collection].gs.to_items())
    observed = observed.sel(time=slice("2024-03-01", "2024-07-31"))
    monthly = temporal.reduce(temporal.resample(observed, "MS"), "median")
    filled = temporal.interpolate(monthly, "nearest")
    files = filled.gs.to_cog(f"{root}/{collection}/{collection}_{{time:%Y%m}}.tif")
    return item_table(files, prefix="c", collection=collection)


@pytest.fixture(scope="module")
def composites(items, tmp_path_factory):
    """Label and DEM beside monthly s2 and s1 composites, March to July."""
    root = tmp_path_factory.mktemp("composites")
    static_rows = items[items["collection"].isin(["label", "dem"])]
    tables = [
        static_rows,
        _composite(items, "s2", root),
        _composite(items, "s1", root),
    ]
    path = GeoCatalog.concat(tables).gs.to_geoparquet(root / "composites.parquet")
    return open_vector(path)
