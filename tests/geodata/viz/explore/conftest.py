import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Point, box

from geosave.geodata.stac.catalog import from_items
from tests.conftest import build_dataset, build_dataset as build_raster
from tests.geodata.viz.conftest import ORIGIN, create_grid
from tests.geodata.conftest import create_file_items, create_test_collection

DATES = np.array(["2024-03-05", "2024-04-05", "2024-05-05"], "datetime64[ns]")


@pytest.fixture
def scene() -> xr.Dataset:
    values = np.random.default_rng(0).random((64, 64)).astype("float32")
    raster = build_raster(
        {
            name: (("y", "x"), values * k)
            for name, k in (("B04", 1), ("B03", 2), ("B02", 3))
        },
        create_grid(),
    )
    raster["cover"] = (("y", "x"), (values > 0.5).astype("uint8"))
    raster = raster.gs.write_legend(
        {0: "water", 1: "trees"}, {0: "#419bdf", 1: "#397d49"}, target="cover"
    )
    return raster.gs.write_rgb("B04", "B03", "B02")


@pytest.fixture
def monthly(scene: xr.Dataset) -> xr.Dataset:
    """The scene on three dates, each brighter than the last."""
    return xr.concat([scene * k for k in (1, 2, 3)], dim="time").assign_coords(
        time=DATES
    )


@pytest.fixture
def fields() -> gpd.GeoDataFrame:
    x, y = ORIGIN
    return gpd.GeoDataFrame(
        {"crop": ["rice", "palm", "rice"], "area": [1.0, 2.0, 3.0]},
        geometry=[
            box(x, y, x + 100, y + 100),
            box(x + 200, y, x + 300, y + 100),
            Point(x + 500, y + 500),
        ],
        crs="EPSG:32749",
    )


@pytest.fixture
def catalog(monthly: xr.Dataset, tmp_path) -> gpd.GeoDataFrame:
    """An item table of the monthly scene, one COG per date."""

    files = monthly[["B04"]].gs.to_cog(
        tmp_path, path_format="s_{time:%Y%m%d}_{band}.tif"
    )
    collection = create_test_collection("monthly", description="Monthly imagery")
    return from_items(
        create_file_items(files, id_format="s_{start:%Y%m%d}", collection=collection),
        collections=[collection],
    )


@pytest.fixture
def rgb_catalog(scene: xr.Dataset, tmp_path):
    """Persist a small real RGB Item table for subprocess browser applications."""

    y, x = np.indices((64, 64), dtype="float32")
    image = build_dataset(
        {
            "red": (("y", "x"), x / 63),
            "green": (("y", "x"), y / 63),
            "blue": (("y", "x"), (x + y) / 126),
        },
        scene.gs.geobox,
        time=np.datetime64("2024-03-05"),
    ).gs.write_rgb("red", "green", "blue")
    files = image.gs.to_cog(tmp_path, path_format="rgb.tif")
    collection = create_test_collection("rgb", description="Browser RGB fixture")
    items = from_items(
        create_file_items(files, id_format="rgb", collection=collection),
        collections=[collection],
    )
    return items.gs.to_geoparquet(tmp_path / "rgb.parquet")
