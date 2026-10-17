import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Point, box

from geosave.geodata.core.raster.create import raster as build_raster
from tests.geodata.viz.conftest import ORIGIN, create_grid

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
    from geosave.geodata import stac

    files = monthly[["B04"]].gs.to_cog(tmp_path / "s_{time:%Y%m%d}_{band}.tif")
    return stac.create_catalog(stac.create_items(files, id_format="s_{start:%Y%m%d}"))
