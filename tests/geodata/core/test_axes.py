"""The spatial coordinates state what the grid says they measure."""

import numpy as np
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import raster


def test_a_projected_grid_describes_its_axes_completely() -> None:
    grid = GeoBox.from_bbox((300_000, 5_000_000, 300_020, 5_000_020), crs="EPSG:32633", resolution=10)

    ds = raster({"b": (("y", "x"), np.zeros(grid.shape, "uint8"))}, grid)

    assert ds.y.attrs == {
        "crs": "EPSG:32633",
        "resolution": -10.0,
        "standard_name": "projection_y_coordinate",
        "units": "metre",
        "axis": "Y",
    }
    assert ds.x.attrs["standard_name"] == "projection_x_coordinate"
    assert ds.x.attrs["axis"] == "X"


def test_a_geographic_grid_names_latitude_and_longitude() -> None:
    grid = GeoBox.from_bbox((10, 20, 13, 22), crs="EPSG:4326", shape=(2, 3))

    ds = raster({"b": (("y", "x"), np.zeros(grid.shape, "uint8"))}, grid)

    assert (ds.y.attrs["standard_name"], ds.x.attrs["standard_name"]) == ("latitude", "longitude")
