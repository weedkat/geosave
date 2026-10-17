import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from geosave.geodata import Dataset, raster


@pytest.fixture
def scene() -> Dataset:
    grid = GeoBox.from_bbox((10, 50, 12, 52), "EPSG:4326", shape=(8, 8))
    return raster(
        {
            "red": (
                ("time", *("y", "x")),
                np.zeros((2, 8, 8), dtype="float32"),
            )
        },
        grid,
        time=np.array(["2024-12-31T14:00", "2024-01-01T03:00"], dtype="datetime64[m]"),
    )
