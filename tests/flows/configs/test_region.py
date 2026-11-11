import json

import numpy as np
from pydantic import TypeAdapter

from geosave.geodata import io
from tests.conftest import build_dataset
from geosave_flows.configs import RegionConfig


def test_coordinate_region_opens_a_native_grid():
    region, grid = (
        TypeAdapter(RegionConfig)
        .validate_python(
            {
                "kind": "coordinates",
                "latitude": 45,
                "longitude": 12,
                "shape": [4, 6],
                "resolution": 10,
                "crs": "EPSG:32633",
                "timespan": "2025-01",
            }
        )
        .open()
    )

    assert grid.shape.yx == (4, 6)
    assert region.timespan is not None


def test_geojson_region_opens_a_native_grid(tmp_path):
    path = tmp_path / "area.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [[12, 45], [12.01, 45], [12.01, 45.01], [12, 45.01], [12, 45]]
                ],
            }
        )
    )

    region, grid = (
        TypeAdapter(RegionConfig)
        .validate_python({"kind": "geojson", "path": path, "shape": [4, 4]})
        .open()
    )

    assert grid.shape.yx == (4, 4)


def test_raster_region_uses_the_file_grid_and_time(tmp_path, grid):
    instant = np.datetime64("2025-01-15")
    label = build_dataset(
        {"class": (("y", "x"), np.ones((4, 4), dtype="uint8"))},
        grid,
    ).assign_coords(time=instant)
    path = io.formats.geotiff.write_cog(label, tmp_path / "label.tif").path

    result, result_grid = (
        TypeAdapter(RegionConfig)
        .validate_python({"kind": "raster", "path": path})
        .open()
    )

    assert result_grid == label.gs.geobox
    assert result.timespan == label.gs.region.timespan
