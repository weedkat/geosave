from __future__ import annotations

from tests.conftest import write_cogs


import numpy as np
import pytest
import rasterio

from tests.conftest import build_dataset
from tests.geodata.conftest import create_file_items


# A 1024² raster in Web Mercator, 10 m pixels, its COGs holding overviews 2 and 4.
PIXEL_SIZE = 10.0


@pytest.fixture
def pyramid_item(tmp_path):
    from odc.geo.geobox import GeoBox

    grid = GeoBox.from_bbox(
        (0, 0, 1024 * PIXEL_SIZE, 1024 * PIXEL_SIZE),
        crs="EPSG:3857",
        resolution=PIXEL_SIZE,
    )
    values = np.arange(1024 * 1024, dtype="uint16").reshape(1024, 1024) % 1000 + 1
    values[:8, :8] = 0
    scene = build_dataset(
        {"red": (("y", "x"), values), "nir": (("y", "x"), values * 2)},
        grid,
        nodata=0,
        time=np.datetime64("2024-03-05"),
    )
    files = write_cogs(
        scene, tmp_path, format_paths="{variable}.tif", blocksize=256, overview_count=2
    )
    paths = {}
    for path in files:
        paths[path.stem] = path
    return next(iter(create_file_items(paths, id_format="pyramid")))


def record_overview_levels(monkeypatch) -> list:
    """Record the `overview_level` of each file opened, None for the base image."""
    levels = []
    original = rasterio.open

    def open_and_record(source, *args, **options):
        levels.append(options.get("overview_level"))
        return original(source, *args, **options)

    monkeypatch.setattr(rasterio, "open", open_and_record)
    return levels


def create_view(left, bottom, right, top, pixel_size):
    from odc.geo.geobox import GeoBox

    return GeoBox.from_bbox(
        (left, bottom, right, top), crs="EPSG:3857", resolution=pixel_size
    )
