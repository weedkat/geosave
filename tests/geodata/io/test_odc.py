"""Selected STAC Items load independently of discovery."""

import dask.array as da
import numpy as np
import pytest
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox

from geosave.geodata import io
from tests.conftest import build_dataset
from tests.geodata.conftest import create_file_items


@pytest.fixture
def observation(tmp_path):
    grid = GeoBox.from_bbox(
        BoundingBox(500000, 5000000, 500040, 5000040, crs="EPSG:32633"), resolution=10
    )
    raster = build_dataset(
        {"red": (("y", "x"), np.arange(16, dtype="uint16").reshape(4, 4))}, grid
    )
    raster = raster.assign_coords(time=np.datetime64("2025-01-15"))
    path = io.formats.geotiff.write_cog(raster, tmp_path / "red.tif")
    item = next(create_file_items({"red": path}))
    item.collection_id = "local"
    item.properties["eo:cloud_cover"] = 20.0
    return item, raster


def test_polygon_and_explicit_grid_load_equal_lazy_pixels(observation):
    item, raster = observation
    grid = raster.gs.geobox
    explicit = io.load_dataset([item], geobox=grid)
    polygon = io.load_dataset(
        iter([item]), geopolygon=grid.extent, crs=grid.crs, resolution=10
    )
    assert explicit.gs.geobox == polygon.gs.geobox == grid
    assert isinstance(polygon.red.data, da.Array)
    np.testing.assert_array_equal(explicit.red.values, polygon.red.values)
    np.testing.assert_array_equal(explicit.red.values[0], raster.red.values)


def test_properties_follow_fused_time_and_keep_common_attrs(observation):
    item, raster = observation
    other = item.clone()
    other.id = "second"
    other.properties["eo:cloud_cover"] = 40.0
    item.properties["platform"] = other.properties["platform"] = "test"
    loaded = io.load_dataset(
        [item, other], geobox=raster.gs.geobox, with_properties=["eo:cloud_cover"]
    )
    assert loaded.gs.stac_properties["eo:cloud_cover"].tolist() == [30.0]
    assert loaded.gs.stac_common_properties == {"platform": "test"}
    assert "eo:cloud_cover" not in loaded.data_vars


def test_loader_rejects_mixed_collections_before_reading(observation):
    item, raster = observation
    other = item.clone()
    other.collection_id = "other"
    with pytest.raises(ValueError, match="one collection"):
        io.load_dataset([item, other], geobox=raster.gs.geobox)


def test_tree_preserves_relative_resolution_families(observation, tmp_path):
    item, raster = observation
    item.properties["platform"] = "test"
    coarse = build_dataset(
        {"swir": (("y", "x"), np.full((2, 2), 7, dtype="uint16"))},
        raster.gs.geobox.zoom_out(2),
    )
    coarse = coarse.assign_coords(time=np.datetime64("2025-01-15"))
    path = io.formats.geotiff.write_cog(coarse, tmp_path / "swir.tif")
    second = next(create_file_items({"swir": path}))
    for key, asset in second.assets.items():
        item.add_asset(key, asset)
    tree = io.load_datatree([item], geobox=raster.gs.geobox)
    assert set(tree.gs.rasters) == {"local/r10m", "local/r20m"}
    assert tree["local"].to_dataset().gs.stac_common_properties["platform"] == "test"
    assert tree["local/r20m"].gs.geobox == coarse.gs.geobox
    np.testing.assert_array_equal(tree["local/r20m"].swir.values[0], coarse.swir.values)


def test_signing_happens_at_load_without_mutating_items(observation):
    item, raster = observation
    original = item.assets["red"].href
    item.assets["red"].href = "https://unsigned.test/red.tif"
    urls = []

    def sign(url):
        urls.append(url)
        return original

    loaded = io.load_dataset([item], geobox=raster.gs.geobox, patch_url=sign)
    np.testing.assert_array_equal(loaded.red.values[0], raster.red.values)
    assert urls == ["https://unsigned.test/red.tif"]
    assert item.assets["red"].href == "https://unsigned.test/red.tif"
