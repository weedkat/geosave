from __future__ import annotations


import dask.array as da
import xarray as xr


def test_an_item_row_opens_lazily_on_its_native_grid(items, wide_grid):
    radar_row = items[items["collection"] == "s1"].iloc[0]

    raster = radar_row.gs.to_datatree().gs.to_dataset()

    assert isinstance(raster, xr.Dataset)
    assert raster.gs.geobox == wide_grid
    assert isinstance(raster.vv.data, da.Array)
    assert raster.vv.dtype == "uint16"


def test_boundary_record_has_an_accessor_but_is_not_a_stac_item():
    import geopandas as gpd
    import pytest
    from shapely.geometry import box

    boundary = gpd.GeoDataFrame(
        {"name": ["area"]}, geometry=[box(0, 0, 1, 1)], crs=4326
    )
    record = boundary.iloc[0]

    with pytest.raises(ValueError, match="STAC Item operations need.*to_catalog"):
        record.gs.to_item()
    assert not hasattr(boundary["name"], "gs")
    assert not hasattr(boundary.geometry, "gs")


def test_an_item_row_reads_its_pixels_onto_a_grid(items, wide_grid):
    import numpy as np

    radar_row = items[items["collection"] == "s1"].iloc[0]

    pixels = radar_row.gs.read_pixels(wide_grid, variables=["vv"])

    assert pixels.odc.geobox == wide_grid
    stored = (
        radar_row.gs.to_datatree()
        .gs.to_dataset()
        .vv.gs.to_nan()
        .astype("float32")
        .values
    )
    np.testing.assert_array_equal(pixels.vv.values, stored)


def test_a_row_on_mixed_grids_reads_each_grid_as_a_node(tmp_path):
    from geosave.geodata.stac.catalog import from_items
    from tests.geodata.io.test_stac import write_mixed_scene

    rows = from_items([write_mixed_scene(tmp_path)])

    scene = rows.iloc[0].gs.to_datatree()
    tree = rows.gs.to_datatree()

    assert list(scene.gs.rasters) == ["r10m", "r20m"]
    assert list(tree.gs.rasters) == ["s2/r10m", "s2/r20m"]
    assert abs(tree["s2"].gs.to_dataset().odc.geobox.resolution.x) == 10
