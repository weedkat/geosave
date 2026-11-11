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
        .vv.gs.mask_and_scale()
        .astype("float32")
        .isel(time=0)
        .values
    )
    np.testing.assert_array_equal(pixels.vv.values, stored)


def test_a_row_on_mixed_grids_reads_each_grid_as_a_node(tmp_path):
    from geosave.geodata.stac.catalog import create_catalog
    from tests.geodata.stac.test_publish import write_mixed_scene

    rows = create_catalog([write_mixed_scene(tmp_path)])

    scene = rows.iloc[0].gs.to_datatree().gs.group_grid()
    tree = rows.gs.to_datatree().gs.group_grid()

    # A row opens as a one-row catalog does.
    assert list(scene.gs.datasets) == list(tree.gs.datasets) == ["s2/r10m", "s2/r20m"]
    scene = tree["s2"].gs.to_dataset(grid="r10m")
    assert abs(scene.odc.geobox.resolution.x) == 10


def test_record_validation_checks_geometry_and_item_fields_independently():
    import geopandas as gpd
    import pytest
    from shapely.geometry import Point

    from geosave.geodata.errors import VectorValidationError

    rows = gpd.GeoDataFrame(geometry=[Point(0, 0), None], index=["site", "missing"])
    assert rows.iloc[0].gs.validate() is None
    with pytest.raises(VectorValidationError, match="STAC Item operations need"):
        rows.iloc[0].gs.validate(stac_item=True)
    assert rows.iloc[1].gs.validate(geometry=False) is None
    with pytest.raises(VectorValidationError, match="Record 'missing'.*missing"):
        rows.iloc[1].gs.validate()
