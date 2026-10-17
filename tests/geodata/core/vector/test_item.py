import dask.array as da
import pytest
import xarray as xr

from geosave.geodata.core import GeoItem


def test_an_item_row_reads_through_geoitem(items):
    assert isinstance(items.iloc[0].gs, GeoItem)


def test_an_item_row_opens_lazily_on_its_native_grid(items, wide_grid):
    radar_row = items[items["collection"] == "s1"].iloc[0]

    raster = radar_row.gs.to_xarray()

    assert isinstance(raster, xr.Dataset)
    assert raster.gs.geobox == wide_grid
    assert isinstance(raster.vv.data, da.Array)
    assert raster.vv.dtype == "uint16"


def test_an_item_row_carries_its_collections_acdd(items):
    raster = items.iloc[0].gs.to_xarray()

    assert raster.attrs["id"] == items.iloc[0]["collection"]


def test_an_item_rows_anchor_is_its_grid_over_its_span(items, grid):
    anchor = items.iloc[1].gs.anchor

    assert anchor.geobox == grid
    assert str(anchor.timespan[0]) == "2024-03-08 00:00:00"


def test_a_row_of_neither_table_has_no_gs_accessor():
    import pandas as pd

    with pytest.raises(AttributeError, match="item table or a sample table"):
        pd.Series({"a": 1}).gs  # noqa: B018


def test_a_timeless_item_reads_without_time_coordinates(grid, tmp_path):
    import numpy as np

    from geosave.geodata import raster
    from geosave.geodata.warnings import TimelessItemWarning
    from tests.geodata.conftest import item_table

    data = raster({"height": (("y", "x"), np.ones(grid.shape, dtype="float32"))}, grid)
    with pytest.warns(TimelessItemWarning):
        items = item_table(data.gs.to_cog(str(tmp_path / "dem.tif")), collection="dem")
    opened = items.iloc[0].gs.to_xarray()
    assert "time" not in opened.coords
    assert "start_datetime" not in opened.coords
    assert "end_datetime" not in opened.coords
    assert opened.attrs["id"] == "dem"
    assert isinstance(opened.height.data, da.Array)
    sample = items.gs.sample().iloc[0].gs.to_xarray()["dem"].dataset
    assert sample.identical(opened)
