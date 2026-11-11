"""STAC raster metadata operations preserve lazy pixels and explicit time."""

from datetime import UTC, datetime

import dask.array as da
import pystac
import pytest
import xarray as xr

from geosave.geodata.stac import raster as stac_raster

from geosave.geodata.stac.raster import (
    read_properties_coordinate,
    stamp_item_properties,
    stamp_item_time,
)
from tests.geodata.conftest import build_raster


def test_dating_and_properties_are_independent_lazy_operations():
    raster = build_raster(times=1).chunk()
    entry = pystac.Item(
        "scene",
        None,
        None,
        datetime(2025, 6, 1, tzinfo=UTC),
        {"platform": "sentinel-2a"},
    )
    dated = stamp_item_time(raster, entry)
    assert read_properties_coordinate(dated) == []

    described = stamp_item_properties(dated, entry)
    assert read_properties_coordinate(described) == [{"platform": "sentinel-2a"}]
    assert isinstance(described.red.data, da.Array)
    assert described.red.data is raster.red.data
    assert described.time.variable.identical(dated.time.variable)


def test_the_properties_coordinate_round_trips_each_date(tmp_path):
    ds = build_raster(times=2)
    mappings = [{"eo:cloud_cover": 20.0, "instruments": ["msi"]}, {"platform": "s2b"}]

    recorded = stac_raster.stamp_properties_coordinate(ds, mappings)
    recorded.to_zarr(tmp_path / "scene.zarr")
    restored = xr.open_zarr(tmp_path / "scene.zarr")

    assert recorded.stac_properties.dims == ("time",)
    assert stac_raster.read_properties_coordinate(restored) == mappings
    assert stac_raster.read_properties_coordinate(restored.isel(time=1)) == [
        {"platform": "s2b"}
    ]
    with pytest.raises(ValueError, match="properties mappings"):
        stac_raster.stamp_properties_coordinate(ds, [{}])


def test_item_supplied_time_has_cf_semantics_without_computing_pixels():
    from dask.callbacks import Callback

    raster = build_raster().chunk()
    entry = pystac.Item("dated", None, None, datetime(2025, 6, 1, tzinfo=UTC), {})
    with Callback(pretask=lambda *_: pytest.fail("stamping computed pixels")):
        dated = stamp_item_time(raster, entry)
        assert dated.cf.coordinates["time"] == ["time"]
        assert dated.cf.axes["T"] == ["time"]
    assert dated.red.data is raster.red.data
    assert "time" not in raster.coords


def test_common_properties_follow_selection_and_roundtrip(tmp_path):
    from dask.callbacks import Callback

    ds = build_raster(times=2).chunk().rename(time="date")
    ds.date.attrs.update(standard_name="time", axis="T")
    properties = [
        {"platform": "s2a", "eo:cloud_cover": 10.0, "instruments": ["msi"]},
        {"platform": "s2a", "eo:cloud_cover": 40.0},
    ]
    with Callback(pretask=lambda *_: pytest.fail("metadata computed pixels")):
        recorded = stac_raster.stamp_properties_coordinate(ds, properties, dim="date")
        assert recorded.gs.stac_common_properties == {"platform": "s2a"}
        assert recorded.isel(date=0).gs.stac_common_properties == properties[0]
        assert recorded.isel(date=slice(0, 0)).gs.stac_common_properties == {}
        reversed_dates = recorded.isel(date=[1, 0])
        assert reversed_dates.gs.stac_properties["eo:cloud_cover"].tolist() == [40.0, 10.0]
        assert reversed_dates.gs.stac_properties.index.name == "date"
        assert recorded.red.data is ds.red.data

    path = tmp_path / "properties.nc"
    recorded.compute(scheduler="synchronous").to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as restored:
        assert restored.gs.stac_common_properties == {"platform": "s2a"}
        assert restored.isel(date=0).gs.stac_common_properties == properties[0]


def test_timeless_properties_use_one_scalar_record():
    ds = build_raster()
    recorded = stac_raster.stamp_properties_coordinate(ds, [{"platform": "static"}])
    assert recorded.stac_properties.ndim == 0
    assert recorded.gs.stac_common_properties == {"platform": "static"}
    assert recorded.gs.stac_properties["platform"].tolist() == ["static"]
