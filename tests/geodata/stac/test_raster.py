"""STAC raster metadata operations preserve lazy pixels and explicit time."""

from datetime import UTC, datetime

import dask.array as da
import pystac

from geosave.geodata.io import open_netcdf, open_zarr
from geosave.geodata.stac.raster import stamp_item_properties, stamp_item_time
from tests.geodata.conftest import build_raster

PROPERTIES = [
    {"eo:cloud_cover": 20.0, "instruments": ["msi"]},
    {"platform": "s2b"},
]


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
    assert "stac_properties" not in dated.coords

    described = stamp_item_properties(dated, entry)
    assert described.stac_properties.values.tolist() == [{"platform": "sentinel-2a"}]
    assert isinstance(described.red.data, da.Array)
    assert described.red.data is raster.red.data
    assert described.time.variable.identical(dated.time.variable)


def test_properties_are_dicts_that_follow_selection():
    ds = build_raster(times=2).chunk()

    recorded = ds.assign_coords(stac_properties=("time", PROPERTIES))

    assert (
        recorded.isel(time=[1, 0]).stac_properties.values.tolist() == PROPERTIES[::-1]
    )
    assert recorded.isel(time=1).stac_properties.item() == {"platform": "s2b"}
    assert recorded.red.data is ds.red.data


def test_properties_round_trip_through_zarr(tmp_path):
    recorded = build_raster(times=2).assign_coords(stac_properties=("time", PROPERTIES))

    recorded.gs.to_zarr(tmp_path / "scene.zarr")
    restored = open_zarr(tmp_path / "scene.zarr")

    assert restored.stac_properties.values.tolist() == PROPERTIES
    assert recorded.stac_properties.values.tolist() == PROPERTIES


def test_properties_round_trip_through_netcdf(tmp_path):
    recorded = build_raster(times=2).assign_coords(stac_properties=("time", PROPERTIES))

    recorded.gs.to_netcdf(tmp_path / "scene.nc")
    with open_netcdf(tmp_path / "scene.nc") as restored:
        assert restored.stac_properties.values.tolist() == PROPERTIES


def test_a_tree_round_trips_each_items_properties(tmp_path):
    scene = build_raster(times=2).assign_coords(stac_properties=("time", PROPERTIES))
    tree = scene.gs.expand()

    tree.gs.to_zarr(tmp_path / "tree.zarr")
    restored = open_zarr(tmp_path / "tree.zarr")

    items = restored["default"].children.values()
    for item, expected in zip(items, PROPERTIES, strict=True):
        assert item.stac_properties.item() == expected
