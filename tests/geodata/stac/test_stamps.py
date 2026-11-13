"""Tests for the load stamp registry."""

import json
from collections.abc import Iterator

from datetime import UTC, datetime

import numpy as np
import pystac
import pytest
import xarray as xr

from geosave.geodata.stac.stamps import (
    DEFAULT_STAMPS,
    STAMPS,
    LoadSource,
    check_stamps,
    register_stamp,
    run_stamps,
    stamp_collection,
    stamp_properties,
    stamp_title,
    write_band_attrs,
)


@pytest.fixture
def registered() -> Iterator[None]:
    before = set(STAMPS)

    @register_stamp("first")
    def first(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
        return loaded.assign_attrs(a="first", b="kept")

    @register_stamp("second")
    def second(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
        return loaded.assign_attrs(a="second")

    yield

    for name in set(STAMPS) - before:
        del STAMPS[name]


@pytest.fixture
def source() -> LoadSource:
    return LoadSource(items=[], collection=None, groupby="id")


def test_stamps_run_in_order_and_later_overrides(registered, source):
    out = run_stamps(xr.Dataset(), source, ["first", "second"])

    assert out.attrs["a"] == "second"
    assert out.attrs["b"] == "kept"


def test_run_stamps_with_no_names_returns_input(source):
    loaded = xr.Dataset(attrs={"a": 1})

    assert run_stamps(loaded, source, []).attrs == {"a": 1}


def test_unknown_stamp_names_the_registered_ones(registered):
    with pytest.raises(KeyError, match="registered") as error:
        check_stamps(["first", "nope"])

    assert "nope" in str(error.value)
    assert "second" in str(error.value)


def test_check_stamps_accepts_registered_names(registered):
    check_stamps(["first", "second"])


def test_run_stamps_rejects_unknown_name(registered, source):
    with pytest.raises(KeyError, match="registered"):
        run_stamps(xr.Dataset(), source, ["nope"])


def test_registering_a_name_twice_raises(registered):
    with pytest.raises(ValueError, match="first"):

        @register_stamp("first")
        def again(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
            return loaded


def test_register_returns_the_function(registered):
    def third(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
        return loaded

    assert register_stamp("third")(third) is third
    assert STAMPS["third"] is third


def test_load_source_is_frozen():
    source = LoadSource(items=[], collection=None, groupby="id")

    with pytest.raises(AttributeError):
        source.groupby = "solar_day"


def test_default_stamps_order():
    assert DEFAULT_STAMPS == (
        "raster",
        "eo",
        "classification",
        "rgb",
        "title",
        "properties",
        "collection",
    )


def _titled_item(id: str = "scene", **asset_fields) -> pystac.Item:
    item = pystac.Item(
        id=id,
        geometry=None,
        bbox=None,
        datetime=datetime(2025, 1, 1, tzinfo=UTC),
        properties={},
    )
    item.add_asset(
        "red",
        pystac.Asset(
            href=f"https://example.com/{id}.tif",
            title="Red reflectance",
            extra_fields=asset_fields,
        ),
    )
    return item


def _pixels(*names: str) -> xr.Dataset:
    variables = {}
    for name in names:
        variables[name] = (("y", "x"), [[1]])
    return xr.Dataset(variables)


def test_title_stamp_is_registered():
    assert STAMPS["title"] is stamp_title


def test_title_names_a_one_band_asset():
    item = _titled_item(**{"eo:bands": [{"common_name": "red"}]})

    stamped = stamp_title(_pixels("red"), LoadSource([item], None, "id"))

    assert stamped.red.attrs["long_name"] == "Red reflectance"


def test_title_skips_multi_band_assets():
    item = _titled_item(**{"raster:bands": [{}, {}]})

    stamped = stamp_title(_pixels("red.1", "red.2"), LoadSource([item], None, "id"))

    assert "long_name" not in stamped["red.1"].attrs
    assert "long_name" not in stamped["red.2"].attrs


def test_title_yields_to_the_band_description():
    item = _titled_item(**{"eo:bands": [{"description": "Reflectance"}]})

    stamped = stamp_title(_pixels("red"), LoadSource([item], None, "id"))

    assert "long_name" not in stamped.red.attrs


def test_title_keeps_the_loaded_name():
    loaded = _pixels("red")
    loaded["red"].attrs["long_name"] = "Mine"

    stamped = stamp_title(loaded, LoadSource([_titled_item()], None, "id"))

    assert stamped.red.attrs["long_name"] == "Mine"


def test_write_band_attrs_keeps_agreed_attrs_and_the_loaded_ones():
    items = [_titled_item("a"), _titled_item("b")]
    states = {"a": {"x": 1, "y": 1, "z": 1}, "b": {"x": 1, "y": 2, "z": 1}}
    loaded = _pixels("red")
    loaded["red"].attrs["z"] = "mine"

    stamped = write_band_attrs(
        loaded, LoadSource(items, None, "id"), lambda item, name: states[item.id]
    )

    assert stamped.red.attrs == {"x": 1, "z": "mine"}


def _scene(id: str, day: int, **properties) -> pystac.Item:
    return pystac.Item(
        id=id,
        geometry=None,
        bbox=None,
        datetime=datetime(2025, 1, day, tzinfo=UTC),
        properties=properties,
    )


def _stepped(*items: pystac.Item) -> xr.Dataset:
    times = []
    for entry in items:
        times.append(entry.datetime.replace(tzinfo=None))
    return xr.Dataset(
        {"red": (("time", "y", "x"), np.zeros((len(items), 1, 1)))},
        coords={"time": np.array(times, dtype="datetime64[ns]")},
    )


def _source(*items: pystac.Item, collection=None) -> LoadSource:
    return LoadSource(items=list(items), collection=collection, groupby="id")


def test_properties_one_item_per_step_vary_along_time():
    first = _scene("a", 1, platform="s2a", **{"eo:cloud_cover": 10.0})
    second = _scene("b", 2, platform="s2a", **{"eo:cloud_cover": 30.0})

    out = stamp_properties(_stepped(first, second), _source(first, second))

    assert out["platform"].dims == ()
    assert out["platform"].item() == "s2a"
    assert out["eo:cloud_cover"].dims == ("time",)
    assert out["eo:cloud_cover"].values.tolist() == [10.0, 30.0]
    assert out.gs.stac_properties == {
        "platform": "platform",
        "eo:cloud_cover": "eo:cloud_cover",
    }


def test_properties_follow_step_order_not_item_order():
    first = _scene("a", 1, **{"eo:cloud_cover": 10.0})
    second = _scene("b", 2, **{"eo:cloud_cover": 30.0})

    out = stamp_properties(_stepped(second, first), _source(first, second))

    assert out["eo:cloud_cover"].values.tolist() == [30.0, 10.0]


def test_properties_mark_missing_values_per_item():
    first = _scene("a", 1, **{"s2:tile": "T1", "sat:orbit": 5})
    second = _scene("b", 2)

    out = stamp_properties(_stepped(first, second), _source(first, second))

    assert out["s2:tile"].values.tolist() == ["T1", None]
    assert out["s2:tile"].dtype == object
    assert out["sat:orbit"].dtype == float
    assert out["sat:orbit"].values[0] == 5
    assert np.isnan(out["sat:orbit"].values[1])


def test_properties_of_fused_steps_keep_only_shared():
    first = _scene("a", 1, platform="s2a", **{"eo:cloud_cover": 10.0})
    second = _scene("b", 2, platform="s2a", **{"eo:cloud_cover": 30.0})
    fused = _stepped(first).assign_coords(time=[np.datetime64("2025-01-01")])

    out = stamp_properties(fused, _source(first, second))

    assert out["platform"].item() == "s2a"
    assert "eo:cloud_cover" not in out.coords


def test_properties_of_a_timeless_load_keep_only_shared():
    first = _scene("a", 1, platform="s2a", orbit=1)
    second = _scene("b", 2, platform="s2a", orbit=2)
    loaded = xr.Dataset({"red": (("y", "x"), [[1]])})

    out = stamp_properties(loaded, _source(first, second))

    assert out["platform"].item() == "s2a"
    assert "orbit" not in out.coords


def test_properties_skip_nested_values():
    first = _scene("a", 1, platform="s2a", instruments=["msi"], extra={"a": 1})

    out = stamp_properties(_stepped(first), _source(first))

    assert "platform" in out.coords
    assert "instruments" not in out.coords
    assert "extra" not in out.coords


def test_properties_exclude_dates_grid_and_bands():
    first = _scene(
        "a",
        1,
        platform="s2a",
        **{"proj:epsg": 32633, "eo:bands": [{"name": "red"}], "created": "x"},
    )

    out = stamp_properties(_stepped(first), _source(first))

    assert set(out.gs.stac_properties) == {"platform"}
    assert "datetime" not in out.coords


def test_collection_stamp_states_collection_and_extensions():
    first = _scene("a", 1)
    first.stac_extensions.append(
        "https://stac-extensions.github.io/eo/v1.1.0/schema.json"
    )
    collection = pystac.Collection(
        id="local",
        description="d",
        extent=pystac.Extent(
            pystac.SpatialExtent([[0, 0, 1, 1]]),
            pystac.TemporalExtent([[None, None]]),
        ),
        license="CC-BY-4.0",
    )

    out = stamp_collection(_stepped(first), _source(first, collection=collection))

    assert out.gs.stac_collection.id == "local"
    assert out.gs.stac_collection.license == "CC-BY-4.0"
    assert "title" not in out.attrs
    assert json.loads(out.attrs["stac_extensions"]) == first.stac_extensions
    assert "collection" in STAMPS
    assert "properties" in STAMPS
