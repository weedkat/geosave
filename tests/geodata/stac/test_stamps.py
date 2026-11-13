"""Tests for the load stamp registry."""

from collections.abc import Iterator

from datetime import UTC, datetime

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
