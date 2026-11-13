"""Tests for the load stamp registry."""

from collections.abc import Iterator

import pytest
import xarray as xr

from geosave.geodata.stac.stamps import (
    DEFAULT_STAMPS,
    STAMPS,
    LoadSource,
    check_stamps,
    register_stamp,
    run_stamps,
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
