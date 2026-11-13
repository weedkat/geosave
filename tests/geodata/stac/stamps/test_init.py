"""The stamp registry: names, order and project stamps."""

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
    """Register two test stamps writing attr `a`, and remove them afterwards."""

    @register_stamp("first")
    def first(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
        return loaded.assign_attrs(a="first", b="kept")

    @register_stamp("second")
    def second(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
        return loaded.assign_attrs(a="second")

    yield
    del STAMPS["first"]
    del STAMPS["second"]


def test_every_default_stamp_is_registered() -> None:
    check_stamps(DEFAULT_STAMPS)


def test_stamps_run_in_order_and_later_overrides(registered) -> None:
    out = run_stamps(xr.Dataset(), LoadSource([]), ["first", "second"])

    assert out.attrs == {"a": "second", "b": "kept"}


def test_unknown_stamp_names_the_registered_ones() -> None:
    with pytest.raises(KeyError, match="registered") as error:
        check_stamps(["raster", "nope"])

    assert "nope" in str(error.value)
    assert "raster" in str(error.value)


def test_an_unknown_name_runs_no_stamp(registered) -> None:
    with pytest.raises(KeyError):
        run_stamps(xr.Dataset(), LoadSource([]), ["first", "nope"])


def test_a_taken_name_cannot_be_registered() -> None:
    with pytest.raises(ValueError, match="raster"):

        @register_stamp("raster")
        def again(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
            return loaded
