"""Item properties every Item of a load shares, as coordinates."""

import xarray as xr

from geosave.geodata.stac.stamps import LoadSource
from geosave.geodata.stac.stamps.properties import stamp_properties
from tests.geodata.stac.stamps.conftest import pixels, scene


def test_shared_properties_become_marked_scalar_coordinates() -> None:
    first = scene("a", 1, platform="s2a", **{"eo:cloud_cover": 10.0})
    second = scene("b", 2, platform="s2a", **{"eo:cloud_cover": 30.0})

    out = stamp_properties(pixels("red"), LoadSource([first, second]))

    assert out["platform"].dims == ()
    assert out["platform"].item() == "s2a"
    assert out.gs.stac_properties == {"platform": "platform"}


def test_a_property_some_items_lack_is_left_out() -> None:
    first = scene("a", 1, platform="s2a", **{"s2:tile": "T1"})
    second = scene("b", 2, platform="s2a")

    out = stamp_properties(pixels("red"), LoadSource([first, second]))

    assert "s2:tile" not in out.coords


def test_equal_values_of_different_types_are_not_shared() -> None:
    first = scene("a", 1, flag=1)
    second = scene("b", 2, flag=True)

    out = stamp_properties(pixels("red"), LoadSource([first, second]))

    assert "flag" not in out.coords


def test_one_item_states_all_its_scalar_properties() -> None:
    item = scene("a", 1, platform="s2a", **{"eo:cloud_cover": 10.0})

    out = stamp_properties(xr.Dataset(), LoadSource([item]))

    assert out["eo:cloud_cover"].item() == 10.0
    assert out["platform"].item() == "s2a"


def test_nested_values_are_left_out() -> None:
    item = scene("a", 1, platform="s2a", instruments=["msi"], extra={"a": 1})

    out = stamp_properties(pixels("red"), LoadSource([item]))

    assert set(out.gs.stac_properties) == {"platform"}


def test_dates_grid_and_bands_are_left_out() -> None:
    item = scene(
        "a",
        1,
        platform="s2a",
        **{"proj:epsg": 32633, "eo:bands": [{"name": "red"}], "created": "x"},
    )

    out = stamp_properties(pixels("red"), LoadSource([item]))

    assert set(out.gs.stac_properties) == {"platform"}
