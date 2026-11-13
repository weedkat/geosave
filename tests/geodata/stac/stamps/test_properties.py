"""Item properties as coordinates marked `stac_property`."""

import numpy as np
import xarray as xr

from geosave.geodata.stac.stamps import LoadSource
from geosave.geodata.stac.stamps.properties import stamp_properties
from tests.geodata.stac.stamps.conftest import scene, stepped


def test_one_item_per_step_varies_along_time() -> None:
    first = scene("a", 1, platform="s2a", **{"eo:cloud_cover": 10.0})
    second = scene("b", 2, platform="s2a", **{"eo:cloud_cover": 30.0})

    out = stamp_properties(stepped(first, second), LoadSource([first, second]))

    assert out["platform"].dims == ()
    assert out["platform"].item() == "s2a"
    assert out["eo:cloud_cover"].dims == ("time",)
    assert out["eo:cloud_cover"].values.tolist() == [10.0, 30.0]
    assert out.gs.stac_properties == {
        "platform": "platform",
        "eo:cloud_cover": "eo:cloud_cover",
    }


def test_values_follow_step_order_not_item_order() -> None:
    first = scene("a", 1, **{"eo:cloud_cover": 10.0})
    second = scene("b", 2, **{"eo:cloud_cover": 30.0})

    out = stamp_properties(stepped(second, first), LoadSource([first, second]))

    assert out["eo:cloud_cover"].values.tolist() == [30.0, 10.0]


def test_missing_values_per_item() -> None:
    first = scene("a", 1, **{"s2:tile": "T1", "sat:orbit": 5})
    second = scene("b", 2)

    out = stamp_properties(stepped(first, second), LoadSource([first, second]))

    assert out["s2:tile"].values.tolist() == ["T1", None]
    assert out["sat:orbit"].dtype == float
    assert out["sat:orbit"].values[0] == 5
    assert np.isnan(out["sat:orbit"].values[1])


def test_strings_stay_whole_objects() -> None:
    first = scene("a", 1, **{"s2:tile": "T1"})
    second = scene("b", 2, **{"s2:tile": "T22"})

    out = stamp_properties(stepped(first, second), LoadSource([first, second]))

    assert out["s2:tile"].dtype == object
    assert out["s2:tile"].values.tolist() == ["T1", "T22"]


def test_equal_values_of_different_types_are_not_shared() -> None:
    first = scene("a", 1, flag=1)
    second = scene("b", 2, flag=True)

    out = stamp_properties(stepped(first, second), LoadSource([first, second]))

    assert out["flag"].dims == ("time",)


def test_fused_steps_keep_only_shared_properties() -> None:
    first = scene("a", 1, platform="s2a", **{"eo:cloud_cover": 10.0})
    second = scene("b", 2, platform="s2a", **{"eo:cloud_cover": 30.0})

    out = stamp_properties(stepped(first), LoadSource([first, second]))

    assert out["platform"].item() == "s2a"
    assert "eo:cloud_cover" not in out.coords


def test_a_timeless_load_keeps_only_shared_properties() -> None:
    first = scene("a", 1, platform="s2a", orbit=1)
    second = scene("b", 2, platform="s2a", orbit=2)
    loaded = xr.Dataset({"red": (("y", "x"), [[1]])})

    out = stamp_properties(loaded, LoadSource([first, second]))

    assert out["platform"].item() == "s2a"
    assert "orbit" not in out.coords


def test_nested_values_are_left_out() -> None:
    first = scene("a", 1, platform="s2a", instruments=["msi"], extra={"a": 1})

    out = stamp_properties(stepped(first), LoadSource([first]))

    assert set(out.gs.stac_properties) == {"platform"}


def test_dates_grid_and_bands_are_left_out() -> None:
    first = scene(
        "a",
        1,
        platform="s2a",
        **{"proj:epsg": 32633, "eo:bands": [{"name": "red"}], "created": "x"},
    )

    out = stamp_properties(stepped(first), LoadSource([first]))

    assert set(out.gs.stac_properties) == {"platform"}
