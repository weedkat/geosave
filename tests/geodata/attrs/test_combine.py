"""Joining attrs inside xarray's concat and merge."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs
from geosave.geodata.warnings import DroppedAttrsWarning


def _timestep(**attributes) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((1, 2), "uint16"), dims=("time", "x"), attrs=attributes
    )


def test_agreeing_groups_and_keys_are_kept() -> None:
    a = _timestep(_FillValue=0, units="1", long_name="red")

    joined = xr.concat([a, a.copy()], "time", combine_attrs=attrs.combine_attrs)

    assert joined.attrs == {"_FillValue": 0, "units": "1", "long_name": "red"}


def test_one_nodata_spelled_two_ways_agrees() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        joined = attrs.combine_attrs(
            [{"_FillValue": "0"}, {"_FillValue": 0}, {"nodata": 0}]
        )

    assert joined == {"_FillValue": 0}


def test_an_agree_group_that_differs_refuses_the_join() -> None:
    with pytest.raises(ValueError, match="Nodata"):
        xr.concat(
            [_timestep(_FillValue=0), _timestep(_FillValue=1)],
            "time",
            combine_attrs=attrs.combine_attrs,
        )


def test_a_drop_group_that_differs_leaves_the_join() -> None:
    joined = attrs.combine_attrs(
        [{"common_name": "red", "units": "1"}, {"common_name": "nir", "units": "1"}]
    )

    assert joined == {"units": "1"}


def test_a_group_one_object_lacks_is_left_out() -> None:
    assert attrs.combine_attrs([{"_FillValue": 0}, {}]) == {}


def test_a_key_one_object_lacks_is_left_out() -> None:
    with pytest.warns(Warning, match="note"):
        assert attrs.combine_attrs([{"note": "a", "units": "1"}, {"units": "1"}]) == {
            "units": "1"
        }


def test_units_that_differ_refuse_the_join() -> None:
    with pytest.raises(ValueError, match="Quantity"):
        attrs.combine_attrs([{"units": "m"}, {"units": "ft"}])


def test_a_quantity_one_object_leaves_unstated_drops() -> None:
    with pytest.warns(DroppedAttrsWarning, match="units"):
        assert attrs.combine_attrs([{"units": "m"}, {}]) == {}


def test_a_quantity_stated_alike_is_kept_whole() -> None:
    quantity = {"standard_name": "surface_reflectance", "units": "1"}

    assert attrs.combine_attrs([quantity, dict(quantity)]) == quantity


def test_plain_keys_that_differ_drop() -> None:
    assert attrs.combine_attrs(
        [{"title": "a", "units": "1"}, {"title": "b", "units": "1"}]
    ) == {"units": "1"}


def test_nan_fills_agree() -> None:
    joined = attrs.combine_attrs(
        [{"_FillValue": float("nan")}, {"_FillValue": np.float32("nan")}]
    )

    assert np.isnan(joined["_FillValue"])


def test_an_each_group_joins_its_keys_one_by_one() -> None:
    first = {"title": "scene", "license": "CC0"}
    second = {"title": "other", "license": "CC0"}

    with pytest.warns(DroppedAttrsWarning, match="title"):
        joined = attrs.combine_attrs([first, second])

    assert joined == {"license": "CC0"}
