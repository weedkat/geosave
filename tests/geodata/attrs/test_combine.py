"""Joining attrs inside xarray's own concat and merge."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs


def _band(**attributes) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((1, 2), "uint16"), dims=("time", "x"), attrs=attributes
    )


def test_agreeing_groups_and_keys_are_kept() -> None:
    a = _band(_FillValue=0, units="1", long_name="red")

    joined = xr.concat([a, a.copy()], "time", combine_attrs=attrs.combine)

    assert joined.attrs == {"_FillValue": 0, "units": "1", "long_name": "red"}


def test_one_nodata_spelled_two_ways_agrees() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        joined = attrs.combine([{"_FillValue": "0"}, {"_FillValue": 0}, {"nodata": 0}])

    assert joined == {"_FillValue": 0}


def test_an_agree_group_that_differs_refuses_the_join() -> None:
    with pytest.raises(ValueError, match="Nodata"):
        xr.concat(
            [_band(_FillValue=0), _band(_FillValue=1)],
            "time",
            combine_attrs=attrs.combine,
        )


def test_a_drop_group_that_differs_leaves_the_join() -> None:
    joined = attrs.combine(
        [{"common_name": "red", "units": "1"}, {"common_name": "nir", "units": "1"}]
    )

    assert joined == {"units": "1"}


def test_a_group_one_object_lacks_is_left_out() -> None:
    assert attrs.combine([{"_FillValue": 0}, {}]) == {}


def test_a_key_one_object_lacks_is_left_out() -> None:
    with pytest.warns(Warning, match="note"):
        assert attrs.combine([{"note": "a", "units": "1"}, {"units": "1"}]) == {
            "units": "1"
        }


def test_units_that_differ_refuse_the_join() -> None:
    with pytest.raises(ValueError, match="'units'"):
        attrs.combine([{"units": "m"}, {"units": "ft"}])


def test_plain_keys_that_differ_drop() -> None:
    assert attrs.combine(
        [{"title": "a", "units": "1"}, {"title": "b", "units": "1"}]
    ) == {"units": "1"}


def test_nan_fills_agree() -> None:
    joined = attrs.combine(
        [{"_FillValue": float("nan")}, {"_FillValue": np.float32("nan")}]
    )

    assert np.isnan(joined["_FillValue"])
