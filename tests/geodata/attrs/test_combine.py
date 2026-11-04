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


def test_a_nodata_that_differs_is_dropped_with_a_warning() -> None:
    with pytest.warns(DroppedAttrsWarning, match="_FillValue"):
        joined = xr.concat(
            [_timestep(_FillValue=0), _timestep(_FillValue=1)],
            "time",
            combine_attrs=attrs.combine_attrs,
        )

    assert joined.attrs == {}


def test_a_group_that_differs_is_dropped_whole() -> None:
    packed = {"scale_factor": 0.1, "add_offset": 5.0}
    other = {"scale_factor": 0.1, "add_offset": 6.0}

    with pytest.warns(DroppedAttrsWarning, match="add_offset"):
        assert attrs.combine_attrs([packed, other]) == {}


def test_a_descriptive_group_that_differs_is_dropped() -> None:
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


def test_nan_fills_agree() -> None:
    joined = attrs.combine_attrs(
        [{"_FillValue": float("nan")}, {"_FillValue": np.float32("nan")}]
    )

    assert np.isnan(joined["_FillValue"])


def test_a_group_joins_whole_so_a_differing_title_drops_the_license() -> None:
    first = {"title": "scene", "license": "CC0"}
    second = {"title": "other", "license": "CC0"}

    with pytest.warns(DroppedAttrsWarning, match="title"):
        joined = attrs.combine_attrs([first, second])

    assert joined == {}
