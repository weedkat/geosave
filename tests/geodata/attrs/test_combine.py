"""Native xarray attr policies with GeoSave drop warnings."""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs
from geosave.geodata.warnings import DroppedAttrsWarning


def test_callback_keeps_agreeing_attrs_without_reading_pixels() -> None:
    import dask.array as da
    from dask.delayed import delayed

    @delayed
    def pixels():
        raise AssertionError("combining attrs must not compute pixels")

    source = xr.DataArray(
        da.from_delayed(pixels(), shape=(1,), dtype="uint16"),
        dims="time",
        attrs={"_FillValue": 0, "units": "1"},
    )
    joined = xr.concat(
        [source, source.copy()], "time", combine_attrs=attrs.combine_attrs
    )

    assert joined.attrs == source.attrs
    assert isinstance(joined.data, da.Array)


def test_conflicts_warn_without_discarding_other_fields_of_a_group() -> None:
    first = {"title": "first", "license": "CC0", "scale_factor": 0.1}
    second = {"title": "second", "license": "CC0", "scale_factor": 0.2}

    with pytest.warns(DroppedAttrsWarning, match="scale_factor.*title"):
        joined = attrs.combine_attrs([first, second])

    assert joined == {"license": "CC0"}


def test_keys_missing_from_some_inputs_are_kept_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DroppedAttrsWarning)
        joined = attrs.combine_attrs([{"_FillValue": 0, "note": "a"}, {}])

    assert joined == {"_FillValue": 0, "note": "a"}


def test_nan_fills_agree() -> None:
    joined = attrs.combine_attrs(
        [{"_FillValue": float("nan")}, {"_FillValue": np.float32("nan")}]
    )

    assert np.isnan(joined["_FillValue"])


def test_custom_policy_receives_xarray_context() -> None:
    context = object()

    def choose_last(mappings, context):
        assert context is expected
        return dict(mappings[-1])

    expected = context
    joined = attrs.combine_attrs(
        [{"title": "first"}, {"title": "last"}], context, policy=choose_last
    )

    assert joined == {"title": "last"}


@pytest.mark.parametrize(
    "policy", [attrs.combine_attrs, partial(attrs.combine_attrs, policy="drop")]
)
def test_warning_callback_as_policy_reports_each_omission_once(policy):
    with pytest.warns(DroppedAttrsWarning) as recorded:
        combined = attrs.combine_attrs(
            [{"title": "first"}, {"title": "last"}], policy=policy
        )

    assert combined == {}
    assert len(recorded) == 1


def test_custom_policy_omissions_warn_and_preserve_its_result():
    def choose_units(mappings, context):
        return {"units": mappings[0]["units"]}

    with pytest.warns(DroppedAttrsWarning, match="title"):
        combined = attrs.combine_attrs(
            [{"units": "m", "title": "scene"}], policy=choose_units
        )

    assert combined == {"units": "m"}
