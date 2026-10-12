"""Reading one attr group off an xarray object."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs


def _timestep(**attributes) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((1, 2), "uint16"), dims=("time", "x"), attrs=attributes
    )


def _band(dtype="uint8", **attributes) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((2, 2), dtype),
        dims=("y", "x"),
        coords={"x": xr.DataArray([0, 1], dims="x", attrs={"axis": "X"})},
        attrs={"units": "1", **attributes},
    )


def test_a_band_stating_none_of_a_groups_keys_parses_to_none() -> None:
    assert attrs.parse(_band(), attrs.Nodata) is None


def test_parsing_validates_against_the_band() -> None:
    with pytest.raises(ValueError, match="uint8"):
        attrs.parse(_band(_FillValue=float("nan")), attrs.Nodata)


def test_text_and_numpy_spellings_parse_as_native_values() -> None:
    assert attrs.parse(_band(_FillValue="0"), attrs.Nodata) == attrs.Nodata(value=0)
    parsed = attrs.parse(_band("uint16", _FillValue=np.uint16(0)), attrs.Nodata)
    assert parsed is not None and type(parsed.value) is int


def test_a_dataset_root_parses_without_a_dtype() -> None:
    ds = xr.Dataset(attrs={"title": "Forest", "license": "CC0"})

    assert attrs.parse(ds, attrs.ACDD) == attrs.ACDD(title="Forest", license="CC0")
