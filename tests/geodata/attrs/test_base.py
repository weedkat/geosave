"""Parsing and stamping one attr group on a band."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave_engine.geodata import attrs
from geosave_engine.geodata.attrs.base import parse, stamp


def _band(dtype="uint8", **attributes) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((2, 2), dtype),
        dims=("y", "x"),
        coords={"x": xr.DataArray([0, 1], dims="x", attrs={"axis": "X"})},
        attrs={"units": "1", **attributes},
    )


def test_a_band_stating_none_of_a_groups_keys_parses_to_none() -> None:
    assert parse(_band(), attrs.Nodata) is None


def test_stamping_replaces_every_spelling_and_keeps_other_attrs() -> None:
    band = _band(nodata=5)

    stamped = stamp(band, attrs.Nodata, attrs.Nodata(value=0))

    assert stamped.attrs == {"units": "1", "_FillValue": 0}
    assert band.attrs == {"units": "1", "nodata": 5}
    assert stamped.x.attrs == {"axis": "X"}


def test_stamping_none_removes_the_group() -> None:
    stamped = stamp(_band(_FillValue=0), attrs.Nodata, None)

    assert stamped.attrs == {"units": "1"}


def test_stamping_validates_against_the_band() -> None:
    with pytest.raises(ValueError, match="uint8"):
        stamp(_band(), attrs.Nodata, attrs.Nodata(value=300))


def test_parsing_validates_against_the_band() -> None:
    with pytest.raises(ValueError, match="uint8"):
        parse(_band(_FillValue=float("nan")), attrs.Nodata)


def test_text_and_numpy_spellings_parse_as_native_values() -> None:
    assert parse(_band(_FillValue="0"), attrs.Nodata) == attrs.Nodata(value=0)
    parsed = parse(_band("uint16", _FillValue=np.uint16(0)), attrs.Nodata)
    assert parsed is not None and type(parsed.value) is int


def test_every_group_is_joined() -> None:
    assert set(attrs.AttrsGroup.__subclasses__()) == {
        attrs.Nodata,
        attrs.Packing,
        attrs.Legend,
        attrs.Spectral,
        attrs.GDALBand,
        attrs.TimeSpec,
    }


def test_keys_are_read_off_the_fields() -> None:
    assert attrs.Nodata.KEYS == ("_FillValue", "nodata")
    assert attrs.Packing.KEYS == ("scale_factor", "add_offset")


def test_to_attrs_writes_no_field_left_at_its_default() -> None:
    assert attrs.Packing(scale_factor=0.5).to_attrs() == {"scale_factor": 0.5}


def test_stamping_a_lazy_band_computes_nothing() -> None:
    import dask.array as da
    from dask.callbacks import Callback

    band = xr.DataArray(da.zeros((4, 4), chunks=2, dtype="uint8"), dims=("y", "x"))

    with Callback(pretask=lambda *_: pytest.fail("stamping computed pixels")):
        stamped = stamp(band, attrs.Nodata, attrs.Nodata(value=0))
        assert parse(stamped, attrs.Nodata) == attrs.Nodata(value=0)

    assert stamped.data is band.data
