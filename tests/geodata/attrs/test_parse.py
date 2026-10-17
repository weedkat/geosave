"""Reading one attr group off an xarray object."""

from __future__ import annotations

import numpy as np
import xarray as xr

from geosave.geodata import attrs
from tests.geodata.attrs.conftest import _band


def test_a_band_stating_none_of_a_groups_keys_parses_to_none() -> None:
    assert attrs.Nodata.from_attrs(_band().attrs) is None


def test_text_and_numpy_spellings_parse_as_native_values() -> None:
    assert attrs.Nodata.from_attrs(_band(_FillValue="0").attrs) == attrs.Nodata(value=0)
    parsed = attrs.Nodata.from_attrs(_band("uint16", _FillValue=np.uint16(0)).attrs)
    assert parsed is not None and type(parsed.value) is int


def test_a_dataset_root_parses_without_a_dtype() -> None:
    ds = xr.Dataset(attrs={"title": "Forest", "license": "CC0"})

    assert attrs.ACDD.from_attrs(ds.attrs) == attrs.ACDD(title="Forest", license="CC0")
