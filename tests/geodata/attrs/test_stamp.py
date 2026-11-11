"""Writing one attr group onto an xarray object."""

from __future__ import annotations

import pytest
import xarray as xr

from geosave.geodata import attrs
from tests.geodata.attrs.conftest import _band


def test_stamping_replaces_every_spelling_and_keeps_other_attrs() -> None:
    band = _band(nodata=5)

    stamped = attrs.stamp_attrs(band, attrs.Packing(fill_value=0))

    assert stamped.attrs == {"units": "1", "_FillValue": 0}
    assert band.attrs == {"units": "1", "nodata": 5}
    assert stamped.x.attrs == {"axis": "X"}


def test_clearing_removes_the_group() -> None:
    stamped = attrs.clear_attrs(_band(_FillValue=0), attrs.Packing)

    assert stamped.attrs == {"units": "1"}


def test_stamping_a_lazy_band_computes_nothing() -> None:
    import dask.array as da
    from dask.callbacks import Callback

    band = xr.DataArray(da.zeros((4, 4), chunks=2, dtype="uint8"), dims=("y", "x"))

    with Callback(pretask=lambda *_: pytest.fail("stamping computed pixels")):
        stamped = attrs.stamp_attrs(band, attrs.Packing(fill_value=0))
        assert attrs.Packing.from_attrs(stamped.attrs) == attrs.Packing(fill_value=0)

    assert stamped.data is band.data
