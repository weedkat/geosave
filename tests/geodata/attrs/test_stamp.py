"""Writing one attr group onto an xarray object."""

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


def test_stamping_replaces_every_spelling_and_keeps_other_attrs() -> None:
    band = _band(nodata=5)

    stamped = attrs.stamp(band, attrs.Nodata, attrs.Nodata(value=0))

    assert stamped.attrs == {"units": "1", "_FillValue": 0}
    assert band.attrs == {"units": "1", "nodata": 5}
    assert stamped.x.attrs == {"axis": "X"}


def test_stamping_none_removes_the_group() -> None:
    stamped = attrs.stamp(_band(_FillValue=0), attrs.Nodata, None)

    assert stamped.attrs == {"units": "1"}


def test_stamping_validates_against_the_band() -> None:
    with pytest.raises(ValueError, match="uint8"):
        attrs.stamp(_band(), attrs.Nodata, attrs.Nodata(value=300))


def test_stamping_a_lazy_band_computes_nothing() -> None:
    import dask.array as da
    from dask.callbacks import Callback

    band = xr.DataArray(da.zeros((4, 4), chunks=2, dtype="uint8"), dims=("y", "x"))

    with Callback(pretask=lambda *_: pytest.fail("stamping computed pixels")):
        stamped = attrs.stamp(band, attrs.Nodata, attrs.Nodata(value=0))
        assert attrs.parse(stamped, attrs.Nodata) == attrs.Nodata(value=0)

    assert stamped.data is band.data


def test_stamping_a_dataset_root_replaces_its_group_and_keeps_other_attrs() -> None:
    ds = xr.Dataset(attrs={"title": "stale", "license": "CC0", "history": "kept"})

    stamped = attrs.stamp(ds, attrs.ACDD, attrs.ACDD(title="Forest"))

    assert stamped.attrs == {"history": "kept", "title": "Forest"}
