from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs


def test_nodata_is_spelled_as_cf_fill_value_only() -> None:
    assert attrs.Nodata(value=0).model_dump(by_alias=True) == {"_FillValue": 0}


def test_odcs_spelling_reads_as_the_same_nodata() -> None:
    assert attrs.Nodata.model_validate({"nodata": 7}) == attrs.Nodata(value=7)


def test_spellings_that_disagree_are_refused() -> None:
    with pytest.raises(ValueError, match="disagree"):
        attrs.Nodata.model_validate({"_FillValue": 0, "nodata": 1})


def _band(dtype: str) -> xr.DataArray:
    return xr.DataArray(np.zeros(2, dtype), dims="x")


def test_nan_marks_absence_in_a_float_band_only() -> None:
    nan = attrs.Nodata(value=float("nan"))
    assert np.isnan(nan.held_by(_band("float32")).value)
    with pytest.raises(ValueError, match="uint16"):
        nan.held_by(_band("uint16"))


def test_the_keys_cover_what_it_writes() -> None:
    assert set(attrs.Nodata(value=0).model_dump(by_alias=True)) <= set(
        attrs.Nodata.KEYS
    )


def test_nodata_survives_a_zarr_round_trip(tmp_path) -> None:
    band = xr.DataArray(np.zeros((2, 2), "float32"), dims=("y", "x"), name="b")
    band.attrs = attrs.Nodata(value=float("nan")).model_dump(by_alias=True)

    band.to_dataset().to_zarr(tmp_path / "b.zarr", mode="w")
    back = xr.open_zarr(tmp_path / "b.zarr", mask_and_scale=False)["b"]

    assert np.isnan(attrs.Nodata.model_validate(dict(back.attrs)).value)


@pytest.mark.parametrize("fill", [-3.40282e38, 0.1])
def test_a_float_fill_reads_at_the_bands_precision_and_masks(fill) -> None:
    from geosave.geodata.transform.nodata import to_nan

    band = xr.DataArray(
        np.array([fill, 1], "float32"), dims="x", attrs={"_FillValue": fill}
    )

    assert attrs.parse(band, attrs.Nodata) == attrs.Nodata(
        value=float(np.float32(fill))
    )
    assert np.isnan(to_nan(band).values[0])


def test_a_fill_overflowing_a_float_band_is_refused() -> None:
    with pytest.raises(ValueError, match="overflows"):
        attrs.Nodata(value=1e39).held_by(_band("float32"))


def test_an_integer_band_refuses_a_fill_it_would_wrap() -> None:
    with pytest.raises(ValueError, match="65535"):
        attrs.Nodata(value=-1).held_by(_band("uint16"))


def test_stamping_writes_the_fill_as_the_band_stores_it() -> None:
    stamped = attrs.stamp(_band("float32"), attrs.Nodata, attrs.Nodata(value=0.1))

    assert stamped.attrs["_FillValue"] == float(np.float32(0.1))
