from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs
from tests.geodata.attrs.conftest import _band


def test_nodata_is_spelled_as_cf_fill_value_only() -> None:
    assert attrs.Nodata(value=0).model_dump(by_alias=True) == {"_FillValue": 0}


def test_odcs_spelling_reads_as_the_same_nodata() -> None:
    assert attrs.Nodata.model_validate({"nodata": 7}) == attrs.Nodata(value=7)


def test_spellings_that_disagree_are_refused() -> None:
    with pytest.raises(ValueError, match="disagree"):
        attrs.Nodata.model_validate({"_FillValue": 0, "nodata": 1})


def test_nodata_survives_a_zarr_round_trip(tmp_path) -> None:
    band = xr.DataArray(np.zeros((2, 2), "float32"), dims=("y", "x"), name="b")
    band.attrs = attrs.Nodata(value=float("nan")).model_dump(by_alias=True)

    band.to_dataset().to_zarr(tmp_path / "b.zarr", mode="w")
    back = xr.open_zarr(tmp_path / "b.zarr", mask_and_scale=False)["b"]

    assert np.isnan(attrs.Nodata.model_validate(dict(back.attrs)).value)


@pytest.mark.parametrize("fill", [-3.40282e38, 0.1])
def test_a_float_fill_masks_a_band_of_lower_precision(fill) -> None:
    from geosave.geodata.transform.nodata import to_nan

    band = xr.DataArray(
        np.array([fill, 1], "float32"), dims="x", attrs={"_FillValue": fill}
    )

    assert attrs.Nodata.from_attrs(band.attrs) == attrs.Nodata(value=fill)
    assert np.isnan(to_nan(band).values[0])


def test_a_band_stating_none_of_a_groups_keys_parses_to_none() -> None:
    assert attrs.Nodata.from_attrs(_band().attrs) is None


def test_text_and_numpy_spellings_parse_as_native_values() -> None:
    assert attrs.Nodata.from_attrs(_band(_FillValue="0").attrs) == attrs.Nodata(value=0)
    parsed = attrs.Nodata.from_attrs(_band("uint16", _FillValue=np.uint16(0)).attrs)
    assert parsed is not None and type(parsed.value) is int
