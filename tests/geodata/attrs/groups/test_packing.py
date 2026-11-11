from __future__ import annotations

import numpy as np
import xarray as xr
import pytest

from geosave.geodata import attrs
from tests.geodata.attrs.conftest import _band


def test_a_zero_scale_is_refused() -> None:
    with pytest.raises(ValueError):
        attrs.Packing(scale_factor=0)


def test_spellings_that_disagree_are_refused() -> None:
    with pytest.raises(ValueError, match="_FillValue as 0 but nodata as 1"):
        attrs.Packing.model_validate({"_FillValue": 0, "nodata": 1})


@pytest.mark.parametrize("fill", [-3.40282e38, 0.1])
def test_a_float_fill_masks_a_band_of_lower_precision(fill) -> None:

    band = xr.DataArray(
        np.array([fill, 1], "float32"), dims="x", attrs={"_FillValue": fill}
    )

    assert attrs.Packing.from_attrs(band.attrs) == attrs.Packing(fill_value=fill)
    assert np.isnan(band.gs.mask_and_scale().values[0])


def test_a_band_stating_none_of_a_groups_keys_parses_to_none() -> None:
    assert attrs.Packing.from_attrs(_band().attrs) is None


def test_text_and_numpy_spellings_parse_as_native_values() -> None:
    assert attrs.Packing.from_attrs(_band(_FillValue="0").attrs) == attrs.Packing(
        fill_value=0
    )
    parsed = attrs.Packing.from_attrs(_band("uint16", _FillValue=np.uint16(0)).attrs)
    assert parsed is not None and type(parsed.fill_value) is int
