"""What a variable's values are, as CF chapter 3 names it."""

import numpy as np
import xarray as xr

from geosave.geodata import attrs


def _band(**band_attrs: object) -> xr.DataArray:
    return xr.DataArray(np.zeros((2, 2), "uint16"), dims=("y", "x"), attrs=band_attrs)


def test_description_reads_cf_chapter_3_attrs() -> None:
    red = _band(
        long_name="Red reflectance",
        standard_name="surface_bidirectional_reflectance",
        units="1",
        cell_methods="time: mean",
    )

    found = red.gs.description

    assert found == attrs.Description(
        long_name="Red reflectance",
        standard_name="surface_bidirectional_reflectance",
        units="1",
    )


def test_a_band_stating_none_has_no_description() -> None:
    assert _band(cell_methods="time: mean").gs.description is None
