from __future__ import annotations

import pytest
from rasterio.enums import ColorInterp

from geosave.geodata import attrs


def test_a_band_states_its_variable_and_colour_interpretation() -> None:
    band = attrs.GDALBand(variable_name="B04", colorinterp="red")

    assert band.model_dump(by_alias=True) == {
        "variable_name": "B04",
        "colorinterp": "red",
    }


def test_rasterios_enum_and_gdals_undefined_read_as_names() -> None:
    assert (
        attrs.GDALBand.model_validate({"colorinterp": ColorInterp.red}).colorinterp
        == "red"
    )
    assert (
        attrs.GDALBand.model_validate({"colorinterp": "undefined"}).colorinterp is None
    )


def test_an_unknown_interpretation_is_refused() -> None:
    with pytest.raises(ValueError, match="GDAL colour interpretation"):
        attrs.GDALBand(colorinterp="purple")
