from __future__ import annotations

from rasterio.enums import ColorInterp

from geosave.geodata import attrs


def test_rasterios_enum_and_gdals_undefined_read_as_names() -> None:
    assert (
        attrs.GDALBand.model_validate({"colorinterp": ColorInterp.red}).colorinterp
        == "red"
    )
    assert (
        attrs.GDALBand.model_validate({"colorinterp": "undefined"}).colorinterp is None
    )
