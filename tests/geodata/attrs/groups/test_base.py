"""What an attr group is: its keys, and the attrs it writes."""

from __future__ import annotations

import pytest

from geosave.geodata import attrs


@pytest.mark.parametrize(
    "value",
    [
        attrs.Legend.model_validate({"classes": {0: "bg"}, "color_map": None}),
        attrs.Spectral(common_name="red", center_wavelength=None),
        attrs.GDALBand(variable_name="red", colorinterp=None),
        attrs.TimeSpec(time_freq="MS", time_offset=None),
    ],
    ids=lambda value: type(value).__name__,
)
def test_a_field_given_as_none_writes_no_attr(value) -> None:
    # netCDF refuses None attrs, so a group never writes one.
    assert None not in value.to_attrs().values()
