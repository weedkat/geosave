"""What an attr group is: its keys, and the attrs it writes."""

from __future__ import annotations

import pytest

from geosave.geodata import attrs


def test_every_group_is_joined() -> None:
    assert set(attrs.AttrsGroup.__subclasses__()) == {
        attrs.ACDD,
        attrs.Nodata,
        attrs.Packing,
        attrs.Quantity,
        attrs.Legend,
        attrs.Spectral,
        attrs.GDALBand,
        attrs.TimeSpec,
    }


def test_keys_are_read_off_the_fields() -> None:
    assert attrs.Nodata.KEYS == ("_FillValue", "nodata")
    assert attrs.Packing.KEYS == ("scale_factor", "add_offset")


def test_to_attrs_writes_no_field_left_at_its_default() -> None:
    assert attrs.Packing(scale_factor=0.5).to_attrs() == {"scale_factor": 0.5}


@pytest.mark.parametrize(
    "value",
    [
        attrs.Legend.model_validate({"classes": {0: "bg"}, "color_map": None}),
        attrs.Spectral(common_name="red", center_wavelength=None),
        attrs.GDALBand(variable_name="red", colorinterp=None),
        attrs.TimeSpec.from_resample("MS", offset=None),
    ],
    ids=lambda value: type(value).__name__,
)
def test_a_field_given_as_none_writes_no_attr(value) -> None:
    # netCDF refuses None attrs, so a group never writes one.
    assert None not in value.to_attrs().values()


@pytest.mark.parametrize(
    "group",
    [
        attrs.Nodata(value=0),
        attrs.Packing(scale_factor=2, add_offset=1),
        attrs.Spectral(
            common_name="red", center_wavelength=0.665, full_width_half_max=0.03
        ),
        attrs.GDALBand(variable_name="B04", colorinterp="red"),
        attrs.Legend(classes={0: "bg"}, color_map={0: "#000000"}),
    ],
    ids=lambda group: type(group).__name__,
)
def test_the_keys_cover_what_a_group_writes(group) -> None:
    assert set(group.model_dump(by_alias=True)) <= set(group.KEYS)
