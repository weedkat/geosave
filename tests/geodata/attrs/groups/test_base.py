"""What an attr group is: its keys, and the attrs it writes."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs


@pytest.mark.parametrize(
    "value",
    [
        attrs.CFFlags.model_validate({"classes": {0: "bg"}}),
        attrs.EOBand(common_name="red", center_wavelength=None),
        attrs.Display(rgb=None),
    ],
    ids=lambda value: type(value).__name__,
)
def test_a_field_given_as_none_writes_no_attr(value) -> None:
    # netCDF refuses None attrs, so a group never writes one.
    assert None not in value.to_attrs().values()


@pytest.mark.parametrize(
    "group,stored",
    [
        (
            attrs.CFPacking,
            {
                "scale_factor": np.float32(1e-4),
                "add_offset": np.float32(0),
                "_FillValue": np.uint16(0),
            },
        ),
        (
            attrs.CFFlags,
            {"flag_values": np.array([0, 1], "uint8"), "flag_meanings": "bg palm"},
        ),
        (attrs.CFPacking, {"valid_range": np.array([1, 100], "int16")}),
    ],
)
def test_unchanged_numpy_attrs_are_written_as_stored(group, stored):
    written = group.from_attrs(stored).to_attrs()

    assert written.keys() == stored.keys()
    for key, value in stored.items():
        assert type(written[key]) is type(value)
        np.testing.assert_array_equal(written[key], value)
        assert np.asarray(written[key]).dtype == np.asarray(value).dtype


def test_stamped_packing_decodes_to_the_stored_float_dtype():
    band = xr.DataArray(
        np.array([1000, 0], "uint16"),
        dims="x",
        attrs={"scale_factor": np.float32(1e-4), "_FillValue": np.uint16(0)},
    )
    stamped = attrs.stamp_attrs(band.copy(), band.gs.packing)

    decoded = xr.decode_cf(stamped.to_dataset(name="band")).band
    assert decoded.dtype == np.float32


def test_changed_numpy_attr_is_written_as_the_new_value():
    packing = attrs.CFPacking.from_attrs(
        {"scale_factor": np.float32(1e-4), "_FillValue": np.uint16(0)}
    )
    changed = packing.model_copy(update={"fill_value": 5})

    assert changed.to_attrs() == {"scale_factor": np.float32(1e-4), "_FillValue": 5}
