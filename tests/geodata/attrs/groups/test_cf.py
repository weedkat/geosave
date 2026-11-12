"""CF variable attribute groups: description, packing and flags."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs
from tests.conftest import write_cogs
from tests.geodata.attrs.conftest import _band
from tests.geodata.conftest import flat


def _described(**band_attrs: object) -> xr.DataArray:
    return xr.DataArray(np.zeros((2, 2), "uint16"), dims=("y", "x"), attrs=band_attrs)


def test_description_reads_cf_chapter_3_attrs() -> None:
    red = _described(
        long_name="Red reflectance",
        standard_name="surface_bidirectional_reflectance",
        units="1",
        cell_methods="time: mean",
    )

    found = red.gs.description

    assert found == attrs.CFDescription(
        long_name="Red reflectance",
        standard_name="surface_bidirectional_reflectance",
        units="1",
    )


def test_a_band_stating_none_has_no_description() -> None:
    assert _described(cell_methods="time: mean").gs.description is None


def test_a_zero_scale_is_refused() -> None:
    with pytest.raises(ValueError):
        attrs.CFPacking(scale_factor=0)


def test_spellings_that_disagree_are_refused() -> None:
    with pytest.raises(ValueError, match="_FillValue as 0 but nodata as 1"):
        attrs.CFPacking.model_validate({"_FillValue": 0, "nodata": 1})


def test_a_band_stating_none_of_a_groups_keys_parses_to_none() -> None:
    assert attrs.CFPacking.from_attrs(_band().attrs) is None


def test_text_and_numpy_spellings_parse_as_native_values() -> None:
    assert attrs.CFPacking.from_attrs(_band(_FillValue="0").attrs) == attrs.CFPacking(
        fill_value=0
    )
    parsed = attrs.CFPacking.from_attrs(_band("uint16", _FillValue=np.uint16(0)).attrs)
    assert parsed is not None and type(parsed.fill_value) is int


def test_gdal_text_flags_read_as_native_values() -> None:
    flags = attrs.CFFlags.model_validate(
        {"flag_values": "[0, 1]", "flag_meanings": "bg palm"}
    )

    assert flags.classes == {0: "bg", 1: "palm"}


def test_a_bitfield_lists_masks_and_no_classes() -> None:
    legend = attrs.CFFlags.model_validate(
        {"flag_masks": [1, 2], "flag_meanings": "cloud shadow"}
    )

    assert legend.classes is None


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"flag_values": [0, 1], "flag_meanings": "bg"}, "2 pixel values"),
        ({"flag_values": [0, 1]}, "names no classes"),
        ({"flag_meanings": "bg"}, "lists no pixel values"),
        ({"flag_values": [1, 0], "flag_meanings": "a b"}, "ascending"),
        ({"classes": {0: "two words"}}, "contain spaces"),
    ],
)
def test_a_listing_that_does_not_line_up_is_refused(given, match) -> None:
    with pytest.raises(ValueError, match=match):
        attrs.CFFlags.model_validate(given)


def _labelled():
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataset

    grid = GeoBox.from_bbox((0, 0, 40, 40), "EPSG:32633", resolution=10)
    codes = np.array([[0, 1, 0, 1]] * 4, "uint8")
    return build_dataset({"label": (("y", "x"), codes)}, grid).gs.write_flags(
        {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}
    )


def test_a_geotiff_read_label_joins_one_built_in_memory(tmp_path) -> None:
    from geosave.geodata import open_raster

    labelled = _labelled()
    (file,) = flat(write_cogs(labelled, tmp_path, format_paths="label.tif"))
    restored = open_raster(file)

    joined = attrs.combine_attrs([restored.label.attrs, labelled.label.attrs])

    assert attrs.CFFlags.from_attrs(joined) == labelled.label.gs.flags


@pytest.mark.parametrize("suffix", ["zarr", "nc", "tif"])
def test_flags_and_colours_survive_each_raster_format(tmp_path, suffix) -> None:
    from geosave.geodata import open_raster

    labelled = _labelled()
    path = tmp_path / f"label.{suffix}"
    if suffix == "zarr":
        written = labelled.gs.to_zarr(path)
    elif suffix == "nc":
        written = labelled.gs.to_netcdf(path)
    else:
        (file,) = flat(write_cogs(labelled, path.parent, format_paths=path.name))
        written = str(file)

    restored = open_raster(written).label
    assert restored.gs.flags == labelled.label.gs.flags
    assert restored.gs.display == labelled.label.gs.display


def test_combined_flags_keep_repeated_codes_and_native_mask_meanings():
    import xarray as xr

    legend = attrs.CFFlags(
        flag_values=[1, 1], flag_masks=[1, 3], flag_meanings="bit_set code_one"
    )
    assert legend.classes is None
    flags = xr.DataArray([0, 1, 2, 3], dims="pixel", attrs=legend.to_attrs())
    assert (flags.cf == "bit_set").values.tolist() == [False, True, False, True]
    assert (flags.cf == "code_one").values.tolist() == [False, True, False, False]
