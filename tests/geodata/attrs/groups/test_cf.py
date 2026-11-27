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


def test_spellings_that_disagree_are_refused() -> None:
    with pytest.raises(ValueError, match="_FillValue as 0 but nodata as 1"):
        attrs.CFPacking.model_validate({"_FillValue": 0, "nodata": 1})


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
    class_ids = np.array([[0, 1, 0, 1]] * 4, "uint8")
    return build_dataset({"label": (("y", "x"), class_ids)}, grid).gs.write_flags(
        {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}
    )


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


def test_colour_hints_read_alike_however_they_are_spelled() -> None:
    classes = {0: "water", 1: "trees"}
    hexed = attrs.CFFlags(classes=classes, colors={0: "#000000", 1: "#00ff00"})
    channels = attrs.CFFlags(classes=classes, colors={0: (0, 0, 0), 1: [0, 255, 0]})
    gdal_text = attrs.CFFlags.model_validate(
        {
            "flag_values": [0, 1],
            "flag_meanings": "water trees",
            "color_hints": '[[0, 0, 0], "00ff00"]',
        }
    )

    assert hexed == channels == gdal_text
    assert hexed.color_hints == ["000000", "00FF00"]


def test_classes_without_a_colour_take_their_palette_colour() -> None:
    from geosave.geodata.utils.color import CATEGORY_PALETTE

    flags = attrs.CFFlags(
        classes={0: "water", 1: "trees", 2: "crops"}, colors={1: "#00ff00"}
    )

    assert flags.colors[1] == "#00FF00"
    assert flags.colors[0] == CATEGORY_PALETTE[0].upper()
    assert flags.colors[2] == CATEGORY_PALETTE[2].upper()


def test_one_colour_hint_per_class_is_required() -> None:
    with pytest.raises(ValueError, match="colour hints"):
        attrs.CFFlags(
            flag_values=[0, 1], flag_meanings="water trees", color_hints=["000000"]
        )
