from __future__ import annotations

import pytest
import xarray as xr

from geosave_engine.geodata import attrs


def test_classes_are_spelled_as_cf_flags() -> None:
    legend = attrs.Legend(classes={1: "palm", 0: "bg"})

    assert legend.model_dump(by_alias=True, exclude_none=True) == {
        "flag_values": [0, 1],
        "flag_meanings": "bg palm",
    }
    assert legend.classes == {0: "bg", 1: "palm"}


def test_cf_flags_read_back_as_classes() -> None:
    legend = attrs.Legend.model_validate(
        {"flag_values": [0, 1], "flag_meanings": "bg palm"}
    )

    assert legend.classes == {0: "bg", 1: "palm"}


def test_gdal_text_flags_and_colours_read_as_native_values() -> None:
    legend = attrs.Legend.model_validate(
        {
            "flag_values": "[0, 1]",
            "flag_meanings": "bg palm",
            "color_map": '{"0": [0, 0, 0], "1": [46, 139, 87]}',
        }
    )

    assert legend.classes == {0: "bg", 1: "palm"}
    assert legend.color_map == {0: (0, 0, 0), 1: (46, 139, 87)}


def test_a_bitfield_lists_masks_and_no_classes() -> None:
    legend = attrs.Legend.model_validate(
        {"flag_masks": [1, 2], "flag_meanings": "cloud shadow"}
    )

    assert legend.classes is None


def test_a_legend_may_carry_colours_without_classes() -> None:
    assert attrs.Legend(color_map={0: "#000000"}).classes is None


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"flag_values": [0, 1], "flag_meanings": "bg"}, "2 codes"),
        ({"flag_values": [0, 1]}, "flag_meanings"),
        ({"flag_meanings": "bg"}, "flag_values"),
        ({"flag_values": [1, 0], "flag_meanings": "a b"}, "ascending"),
        ({"classes": {0: "two words"}}, "whitespace"),
        ({"classes": {0: "bg"}, "color_map": {1: "#000000"}}, r"\[1\]"),
    ],
)
def test_a_listing_that_does_not_line_up_is_refused(given, match) -> None:
    with pytest.raises(ValueError, match=match):
        attrs.Legend.model_validate(given)


def test_codes_must_fit_the_bands_dtype() -> None:
    import numpy as np

    band = xr.DataArray(
        np.zeros(2, "uint8"),
        dims="x",
        attrs={"flag_values": [0, 300], "flag_meanings": "bg palm"},
    )

    with pytest.raises(ValueError, match="uint8"):
        attrs.parse(band, attrs.Legend)


def test_the_keys_cover_what_it_writes() -> None:
    written = attrs.Legend(classes={0: "bg"}, color_map={0: "#000000"}).model_dump(
        by_alias=True
    )
    assert set(written) <= set(attrs.Legend.KEYS)


def test_a_float_band_holds_class_codes_exactly() -> None:
    import numpy as np

    band = xr.DataArray(np.zeros(2, "float32"), dims="x")
    legend = attrs.Legend(classes={0: "bg", 1: "palm"}).held_by(band)
    assert legend.classes == {0: "bg", 1: "palm"}
