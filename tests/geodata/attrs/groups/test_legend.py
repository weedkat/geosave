from __future__ import annotations

import numpy as np
import pytest

from geosave.geodata import attrs
from tests.geodata.conftest import flat


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
    assert legend.color_map == {0: "#000000", 1: "#2E8B57"}


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


def test_a_palette_reads_alike_however_its_colours_are_spelled() -> None:
    hexed = attrs.Legend(
        classes={0: "bg", 1: "palm"}, color_map={0: "#000000", 1: "#00ff00"}
    )
    rgb = attrs.Legend(
        classes={0: "bg", 1: "palm"}, color_map={0: (0, 0, 0), 1: [0, 255, 0]}
    )

    assert hexed == rgb
    assert rgb.color_map == {0: "#000000", 1: "#00FF00"}


def test_a_palette_is_written_as_json_text() -> None:
    legend = attrs.Legend(
        classes={1: "palm", 0: "bg"}, color_map={1: "#00ff00", 0: (0, 0, 0)}
    )

    assert legend.to_attrs()["color_map"] == '{"0":"#000000","1":"#00FF00"}'


def _labelled():
    from odc.geo.geobox import GeoBox

    from geosave.geodata import raster

    grid = GeoBox.from_bbox((0, 0, 40, 40), "EPSG:32633", resolution=10)
    codes = np.array([[0, 1, 0, 1]] * 4, "uint8")
    return raster({"label": (("y", "x"), codes)}, grid).gs.write_legend(
        {0: "bg", 1: "palm"}, {0: "#000000", 1: "#00ff00"}
    )


def test_a_geotiff_read_label_joins_one_built_in_memory(tmp_path) -> None:
    from geosave.geodata import open_raster

    labelled = _labelled()
    (file,) = flat(labelled.gs.to_cog(str(tmp_path / "label.tif")))
    restored = open_raster(file)

    joined = attrs.combine_attrs([restored.label.attrs, labelled.label.attrs])

    assert attrs.Legend.from_attrs(joined) == labelled.label.gs.legend


@pytest.mark.parametrize("suffix", ["zarr", "nc", "tif"])
def test_a_legend_survives_each_raster_format(tmp_path, suffix) -> None:
    from geosave.geodata import open_raster

    labelled = _labelled()
    path = tmp_path / f"label.{suffix}"
    if suffix == "zarr":
        written = labelled.gs.to_zarr(path)
    elif suffix == "nc":
        written = labelled.gs.to_netcdf(path)
    else:
        (file,) = flat(labelled.gs.to_cog(str(path)))
        written = str(file)

    assert open_raster(written).label.gs.legend == labelled.label.gs.legend
