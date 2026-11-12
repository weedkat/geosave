"""How a raster draws: class colours and its RGB composite."""

from __future__ import annotations

from geosave.geodata import attrs


def test_a_palette_reads_alike_however_its_colours_are_spelled() -> None:
    hexed = attrs.Display(color_map={0: "#000000", 1: "#00ff00"})
    rgb = attrs.Display(color_map={0: (0, 0, 0), 1: [0, 255, 0]})

    assert hexed == rgb
    assert rgb.color_map == {0: "#000000", 1: "#00FF00"}


def test_gdal_text_colours_read_as_hex() -> None:
    display = attrs.Display.model_validate(
        {"color_map": '{"0": [0, 0, 0], "1": [46, 139, 87]}'}
    )

    assert display.color_map == {0: "#000000", 1: "#2E8B57"}


def test_colors_keep_given_colours_and_fill_the_rest_by_position() -> None:
    from geosave.geodata.utils.color import CATEGORY_PALETTE

    display = attrs.Display(color_map={1: "#00ff00"})

    colors = display.colors({0: "water", 1: "trees", 2: "crops"})

    assert colors[1] == "#00FF00"
    assert colors[0] == CATEGORY_PALETTE[0].upper()
    assert colors[2] == CATEGORY_PALETTE[2].upper()


def test_rgb_round_trips_through_attrs() -> None:
    display = attrs.Display(rgb=["B04", "B03", "B02"])

    assert attrs.Display.from_attrs(display.to_attrs()) == display
