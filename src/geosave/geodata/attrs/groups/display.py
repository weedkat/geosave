"""How a raster draws: its RGB composite and its class colours."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, cast

import orjson
from pydantic import BeforeValidator, field_serializer

from geosave.geodata.utils.color import CATEGORY_PALETTE, parse_color

from .base import AttrsGroup, Decoded, decode


def _normalize_colors(colors: object) -> object:
    """Spell every colour `#RRGGBB`, so one palette compares equal however it was given."""
    colors = decode(colors)
    if not isinstance(colors, Mapping):
        return colors
    return {
        int(value): "#%02X%02X%02X" % parse_color(colors[value])
        for value in sorted(colors, key=int)
    }


# Hex, RGB, or GDAL's JSON text of either, read as hex.
type Hexes = Annotated[dict[int, str], BeforeValidator(_normalize_colors)]


class Display(AttrsGroup):
    """Say how a raster draws, independently of any file format.

    `rgb` lives on a Dataset's attrs, `color_map` on a class variable's. The
    GeoTIFF writer maps them to GDAL colour interpretations and colour tables.

    Attributes:
        rgb: Variables drawing red, green and blue, in that order.
        color_map: `{pixel value: hex or RGB}`. Read back as `"#RRGGBB"` and
            written as JSON text, which NetCDF, Zarr and GeoTIFF tags all hold.

    Examples:
        >>> attrs.Display.from_attrs(ds.attrs).rgb
        ['B04', 'B03', 'B02']
    """

    rgb: Decoded[list[str]] | None = None
    color_map: Hexes | None = None

    def colors(self, classes: Mapping[int, str]) -> dict[int, str]:
        """Return the colour of every class, in class order.

        A class takes its `color_map` colour, else the `CATEGORY_PALETTE`
        colour at its position among the classes, so it keeps that colour
        whichever other classes carry their own.

        Examples:
            >>> Display(color_map={0: "#0000ff"}).colors({0: "water", 1: "trees"})[0]
            '#0000FF'
        """
        color_map = self.color_map or {}
        colors = {}
        for position, code in enumerate(classes):
            if code in color_map:
                colors[code] = color_map[code]
            else:
                colors[code] = CATEGORY_PALETTE[position % len(CATEGORY_PALETTE)]
        return cast("dict[int, str]", _normalize_colors(colors))

    @field_serializer("rgb", "color_map")
    def _serialize_as_json(self, value: object) -> str | None:
        """Write a field as JSON text, since NetCDF attrs cannot hold a mapping."""
        if value is None:
            return None
        return orjson.dumps(value, option=orjson.OPT_NON_STR_KEYS).decode()
