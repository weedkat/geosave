"""How a raster draws: its RGB composite."""

from __future__ import annotations

import orjson
from pydantic import field_serializer

from .base import AttrsGroup, Decoded


class Display(AttrsGroup):
    """Say which variables draw red, green and blue, independently of any file format.

    It lives on a Dataset's attrs. The GeoTIFF writer maps it to GDAL colour
    interpretations. Class colours are `CFFlags.color_hints`, per variable.

    Attributes:
        rgb: Variables drawing red, green and blue, in that order.

    Examples:
        >>> attrs.Display.from_attrs(ds.attrs).rgb
        ['B04', 'B03', 'B02']
    """

    rgb: Decoded[list[str]] | None = None

    @field_serializer("rgb")
    def _serialize_as_json(self, value: list[str] | None) -> str | None:
        """Write the composite as JSON text, since NetCDF attrs cannot hold a list of strings."""
        if value is None:
            return None
        return orjson.dumps(value).decode()
