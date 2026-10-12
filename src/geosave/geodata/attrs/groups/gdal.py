"""What a GDAL band says about the variable it holds."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import Field, field_validator
from rasterio.enums import ColorInterp

from .base import AttrsGroup, Join

# GDAL's names for the three display channels, in the order a composite draws them.
RGB = (ColorInterp.red.name, ColorInterp.green.name, ColorInterp.blue.name)


class GDALBand(AttrsGroup):
    """Name the variable one raster band holds, and how GDAL draws it.

    GDAL's band description carries CF's `long_name`, which describes the
    pixels rather than identifying them, so a band carries its variable name
    as its own metadata item.

    Args:
        variable_name: Name of the variable this band holds.
        colorinterp: How a reader should draw the band, named as GDAL names
            it, such as `"red"` or `"palette"`. GDAL's `"undefined"` reads as
            None.

    Raises:
        ValueError: `colorinterp` names no interpretation GDAL knows.

    Examples:
        >>> attrs.parse(ds["B04"], attrs.GDALBand).colorinterp
        'red'
    """

    JOIN: ClassVar[Join] = "drop"

    variable_name: Annotated[str, Field(min_length=1)] | None = None
    colorinterp: str | None = None

    @field_validator("colorinterp", mode="before")
    @classmethod
    def _interpretation(cls, value: object) -> object:
        """Read the interpretation, however rasterio or the file spells it."""
        if isinstance(value, ColorInterp):
            value = value.name
        if value == ColorInterp.undefined.name:
            return None
        if isinstance(value, str) and value not in ColorInterp.__members__:
            raise ValueError(
                f"{value!r} names no GDAL colour interpretation; use one of "
                f"{sorted(ColorInterp.__members__)}"
            )
        return value
