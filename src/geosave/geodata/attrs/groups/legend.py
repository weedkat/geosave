"""What a label variable's pixel values mean, and how they colour."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self, cast

import numpy as np
import orjson
from pydantic import BeforeValidator, field_serializer, model_validator

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


class Legend(AttrsGroup):
    """Say what a label variable's pixel values mean, and how they colour.

    CF lists classes two ways: `flag_values` enumerates the values a pixel
    takes and `flag_masks` the bits a value packs flags into; `flag_meanings`
    names them in order. `classes` reads an enumeration back keyed by value.

    Attributes:
        flag_values: Enumeration codes, ascending and unique without masks;
            masked codes may repeat when their masks differ.
        flag_masks: Bit masks, for values packing several flags at once.
        flag_meanings: Space-separated class names, in listing order.
        color_map: `{pixel value: hex or RGB}`, keyed to listed values.
            Read back as `"#RRGGBB"` and written as JSON text, which
            NetCDF, Zarr and GeoTIFF tags all hold.

    Raises:
        ValueError: Names and codes differ in number or only one is set,
            `flag_values` is not ascending and unique, a class name carries
            whitespace, a colour keys to an unlisted value, or a code does
            not fit the band's dtype.

    Examples:
        >>> ds["label"].gs.legend.classes
        {0: 'bg', 1: 'palm'}
    """

    flag_values: Decoded[list[int]] | None = None
    flag_masks: Decoded[list[int]] | None = None
    flag_meanings: str | None = None
    color_map: Hexes | None = None

    @property
    def classes(self) -> dict[int, str] | None:
        """Return each pixel value mapped to its class, None for a bitfield or colours only."""
        if self.flag_masks is not None or self.flag_values is None or self.flag_meanings is None:
            return None
        return dict(zip(self.flag_values, self.flag_meanings.split(), strict=True))

    @property
    def colors(self) -> dict[int, str]:
        """Return the colour of every class, in class order.

        A class takes its `color_map` colour, else the `CATEGORY_PALETTE`
        colour at its position among the classes, so it keeps that colour
        whichever other classes carry their own.

        Raises:
            ValueError: No classes are listed.

        Examples:
            >>> Legend(classes={0: "water", 1: "trees"}, color_map={0: "#0000ff"}).colors[0]
            '#0000FF'
        """
        classes = self.classes
        if classes is None:
            raise ValueError(
                "The legend has no classes to color. Give it classes, for example "
                "Legend(classes={0: 'water', 1: 'trees'})."
            )
        color_map = self.color_map or {}
        colors = {}
        for position, code in enumerate(classes):
            if code in color_map:
                colors[code] = color_map[code]
            else:
                colors[code] = CATEGORY_PALETTE[position % len(CATEGORY_PALETTE)]
        return cast("dict[int, str]", _normalize_colors(colors))

    def check_dtype(self, dtype: np.dtype) -> None:
        """Refuse class values that `dtype` would store as other numbers.

        A class value names a class only as the exact number stored: 300 wraps
        to 44 on uint8, so it would name another class or none.

        Raises:
            ValueError: A class value or mask does not fit `dtype`.
        """
        values = [*(self.flag_values or ()), *(self.flag_masks or ())]
        with np.errstate(invalid="ignore", over="ignore"):
            stored = np.asarray(values).astype(dtype)
        misstored = [
            value for value, kept in zip(values, stored, strict=True) if kept != value
        ]
        if misstored:
            raise ValueError(
                f"The class values {misstored} do not fit this {dtype} band: they would "
                f"be stored as other numbers and label the wrong pixels. Use class "
                f"values the band can hold, or convert the band to a wider dtype first."
            )

    @field_serializer("color_map")
    def _serialize_color_map(self, colors: dict[int, str] | None) -> str | None:
        """Write the palette as JSON text, since NetCDF attrs cannot hold a mapping."""
        if colors is None:
            return None
        return orjson.dumps(colors, option=orjson.OPT_NON_STR_KEYS).decode()

    @model_validator(mode="before")
    @classmethod
    def _expand_classes(cls, data: Any) -> Any:
        """Turn the `classes` argument into the `flag_values` and `flag_meanings` CF stores."""
        if not isinstance(data, dict) or data.get("classes") is None:
            return data
        given = dict(data)
        classes = given.pop("classes")
        split = sorted(name for name in classes.values() if name.split() != [name])
        if split:
            raise ValueError(
                f"Class names {split} are empty or contain spaces, which the CF "
                f"flag_meanings format cannot store. Use single words such as "
                f"'tall_grass'."
            )
        values = sorted(classes)
        given["flag_values"] = values
        given["flag_meanings"] = " ".join(classes[value] for value in values)
        return given

    @model_validator(mode="after")
    def _check_names_match_values(self) -> Self:
        """Require one class name for each pixel value or bit mask listed."""
        names = [] if self.flag_meanings is None else self.flag_meanings.split()
        listings = {"pixel values": self.flag_values, "bit masks": self.flag_masks}
        listings = {label: listed for label, listed in listings.items() if listed}

        if listings and not names:
            raise ValueError(
                f"The legend lists {' and '.join(listings)} but names no classes. "
                f"Give each class a name."
            )
        if names and not listings:
            raise ValueError(
                f"The legend names the classes {names} but lists no pixel values. "
                f"Give each class the pixel value it stands for."
            )
        for label, listed in listings.items():
            if len(listed) != len(names):
                raise ValueError(
                    f"The legend lists {len(listed)} {label} but {len(names)} class "
                    f"names. Each class needs exactly one name and one of the {label}."
                )
        return self

    @model_validator(mode="after")
    def _check_values_ascending(self) -> Self:
        """Require pixel values in ascending order without repeats."""
        values = self.flag_values
        if self.flag_masks is None and values is not None and values != sorted(set(values)):
            raise ValueError(
                f"The legend's pixel values {values} must be ascending with no "
                f"repeats. Sort the classes by pixel value and remove duplicates."
            )
        return self

    @model_validator(mode="after")
    def _check_colors_match_classes(self) -> Self:
        """Require every colored pixel value to belong to a listed class."""
        if self.color_map is None or self.flag_values is None:
            return self
        uncolored = sorted(set(self.color_map) - set(self.flag_values))
        if uncolored:
            raise ValueError(
                f"The color map colors pixel values {uncolored}, but no class has "
                f"those values (the classes are {self.flag_values}). Add the "
                f"missing classes or remove their colors."
            )
        return self
