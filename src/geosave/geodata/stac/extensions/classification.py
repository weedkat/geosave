"""Classification extension: the classes a band's pixel values name."""

from __future__ import annotations

from collections.abc import Mapping

import re
from typing import TypedDict, cast

import pystac
from pystac.extensions.classification import Classification, ClassificationExtension
from pystac.extensions.raster import RasterBand

import geosave.geodata.attrs as attrs
from geosave.geodata.utils.color import parse_color

# What the schema lets a class be named.
_CLASS_NAME = re.compile(r"[0-9A-Za-z_-]+")


def build_classes(
    flags: attrs.CFFlags | None, display: attrs.Display | None = None
) -> list[Classification] | None:
    """Build the classes a Raster band lists from its CF flags and colours.

    Args:
        flags: CF flags of the band's variable or file band.
        display: Its class colours, sent as color hints.

    Returns:
        Its classes, or None where there are no flags or they list bit masks,
        whose offset, length and names CF does not state.

    Raises:
        ValueError: A class name carries a character the schema refuses.

    Examples:
        >>> [c.name for c in build_classes(label.label.gs.flags)]
        ['background', 'forest']
    """
    if flags is None or flags.classes is None or flags.flag_masks is not None:
        return None
    colors = {} if display is None else display.color_map or {}
    return _classes(flags.classes, colors)


class ClassificationAttrs(TypedDict, total=False):
    """CF enumeration and serialized colors translated from class metadata."""

    flag_values: list[int]
    flag_meanings: str
    color_map: str


def read_attrs(band: RasterBand) -> ClassificationAttrs:
    """Translate a Raster band's classification into CF flags and colour attrs."""
    classes = ClassificationExtension.ext(band).classes
    if not classes:
        return {}
    colors = {
        entry.value: f"#{entry.color_hint}"
        for entry in classes
        if entry.color_hint is not None
    }
    # `classes` is spelled into CF codes and names by CFFlags' own validator.
    flags = attrs.CFFlags.model_validate(
        {"classes": {entry.value: entry.name for entry in classes}}
    )
    found = {**flags.to_attrs(), **attrs.Display(color_map=colors or None).to_attrs()}
    return cast("ClassificationAttrs", found)


def _classes(
    named: Mapping[int, str], colors: Mapping[int, str]
) -> list[Classification]:
    """Spell classes and their colours as Classification classes."""
    classes = []
    for value, name in named.items():
        if not _CLASS_NAME.fullmatch(name):
            raise ValueError(
                f"class name {name!r} for value {value} carries a character STAC "
                f"refuses; use only letters, digits, '-' and '_'"
            )
        hint = None
        if value in colors:
            red, green, blue = parse_color(colors[value])
            hint = f"{red:02X}{green:02X}{blue:02X}"
        classes.append(Classification.create(value=value, name=name, color_hint=hint))
    return classes


def write_band(
    asset: pystac.Asset,
    band: RasterBand,
    flags: attrs.CFFlags | None,
    display: attrs.Display | None,
) -> None:
    """Attach CF flags as classes to a band and declare the schema on its owner."""
    if classes := build_classes(flags, display):
        ClassificationExtension.ext(band).apply(classes=classes)
        ClassificationExtension.ext(asset, add_if_missing=True)
