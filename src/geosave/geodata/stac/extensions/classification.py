"""Classification extension: the classes a band's pixel values name."""

from __future__ import annotations

import re
from typing import TypedDict, cast

import pystac
from pystac.extensions.classification import Classification, ClassificationExtension
from pystac.extensions.raster import RasterBand

import geosave.geodata.attrs as attrs
from geosave.geodata.utils.color import parse_color

# What the schema lets a class be named.
_CLASS_NAME = re.compile(r"[0-9A-Za-z_-]+")


def build_classes(legend: attrs.Legend | None) -> list[Classification] | None:
    """Build the classes a Raster band lists from its legend.

    Args:
        legend: Legend of the band's variable or file band.

    Returns:
        Its classes, or None where there is no legend or it lists bit masks,
        whose offset, length and names CF does not state.

    Raises:
        ValueError: A class name carries a character the schema refuses.

    Examples:
        >>> [c.name for c in build_classes(label.label.gs.legend)]
        ['background', 'forest']
    """
    if legend is None or legend.classes is None or legend.flag_masks is not None:
        return None
    return _classes(legend)


class ClassificationAttrs(TypedDict, total=False):
    """CF enumeration and serialized colors translated from class metadata."""

    flag_values: list[int]
    flag_meanings: str
    color_map: str


def read_attrs(band: RasterBand) -> ClassificationAttrs:
    """Translate a Raster band's classification into CF legend attrs."""
    classes = ClassificationExtension.ext(band).classes
    if not classes:
        return {}
    colors = {
        entry.value: f"#{entry.color_hint}"
        for entry in classes
        if entry.color_hint is not None
    }
    # `classes` is spelled into CF codes and names by Legend's own validator.
    legend = attrs.Legend.model_validate(
        {
            "classes": {entry.value: entry.name for entry in classes},
            "color_map": colors or None,
        }
    )
    return cast("ClassificationAttrs", legend.to_attrs())


def _classes(legend: attrs.Legend) -> list[Classification]:
    """Spell a legend's classes as Classification classes."""
    colors = legend.color_map or {}
    classes = []
    for value, name in (legend.classes or {}).items():
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
    asset: pystac.Asset, band: RasterBand, legend: attrs.Legend | None
) -> None:
    """Attach a legend's classes to a band and declare the schema on its owner."""
    if classes := build_classes(legend):
        ClassificationExtension.ext(band).apply(classes=classes)
        ClassificationExtension.ext(asset, add_if_missing=True)
