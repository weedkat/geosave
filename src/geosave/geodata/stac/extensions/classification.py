"""Classification extension: the classes a band's pixel values name."""

from __future__ import annotations

from collections.abc import Mapping

import re
from typing import TypedDict, cast

import pystac
from pystac.extensions.classification import Classification, ClassificationExtension
from pystac.extensions.raster import RasterBand

import geosave.geodata.attrs as attrs

# What the schema lets a class be named.
_CLASS_NAME = re.compile(r"[0-9A-Za-z_-]+")


def build_classes(flags: attrs.CFFlags | None) -> list[Classification] | None:
    """Build the classes a Raster band lists from its CF flags and colour hints.

    Args:
        flags: CF flags of the band's variable or file band.

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
    hints = {}
    if flags.color_hints is not None:
        hints = dict(zip(flags.classes, flags.color_hints, strict=True))
    return _classes(flags.classes, hints)


class ClassificationAttrs(TypedDict, total=False):
    """CF enumeration and colour hints translated from class metadata."""

    flag_values: list[int]
    flag_meanings: str
    color_hints: str


def read_attrs(band: RasterBand) -> ClassificationAttrs:
    """Translate a Raster band's classification into CF flags and colour attrs."""
    classes = ClassificationExtension.ext(band).classes
    if not classes:
        return {}
    colors = {
        entry.value: entry.color_hint
        for entry in classes
        if entry.color_hint is not None
    }
    # CFFlags spells `classes` into CF flag values and names, and `colors` into hints.
    flags = attrs.CFFlags.model_validate(
        {"classes": {entry.value: entry.name for entry in classes}, "colors": colors}
    )
    return cast("ClassificationAttrs", flags.to_attrs())


def _classes(
    named: Mapping[int, str], hints: Mapping[int, str]
) -> list[Classification]:
    """Spell classes and their colour hints as Classification classes."""
    classes = []
    for value, name in named.items():
        if not _CLASS_NAME.fullmatch(name):
            raise ValueError(
                f"class name {name!r} for value {value} carries a character STAC "
                f"refuses; use only letters, digits, '-' and '_'"
            )
        hint = hints.get(value)
        classes.append(Classification.create(value=value, name=name, color_hint=hint))
    return classes


def write_band(
    asset: pystac.Asset,
    band: RasterBand,
    flags: attrs.CFFlags | None,
) -> None:
    """Attach CF flags as classes to a band and declare the schema on its owner."""
    if classes := build_classes(flags):
        ClassificationExtension.ext(band).apply(classes=classes)
        ClassificationExtension.ext(asset, add_if_missing=True)
