"""Classification extension: the classes a band's pixel values name."""

from __future__ import annotations

import re
import pystac

import xarray as xr
from pystac.extensions.classification import Classification, ClassificationExtension
from pystac.extensions.raster import RasterBand

import geosave.geodata.attrs as attrs
from geosave.geodata.utils.color import parse_color

# What the schema lets a class be named.
_CLASS_NAME = re.compile(r"[0-9A-Za-z_-]+")


def from_dataarray(variable: xr.DataArray) -> list[Classification] | None:
    """Describe a variable's legend as the classes its Raster band lists.

    Args:
        variable: Variable as `open_raster` opens it.

    Returns:
        Its classes, or None where it has no legend or its legend lists bit
        masks, whose offset, length and names CF does not state.

    Raises:
        ValueError: A class name carries a character the schema refuses.

    Examples:
        >>> [c.name for c in from_dataarray(label.label)]
        ['background', 'forest']
    """
    legend = variable.gs.legend
    if legend is None or legend.classes is None or legend.flag_masks is not None:
        return None
    return _classes(legend)


def stamp(variable: xr.DataArray, band: RasterBand) -> xr.DataArray:
    """Stamp the legend a Raster band's classes state onto its variable.

    Args:
        variable: Variable the band describes.
        band: Its Raster band, listing its classes.

    Returns:
        The variable carrying its classes as `flag_values`, `flag_meanings`
        and, where classes carry colour hints, `color_map`; unchanged where
        the band lists none.

    Examples:
        >>> stamp(variable, band).attrs["flag_meanings"]
        'background forest'
    """
    classes = ClassificationExtension.ext(band).classes
    if not classes:
        return variable
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
    return variable.assign_attrs(legend.to_attrs())


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


def write_band(asset: pystac.Asset, band: RasterBand, variable: xr.DataArray) -> None:
    """Attach classes to a band and declare the schema on its owning Item."""
    if classes := from_dataarray(variable):
        ClassificationExtension.ext(band).apply(classes=classes)
        ClassificationExtension.ext(asset, add_if_missing=True)


def read_fields(variable: xr.DataArray, fields: dict) -> xr.DataArray:
    """Read a band's classification fields."""
    return stamp(variable, RasterBand(fields))
