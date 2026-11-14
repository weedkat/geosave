"""Translate raster metadata between xarray attrs and GDAL band properties."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import orjson
import rasterio
import xarray as xr
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.io import DatasetWriter

import geosave.geodata.attrs as attrs
from geosave import __path__ as _package_paths
from geosave.geodata.core.conventions import (
    END_DATETIME,
    START_DATETIME,
    TIME_COORDINATE,
)
from geosave.geodata.utils.color import parse_color
from geosave.geodata.warnings import UnreadMaskWarning

# The text GDAL writes in the TIFF datetime tag, whole seconds only.
GDAL_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"

# ACDD attrs and the baseline TIFF tags GDAL writes them as, which any TIFF
# reader shows; other attrs go to GDAL's private metadata XML.
TIFF_TAGS = {
    "title": "TIFFTAG_DOCUMENTNAME",
    "summary": "TIFFTAG_IMAGEDESCRIPTION",
    "license": "TIFFTAG_COPYRIGHT",
    "institution": "TIFFTAG_ARTIST",
}


def _read_class_attrs(src: rasterio.DatasetReader, index: int) -> dict[str, Any]:
    """Read a band's CF flags and, for a palette band, its class colours.

    Args:
        src: Open raster.
        index: One-based band index.

    Returns:
        `CFFlags` attrs, empty where the band's tags list no `flag_values`.
        A palette band takes its classes' colour hints from its table.
    """
    tags = src.tags(index)
    if "flag_values" not in tags:
        return {}
    flags = attrs.CFFlags.model_validate(tags)
    if src.colorinterp[index - 1] is not ColorInterp.palette or flags.classes is None:
        return flags.to_attrs()

    colours = src.colormap(index)
    palette = {}
    for value in flags.classes:
        if value in colours:
            palette[value] = colours[value][:3]
    painted = attrs.CFFlags.model_validate(
        {"classes": flags.classes, "colors": palette}
    )
    return painted.to_attrs()


def read_metadata(
    src: rasterio.DatasetReader,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read the file's tags and each band's native properties as CF attrs.

    Args:
        src: Open raster.

    Returns:
        (attrs of the whole file, variable name mapped to its band's attrs,
        in band order). A band is named by its `variable_name` tag, else by
        rasterio's positional `band_<n>`.

    Raises:
        ValueError: Two bands name one variable.

    Warns:
        UnreadMaskWarning: A band records absence in a mask that is not read.
    """
    with_mask_band = [
        index
        for index, flags in zip(src.indexes, src.mask_flag_enums, strict=True)
        if MaskFlags.per_dataset in flags
    ]
    if with_mask_band:
        warnings.warn(
            f"bands {with_mask_band} mark their absent pixels in a mask band, "
            f"which this reader does not read, so every pixel reads as present; "
            f"give them a fill value, or read the mask with rasterio",
            UnreadMaskWarning,
            skip_file_prefixes=tuple(_package_paths),
        )

    bands: dict[str, dict[str, Any]] = {}
    for index in src.indexes:
        position = index - 1
        band: dict[str, Any] = {
            key: value
            for key, value in src.tags(index).items()
            if not key.startswith("STATISTICS_")
        }
        native: dict[str, Any] = {
            "long_name": src.descriptions[position],
            "units": src.units[position],
        }
        band.update({key: value for key, value in native.items() if value is not None})

        # GDAL keeps packing and fill as band properties; they read as CF's attrs.
        packing = attrs.CFPacking(
            scale_factor=src.scales[position],
            add_offset=src.offsets[position],
            fill_value=src.nodatavals[position],
        )
        band.update(packing.to_attrs())

        name = band.pop("variable_name", None) or f"band_{index}"
        # rioxarray names an unlabelled band after its position, which is no label.
        if band.get("long_name") == name:
            del band["long_name"]

        band.update(_read_class_attrs(src, index))

        if name in bands:
            raise ValueError(
                f"band {index} names variable {name!r}, which an earlier band "
                f"already names; one file names each variable once"
            )
        bands[name] = band

    root = dict(src.tags())
    # Baseline TIFF tags read back as the ACDD attrs a `tiff_tags` write moved.
    for key, tag in TIFF_TAGS.items():
        if tag in root and key not in root:
            root[key] = root.pop(tag)
    # GDAL states a composite per band; GeoSave states it once, on the Dataset.
    drawing = {}
    for name, interpretation in zip(bands, src.colorinterp, strict=True):
        if interpretation.name in ("red", "green", "blue"):
            drawing[interpretation.name] = name
    if len(drawing) == 3:
        rgb = [drawing["red"], drawing["green"], drawing["blue"]]
        root.update(attrs.Display(rgb=rgb).to_attrs())
    return root, bands


def write_metadata(
    dst: DatasetWriter, raster: xr.Dataset, *, tiff_tags: bool = False
) -> None:
    """Write every attr onto an open GDAL raster, in the slot GDAL reads it from.

    GDAL keeps the description, unit, scale, offset, fill value, interpretation
    and palette as band properties of its own, where `gdalinfo` and QGIS look;
    every other attr is a plain metadata item, and none is stated twice. The
    `Display` maps to GDAL: `rgb` to red/green/blue interpretations, a
    `CFFlags.color_hints` to a colour table. Each band names its variable in a
    `variable_name` item.

    Args:
        dst: Raster open for update.
        raster: Prepared Dataset with encoded time tags, one variable per band.
        tiff_tags: Map ACDD fields to baseline TIFF tags.

    Raises:
        ValueError: `bands` does not hold one variable per band of `dst`.

    Examples:
        >>> with rasterio.open(path, "r+") as dst:
        ...     write_metadata(dst, ds)
    """
    root = dict(raster.attrs)
    if tiff_tags:
        root = _map_tiff_tags(root)
    bands = list(raster.data_vars.values())
    if len(bands) != dst.count:
        raise ValueError(
            f"{dst.name} holds {dst.count} bands but {len(bands)} variables "
            f"name them; pass one variable per band, in band order"
        )

    # GDAL stores these as band properties, never as dataset JSON tags; class
    # colours become each band's colour table.
    owned = {*attrs.CFPacking.KEYS, *attrs.Display.KEYS, "color_hints"}
    root_tags = {key: value for key, value in root.items() if key not in owned}
    dst.update_tags(**_as_text(root_tags))
    display = attrs.Display.from_attrs(root)
    rgb = [] if display is None or display.rgb is None else display.rgb

    scales: list[float] = []
    offsets: list[float] = []
    interpretations: list[ColorInterp] = []
    for index, band in enumerate(bands, start=1):
        packing = band.gs.packing
        flags = band.gs.flags
        tags = {key: value for key, value in band.attrs.items() if key not in owned}
        tags["variable_name"] = str(band.name)

        # Each of these leaves the tags for a band property GDAL keeps it in.
        description = tags.pop("long_name", None)
        unit = tags.pop("units", None)
        dst.update_tags(index, **_as_text(tags))

        if description is not None:
            dst.set_band_description(index, description)
        if unit is not None:
            dst.set_band_unit(index, unit)

        scales.append(1.0 if packing is None else packing.scale_factor)
        offsets.append(0.0 if packing is None else packing.add_offset)
        palette = None
        if flags is not None and flags.color_hints is not None:
            palette = flags.colors
        if palette is not None:
            interpretations.append(ColorInterp.palette)
        elif str(band.name) in rgb:
            channel = ("red", "green", "blue")[rgb.index(str(band.name))]
            interpretations.append(ColorInterp[channel])
        else:
            interpretations.append(ColorInterp.undefined)

        if palette is not None:
            dst.write_colormap(
                index,
                {
                    int(value): (*parse_color(colour), 255)
                    for value, colour in palette.items()
                },
            )

    if any(scale != 1.0 for scale in scales):
        dst.scales = scales
    if any(offset != 0.0 for offset in offsets):
        dst.offsets = offsets
    # Assigning none of them would clear what GDAL inferred for itself.
    if any(band is not ColorInterp.undefined for band in interpretations):
        dst.colorinterp = interpretations

    shared = bands[0].gs.nodata if bands else None
    if shared is not None:
        dst.nodata = shared


def format_time_tags(cube: xr.Dataset) -> dict[str, str]:
    """State the file's time as `TIFFTAG_DATETIME` and its period as ACDD attrs.

    A TIFF has no coordinates. The scalar `time` coordinate becomes the TIFF
    datetime tag, which holds whole seconds; `start_datetime` and
    `end_datetime` become ACDD `time_coverage_start` and `time_coverage_end`.

    Raises:
        ValueError: `time` is not a datetime.
    """
    time = cube.coords.get(TIME_COORDINATE)
    if time is None or time.ndim != 0:
        return {}
    instant = time.values
    if not isinstance(instant, np.datetime64):
        raise ValueError(
            f"the time coordinate must be a datetime to be stored in a GeoTIFF, "
            f"got {type(instant).__name__}; convert it with pd.to_datetime first"
        )
    if np.isnat(instant):
        return {}

    stamp = instant.astype("datetime64[s]").item()
    tags = {"TIFFTAG_DATETIME": stamp.strftime(GDAL_DATETIME_FORMAT)}
    if START_DATETIME not in cube.coords:
        return tags

    coverage = attrs.ACDD(
        time_coverage_start=np.datetime_as_string(
            cube[START_DATETIME].values.min(), unit="auto"
        ),
        time_coverage_end=np.datetime_as_string(
            cube[END_DATETIME].values.max(), unit="auto"
        ),
    )
    return {**tags, **coverage.to_attrs()}


def _map_tiff_tags(root: Mapping[str, Any]) -> dict[str, Any]:
    """Move ACDD attrs onto the baseline TIFF tags `TIFF_TAGS` names."""
    from importlib.metadata import version

    mapped = dict(root)
    for key, tag in TIFF_TAGS.items():
        if key in mapped:
            mapped[tag] = mapped.pop(key)
    mapped["TIFFTAG_SOFTWARE"] = f"geosave {version('geosave')}"
    return mapped


def _as_text(attrs: Mapping[str, Any]) -> dict[str, str]:
    """Spell attrs as the text a GDAL metadata tag holds.

    Args:
        attrs: Flat attrs to write as tags.

    Returns:
        {
            "<attr key>": its value as text,
        }
        A collection is spelled as JSON, which reads back as itself.
    """
    return {
        key: value
        if isinstance(value, str)
        else orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY).decode()
        for key, value in attrs.items()
    }
