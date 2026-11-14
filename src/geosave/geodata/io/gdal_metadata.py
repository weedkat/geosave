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


# These attrs have native GDAL properties rather than generic metadata tags.
_BAND_PROPERTIES = {
    *attrs.CFPacking.KEYS,
    *attrs.Display.KEYS,
    "color_hints",
    "long_name",
    "units",
}


def file_tags(raster: xr.Dataset, *, tiff_tags: bool = False) -> dict[str, str]:
    """Translate root attrs and observation coordinates into GDAL tags."""
    owned = {*attrs.CFPacking.KEYS, *attrs.Display.KEYS, "color_hints"}
    root = {key: value for key, value in raster.attrs.items() if key not in owned}
    root.update(format_time_tags(raster))
    if tiff_tags:
        root = _map_tiff_tags(root)
    return _as_text(root)


def write_band_tags(dst: DatasetWriter, bands: xr.Dataset) -> None:
    """Write variable identity and attrs without native GDAL properties."""
    for index, (name, band) in enumerate(bands.data_vars.items(), start=1):
        tags = {
            key: value
            for key, value in band.attrs.items()
            if key not in _BAND_PROPERTIES
        }
        tags["variable_name"] = str(name)
        dst.update_tags(index, **_as_text(tags))


def write_descriptions(dst: DatasetWriter, bands: xr.Dataset) -> None:
    """Assign each band's description and unit."""
    for index, band in enumerate(bands.data_vars.values(), start=1):
        description = band.attrs.get("long_name", str(band.name))
        dst.set_band_description(index, description)
        unit = band.attrs.get("units")
        if unit is not None:
            dst.set_band_unit(index, unit)


def write_packing(dst: DatasetWriter, bands: xr.Dataset) -> None:
    """Assign stored bands' scale and offset; nodata belongs to the profile."""
    dst.scales = [
        band.attrs.get("scale_factor", 1.0) for band in bands.data_vars.values()
    ]
    dst.offsets = [
        band.attrs.get("add_offset", 0.0) for band in bands.data_vars.values()
    ]


def write_colorinterp(dst: DatasetWriter, raster: xr.Dataset) -> None:
    """Assign declared RGB channels; palette writers own palette interpretation."""
    rgb = raster.gs.rgb
    if rgb is None:
        return
    channels = dict(
        zip(rgb, (ColorInterp.red, ColorInterp.green, ColorInterp.blue), strict=True)
    )
    dst.colorinterp = [
        channels.get(str(name), ColorInterp.undefined) for name in raster.data_vars
    ]


def write_colormaps(dst: DatasetWriter, bands: xr.Dataset) -> None:
    """Write a colour table for each band declaring class colours."""
    for index, band in enumerate(bands.data_vars.values(), start=1):
        flags = band.gs.flags
        if flags is None or flags.color_hints is None:
            continue
        dst.write_colormap(
            index,
            {
                int(value): (*parse_color(colour), 255)
                for value, colour in flags.colors.items()
            },
        )


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
        raise ValueError("a GeoTIFF observation time cannot be NaT")

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
