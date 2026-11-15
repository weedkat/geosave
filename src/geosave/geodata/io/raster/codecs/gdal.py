"""GDAL codec: a Dataset as one band stack with GDAL properties, and back.

A GeoTIFF holds one band stack, as `Dataset.gs.to_array` builds it: shared
pixel meaning and Dataset attrs in the stack's attrs, each band's own attrs as
coords along `band`. rioxarray writes and reads pixels, grid, nodata, packing,
descriptions and tags; `encode_dataset` spells the stack in those slots and
`decode_dataset` reads them back. What rioxarray has no slot for is written
by `write_extras`, one writer per property: band units, RGB roles and class
colour tables. `read_properties` snapshots all of it from an open file.

    stack attrs          <-> file tags (and TIFFTAG_* with tiff_tags)
    scalar time coords   <-> TIFFTAG_DATETIME, ACDD coverage tags
    band labels          <-> band tag `variable_name`
    band coords          <-> band tags; `long_name` <-> band description
    units                <-> file tag and GDAL band unit
    Display.rgb          <-> file tag and colour interpretation
    CFFlags colours      <-> file tags and colour table
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import orjson
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.io import DatasetWriter

import geosave.geodata.attrs as attrs
from geosave import __path__ as package_paths
from geosave.geodata.core.conventions import (
    BAND_DIMENSION,
    END_DATETIME,
    START_DATETIME,
    TIME_COORDINATE,
)
from geosave.geodata.utils.color import parse_color
from geosave.geodata.utils.datetime import parse_stem_dates
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

# The band tag naming the label a band had in its stack.
LABEL_TAG = "variable_name"


@dataclass(frozen=True)
class BandProperties:
    """Native properties of one indexed band, retained after its reader closes."""

    index: int
    tags: dict[str, str]
    description: str | None
    unit: str | None
    packing: attrs.CFPacking
    interpretation: ColorInterp


@dataclass(frozen=True)
class RasterProperties:
    """File tags, block layout and band properties; no pixels or open handles."""

    tags: dict[str, str]
    bands: tuple[BandProperties, ...]
    width: int
    block_height: int
    block_width: int

    @property
    def is_tiled(self) -> bool:
        """Return whether blocks are tiles rather than full-width strips."""
        return self.block_width < self.width


@dataclass(frozen=True)
class ObservationTime:
    """One observation instant and optional coverage interval."""

    instant: np.datetime64 | None
    start: np.datetime64 | None = None
    end: np.datetime64 | None = None


# A Dataset as one band stack, and back


def encode_dataset(ds: xr.Dataset, *, tiff_tags: bool = False) -> xr.DataArray:
    """Stack one observation as the band array rioxarray writes.

    `ds.gs.to_array` stacks the bands with one pixel meaning; their metadata
    moves into the attrs `DataArray.rio.to_raster` translates: packing as
    nodata, scales and offsets, `long_name` as descriptions, `band_tags` and
    file tags. Pixels stay lazy.

    Args:
        ds: Raster with at most one instant along `time`.
        tiff_tags: Write ACDD attrs as baseline TIFF tags.

    Returns:
        Array shaped `(band, y, x)`.

    Raises:
        ValueError: The raster holds several instants.
        RasterValidationError: Variables state different pixel meaning.

    Examples:
        >>> encode_dataset(ds).attrs["band_tags"]
        [{'variable_name': 'red'}, {'variable_name': 'nir'}]
    """
    raster = ds.gs.to_array()
    if TIME_COORDINATE in raster.dims:
        if raster.sizes[TIME_COORDINATE] != 1:
            raise ValueError("one GeoTIFF holds one instant; select one observation")
        raster = raster.squeeze(TIME_COORDINATE)

    packing = raster.gs.packing
    encoded = raster.copy(deep=False)
    encoded.attrs = {
        **(packing.to_attrs() if packing is not None else {}),
        **format_file_tags(raster, tiff_tags=tiff_tags),
        "long_name": format_descriptions(raster),
        "band_tags": format_band_tags(raster),
    }
    return encoded


def decode_dataset(
    raster: xr.DataArray, properties: RasterProperties, *, filename: str
) -> xr.Dataset:
    """Split a band array rioxarray read into the Dataset `encode_dataset` stacked.

    Args:
        raster: Stored pixels shaped `(band, y, x)`, as rioxarray opens them.
        properties: The file's GDAL properties.
        filename: File name, whose date stands in for a missing time tag.

    Returns:
        Dataset of stored values, one variable per band.

    Raises:
        ValueError: Two bands carry the same `variable_name` tag.
    """
    raster = assign_band_coords(raster, properties)
    raster.attrs = read_file_tags(properties.tags)
    ds = raster.gs.to_dataset()
    ds = restore_packing(ds, properties)
    ds = restore_display(ds, properties)
    ds = restore_observation(ds, read_observation(properties.tags, filename))
    return restore_crs(ds)


def restore_crs(ds: xr.Dataset) -> xr.Dataset:
    """Spell the grid's coordinates as odc-geo does, as rasters hold them in memory.

    rioxarray leaves the spatial coordinates without attrs and adds
    `crs_wkt` and `GeoTransform` to the CRS coordinate; the grid is unchanged.
    """
    geobox = ds.odc.geobox
    if geobox is None or geobox.crs is None or not geobox.axis_aligned:
        return ds
    # GDAL hands back WKT; a CRS with an authority code is named by it.
    if geobox.crs.epsg is not None:
        geobox = GeoBox(geobox.shape, geobox.affine, f"EPSG:{geobox.crs.epsg}")
    return ds.assign_coords(xr_coords(geobox, always_yx=True))


# Tags


def format_text(values: Mapping[str, Any]) -> dict[str, str]:
    """Spell attrs as the text a GDAL metadata tag holds.

    A collection is spelled as JSON, which attrs groups decode back.
    """
    text = {}
    for key, value in values.items():
        if isinstance(value, str):
            text[key] = value
        else:
            text[key] = orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY).decode()
    return text


def map_tiff_tags(root: Mapping[str, Any]) -> dict[str, Any]:
    """Move ACDD attrs onto the baseline TIFF tags `TIFF_TAGS` names."""
    from importlib.metadata import version

    mapped = dict(root)
    for key, tag in TIFF_TAGS.items():
        if key in mapped:
            mapped[tag] = mapped.pop(key)
    mapped["TIFFTAG_SOFTWARE"] = f"geosave {version('geosave')}"
    return mapped


def format_file_tags(raster: xr.DataArray, *, tiff_tags: bool) -> dict[str, str]:
    """Spell a stack's attrs and observation coords as file tags.

    Packing is left out; rioxarray writes it as GDAL's own scale, offset and
    nodata.
    """
    root = {}
    for key, value in raster.attrs.items():
        if key not in attrs.CFPacking.KEYS:
            root[key] = value
    root.update(format_time_tags(raster))
    if tiff_tags:
        root = map_tiff_tags(root)
    return format_text(root)


def read_file_tags(tags: Mapping[str, str]) -> dict[str, str]:
    """Map baseline TIFF labels back to ACDD names, retaining arbitrary tags."""
    root = dict(tags)
    for key, tag in TIFF_TAGS.items():
        if tag in root and key not in root:
            root[key] = root.pop(tag)
    root.pop("TIFFTAG_SOFTWARE", None)
    return root


# Band labels, band coords and descriptions


def band_coords(raster: xr.DataArray) -> list[str]:
    """Name the stack's coords along `band`, other than its labels."""
    names = []
    for name, coord in raster.coords.items():
        if coord.dims == (BAND_DIMENSION,) and name != BAND_DIMENSION:
            names.append(str(name))
    return names


def format_band_tags(raster: xr.DataArray) -> list[dict[str, str]]:
    """Spell each band's label and band coords as its tags; `long_name` is its description."""
    names = band_coords(raster)
    tags = []
    for index, label in enumerate(raster[BAND_DIMENSION].values):
        band = {LABEL_TAG: str(label)}
        for name in names:
            value = raster[name].values[index]
            if name != "long_name" and value is not None:
                band[name] = value
        tags.append(format_text(band))
    return tags


def format_descriptions(raster: xr.DataArray) -> list[str]:
    """Describe each band by its `long_name` coord, else by its label."""
    labels = raster[BAND_DIMENSION].values
    descriptions = []
    for index, label in enumerate(labels):
        description = str(label)
        if "long_name" in raster.coords:
            long_name = raster["long_name"].values[index]
            if long_name is not None:
                description = str(long_name)
        descriptions.append(description)
    return descriptions


def read_labels(properties: RasterProperties) -> list[str]:
    """Label each band by its `variable_name` tag, else as `band_<index>`."""
    labels = []
    for band in properties.bands:
        labels.append(band.tags.get(LABEL_TAG) or f"band_{band.index}")
    if len(set(labels)) != len(labels):
        raise ValueError("one file names each band once; duplicate variable_name tags")
    return labels


def assign_band_coords(
    raster: xr.DataArray, properties: RasterProperties
) -> xr.DataArray:
    """Label the bands and attach their tags, descriptions and units as coords.

    A description repeating the band's label states no `long_name`. GDAL's
    own statistics tags are left out. A band lacking a tag another band has
    holds None there.

    Args:
        raster: Stored pixels shaped `(band, y, x)`.
        properties: The file's GDAL properties.

    Returns:
        The array labelled along `band`, with one coord per band property.
    """
    labels = read_labels(properties)
    labelled = raster.assign_coords({BAND_DIMENSION: labels})
    for index, band in enumerate(properties.bands):
        stated = read_band_attrs(band, labels[index])
        for key, value in stated.items():
            if key not in labelled.coords:
                absent = np.full(len(labels), None, dtype=object)
                labelled = labelled.assign_coords({key: (BAND_DIMENSION, absent)})
            labelled[key].values[index] = value
    return labelled


def read_band_attrs(band: BandProperties, label: str) -> dict[str, str]:
    """Gather what one band states about itself, as the attrs it had."""
    stated = {}
    for key, value in band.tags.items():
        if key != LABEL_TAG and not key.startswith("STATISTICS_"):
            stated[key] = value
    if band.description is not None and band.description != label:
        stated["long_name"] = band.description
    if band.unit is not None:
        stated["units"] = band.unit
    return stated


# Snapshot


def read_band_properties(src: rasterio.DatasetReader, index: int) -> BandProperties:
    """Snapshot one band's tags, labels, packing and colour interpretation."""
    return BandProperties(
        index=index,
        tags=src.tags(index),
        description=src.descriptions[index - 1],
        unit=src.units[index - 1],
        packing=read_packing(src, index),
        interpretation=src.colorinterp[index - 1],
    )


def read_properties(address: str, **open_options: Any) -> RasterProperties:
    """Snapshot a file's tags, block layout and band properties.

    Args:
        address: Path or URL Rasterio opens.
        **open_options: Rasterio open options, such as `opener`.

    Returns:
        The properties, readable after the file has closed.

    Warns:
        UnreadMaskWarning: A band marks absent pixels in a mask band.
    """
    with rasterio.open(address, **open_options) as src:
        warn_unread_masks(src)
        bands = []
        for index in src.indexes:
            bands.append(read_band_properties(src, index))
        block_height, block_width = src.block_shapes[0]
        return RasterProperties(
            tags=src.tags(),
            bands=tuple(bands),
            width=src.width,
            block_height=block_height,
            block_width=block_width,
        )


def warn_unread_masks(src: rasterio.DatasetReader) -> None:
    """Report dataset masks that the stored-value reader does not apply."""
    indexes = []
    for index, flags in zip(src.indexes, src.mask_flag_enums, strict=True):
        if MaskFlags.per_dataset in flags:
            indexes.append(index)
    if indexes:
        warnings.warn(
            f"bands {indexes} mark their absent pixels in a mask band, "
            f"which this reader does not read, so every pixel reads as present; "
            f"give them a fill value, or read the mask with rasterio",
            UnreadMaskWarning,
            skip_file_prefixes=tuple(package_paths),
        )


# Packing


def read_packing(src: rasterio.DatasetReader, index: int) -> attrs.CFPacking:
    """Translate native band scale, offset and nodata into CF packing."""
    return attrs.CFPacking(
        scale_factor=src.scales[index - 1],
        add_offset=src.offsets[index - 1],
        fill_value=src.nodatavals[index - 1],
    )


def restore_packing(ds: xr.Dataset, properties: RasterProperties) -> xr.Dataset:
    """State each band's stored packing in its attrs; a file may scale bands apart."""
    ds = ds.copy(deep=False)
    for name, band in zip(ds.data_vars, properties.bands, strict=True):
        ds[name] = attrs.stamp_attrs(ds[name], band.packing)
    return ds


# Units


def write_units(dst: DatasetWriter, raster: xr.DataArray) -> None:
    """Assign the stack's shared unit as each band's GDAL unit."""
    unit = raster.attrs.get("units")
    if unit is None:
        return
    for index in dst.indexes:
        dst.set_band_unit(index, unit)


# Class colour tables


def write_colormaps(dst: DatasetWriter, raster: xr.DataArray) -> None:
    """Write the stack's class colours as every band's colour table."""
    flags = attrs.CFFlags.from_attrs(raster.attrs)
    if flags is None or flags.color_hints is None:
        return
    colormap = {}
    for value, colour in flags.colors.items():
        colormap[int(value)] = (*parse_color(colour), 255)
    for index in dst.indexes:
        dst.write_colormap(index, colormap)


# RGB display


def restore_display(ds: xr.Dataset, properties: RasterProperties) -> xr.Dataset:
    """State the file's RGB roles as `Display`, unless its tags already do."""
    if attrs.Display.is_stated_in(ds.attrs):
        return ds
    display = read_display([str(name) for name in ds.data_vars], properties)
    if display is None:
        return ds
    return attrs.stamp_attrs(ds, display)


def write_colorinterp(dst: DatasetWriter, raster: xr.DataArray) -> None:
    """Assign the declared RGB channels; other bands keep GDAL's interpretation."""
    display = attrs.Display.from_attrs(raster.attrs)
    if display is None or display.rgb is None:
        return
    roles = dict(
        zip(
            display.rgb,
            (ColorInterp.red, ColorInterp.green, ColorInterp.blue),
            strict=True,
        )
    )
    interpretation = list(dst.colorinterp)
    for index, label in enumerate(raster[BAND_DIMENSION].values):
        if str(label) in roles:
            interpretation[index] = roles[str(label)]
    dst.colorinterp = interpretation


def read_display(
    labels: list[str], properties: RasterProperties
) -> attrs.Display | None:
    """Translate complete native RGB roles into band labels."""
    channels = {}
    for label, band in zip(labels, properties.bands, strict=True):
        channels[band.interpretation] = label
    rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    for channel in rgb:
        if channel not in channels:
            return None
    return attrs.Display(rgb=[channels[channel] for channel in rgb])


# Observation time


def format_time_tags(raster: xr.DataArray) -> dict[str, str]:
    """State the scalar time as `TIFFTAG_DATETIME` and its period as ACDD attrs.

    A TIFF has no coordinates. `TIFFTAG_DATETIME` holds whole seconds;
    `start_datetime` and `end_datetime` become ACDD `time_coverage_start` and
    `time_coverage_end`.

    Raises:
        ValueError: `time` is not a datetime, or is NaT.
    """
    time = raster.coords.get(TIME_COORDINATE)
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
    if START_DATETIME not in raster.coords:
        return tags

    coverage = attrs.ACDD(
        time_coverage_start=np.datetime_as_string(
            raster[START_DATETIME].values.min(), unit="auto"
        ),
        time_coverage_end=np.datetime_as_string(
            raster[END_DATETIME].values.max(), unit="auto"
        ),
    )
    return {**tags, **coverage.to_attrs()}


def read_observation(tags: Mapping[str, str], filename: str) -> ObservationTime:
    """Read a TIFF instant or filename fallback and an optional ACDD interval."""
    instant = None
    if stamp := tags.get("TIFFTAG_DATETIME"):
        try:
            instant = np.datetime64(datetime.strptime(stamp, GDAL_DATETIME_FORMAT))  # noqa: DTZ007
        except ValueError:
            pass
    if instant is None:
        period = parse_stem_dates(PurePath(filename).stem)
        if period is not None:
            instant = np.datetime64(period[0])
    coverage = attrs.ACDD.from_attrs(tags)
    if (
        coverage is not None
        and coverage.time_coverage_start is not None
        and coverage.time_coverage_end is not None
    ):
        return ObservationTime(
            instant,
            np.datetime64(coverage.time_coverage_start, "ns"),
            np.datetime64(coverage.time_coverage_end, "ns"),
        )
    return ObservationTime(instant)


def restore_observation(ds: xr.Dataset, observation: ObservationTime) -> xr.Dataset:
    """Restore scalar time coordinates and consume their transport tags."""
    ds = ds.copy(deep=False)
    if observation.instant is not None:
        ds = ds.assign_coords({TIME_COORDINATE: observation.instant})
    if observation.start is not None and observation.end is not None:
        ds = ds.assign_coords(
            {START_DATETIME: observation.start, END_DATETIME: observation.end}
        )
    for key in ("TIFFTAG_DATETIME", "time_coverage_start", "time_coverage_end"):
        ds.attrs.pop(key, None)
    return ds


# Extras: properties rioxarray has no slot for, one writer each

EXTRA_WRITERS: tuple[Callable[[DatasetWriter, xr.DataArray], None], ...] = (
    write_units,
    write_colorinterp,
    write_colormaps,
)


def write_extras(local: Path, raster: xr.DataArray) -> None:
    """Add the properties rioxarray does not write to a finished GeoTIFF.

    Args:
        local: The GeoTIFF rioxarray wrote, on the local filesystem.
        raster: The stack `encode_dataset` built for it.
    """
    with rasterio.open(local, "r+") as dst:
        for write in EXTRA_WRITERS:
            write(dst, raster)
