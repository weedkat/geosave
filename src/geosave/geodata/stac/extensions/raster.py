"""Raster extension: a band's stored type, fill, packing and unit."""

from __future__ import annotations

from math import isfinite
from typing import TypedDict, cast


import pystac
import xarray as xr
from pystac.extensions.raster import (
    DataType,
    NoDataStrings,
    RasterBand,
    RasterExtension,
)

import geosave.geodata.attrs as attrs


def read_json_number(value: float) -> float:
    """Spell a float as the shortest number that reads back as its own dtype.

    A float32 scale of `1e-4` is `9.999999747378752e-05` as a Python float;
    its own shortest spelling, `1e-04`, is what a reader expects.
    """
    return float(str(value))


def build_band(variable: xr.DataArray) -> RasterBand:
    """Build the Raster band storing one variable.

    Args:
        variable: Variable as `open_raster` opens it, holding stored values.
            No pixel is read.

    Returns:
        Its band: `data_type`, `nodata`, `scale` and `offset`, and `unit`.

    Examples:
        >>> build_band(scene.red).scale
        0.0001
    """
    fill = variable.gs.nodata
    if isinstance(fill, float) and not isfinite(fill):
        fill = NoDataStrings(str(fill))
    dtype = variable.dtype.name

    # Packing as stored, in its own dtype, spelled as short as it reads back.
    scale = variable.attrs.get("scale_factor")
    offset = variable.attrs.get("add_offset")
    return RasterBand.create(
        nodata=fill,
        data_type=DataType(dtype) if dtype in list(DataType) else DataType.OTHER,
        scale=None if scale is None else read_json_number(scale),
        offset=None if offset is None else read_json_number(offset),
        unit=variable.attrs.get("units"),
    )


class RasterAttrs(TypedDict, total=False):
    """CF storage attrs translated from one Raster band."""

    scale_factor: float
    add_offset: float
    units: str


def read_attrs(band: RasterBand) -> RasterAttrs:
    """Translate a band's packing and units into CF attrs.

    Stored dtype and nodata belong to the file reader and are not copied.
    """
    fields: RasterAttrs = {}
    if band.scale is not None or band.offset is not None:
        packing = attrs.CFPacking(
            scale_factor=1.0 if band.scale is None else band.scale,
            add_offset=0.0 if band.offset is None else band.offset,
        )
        fields = cast("RasterAttrs", packing.to_attrs())
    if band.unit is not None:
        fields["units"] = band.unit
    return fields


def write_asset(asset: pystac.Asset, bands: list[RasterBand]) -> None:
    """Write storage bands and declare their schema on the owner."""
    RasterExtension.ext(asset, add_if_missing=True).bands = bands
