"""Raster extension: a band's stored type, fill, packing and unit."""

from __future__ import annotations

from math import isfinite

import xarray as xr
import pystac
from pystac.extensions.raster import (
    DataType,
    NoDataStrings,
    RasterBand,
    RasterExtension,
)

import geosave.geodata.attrs as attrs


def from_dataarray(variable: xr.DataArray) -> RasterBand:
    """Describe one variable as the Raster band storing it.

    Args:
        variable: Variable as `open_raster` opens it, holding stored values.
            No pixel is read.

    Returns:
        Its band: `data_type`, `nodata`, `scale` and `offset`, and `unit`.

    Examples:
        >>> from_dataarray(scene.red).scale
        0.0001
    """
    packing = variable.gs.packing
    fill = variable.gs.nodata
    if isinstance(fill, float) and not isfinite(fill):
        fill = NoDataStrings(str(fill))
    dtype = variable.dtype.name
    return RasterBand.create(
        nodata=fill,
        data_type=DataType(dtype) if dtype in list(DataType) else DataType.OTHER,
        scale=None if packing is None else packing.scale_factor,
        offset=None if packing is None else packing.add_offset,
        unit=variable.attrs.get("units"),
    )


def stamp(variable: xr.DataArray, band: RasterBand) -> xr.DataArray:
    """Stamp the packing and unit a Raster band states onto its variable.

    `data_type` and `nodata` describe the file, which states them itself
    when opened, so neither is stamped.

    Args:
        variable: Variable the band describes.
        band: Its band.

    Returns:
        The variable carrying the band's packing as `scale_factor` and
        `add_offset`, and its unit as `units`.

    Examples:
        >>> stamp(variable, RasterBand.create(scale=0.0001, unit="1")).attrs
        {'scale_factor': 0.0001, 'add_offset': 0.0, 'units': '1'}
    """
    if band.scale is not None or band.offset is not None:
        packing = attrs.Packing(
            scale_factor=1.0 if band.scale is None else band.scale,
            add_offset=0.0 if band.offset is None else band.offset,
        )
        variable = variable.assign_attrs(packing.to_attrs())
    if band.unit is not None:
        variable = variable.assign_attrs(units=band.unit)
    return variable


def write_asset(asset: pystac.Asset, bands: list[RasterBand]) -> None:
    """Write storage bands and declare their schema on the owner."""
    RasterExtension.ext(asset, add_if_missing=True).bands = bands


def read_fields(variable: xr.DataArray, fields: dict) -> xr.DataArray:
    """Read core or extension band metadata."""
    return stamp(
        variable,
        RasterBand(
            {key.removeprefix("raster:"): value for key, value in fields.items()}
        ),
    )
