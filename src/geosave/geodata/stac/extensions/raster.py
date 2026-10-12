"""Raster extension: how each band's stored numbers are typed, filled and packed."""

from __future__ import annotations

from math import isfinite

import pystac
import xarray as xr
from pystac.extensions.raster import (
    DataType,
    NoDataStrings,
    RasterBand,
    RasterExtension,
)

from typing import Any

import geosave.geodata.attrs as attrs


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write one Raster band per variable, in file order.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `open_raster` opens it, holding stored values. No
            pixel is read.

    Examples:
        >>> write(asset, scene)
        >>> RasterExtension.ext(asset).bands[0].scale
        0.0001
    """
    bands = []
    for name in raster.gs.variables:
        variable = raster[name]
        packing = variable.gs.packing
        fill = variable.gs.nodata
        if isinstance(fill, float) and not isfinite(fill):
            fill = NoDataStrings(str(fill))
        dtype = raster[name].dtype.name
        bands.append(
            RasterBand.create(
                nodata=fill,
                data_type=DataType(dtype)
                if dtype in list(DataType)
                else DataType.OTHER,
                scale=None if packing is None else packing.scale_factor,
                offset=None if packing is None else packing.add_offset,
                unit=variable.attrs.get("units"),
            )
        )
    RasterExtension.ext(asset, add_if_missing=True).apply(bands)


def read(band: RasterBand) -> dict[str, Any]:
    """Read the CF attrs one Raster band states.

    `nodata` and `data_type` describe the file, and the loader that opens it
    states them, so neither is read.

    Args:
        band: One band of an asset's Raster listing.

    Returns:
        The band's packing as `scale_factor` and `add_offset`, and its unit
        as `units`. Empty where it states neither.

    Examples:
        >>> read(RasterBand.create(unit="1", scale=0.0001))
        {'scale_factor': 0.0001, 'units': '1'}
    """
    found: dict[str, Any] = {}
    if band.scale is not None or band.offset is not None:
        packing = attrs.Packing(
            scale_factor=1.0 if band.scale is None else band.scale,
            add_offset=0.0 if band.offset is None else band.offset,
        )
        found |= packing.to_attrs()
    if band.unit is not None:
        found["units"] = band.unit
    return found
