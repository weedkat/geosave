"""Map layers: one per raster or vector, each keeping its own display choices."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import geopandas as gpd
import xarray as xr

from .layer import Layer
from .raster import RasterLayer
from .vector import VectorLayer

type GeoData = xr.Dataset | xr.DataArray | gpd.GeoDataFrame


def create_layer(data: GeoData | Layer, name: str | None = None, **style: Any) -> Layer:
    """Create display settings for native raster or vector data."""
    if isinstance(data, Layer):
        return data
    if isinstance(data, xr.Dataset | xr.DataArray):
        return RasterLayer(data, name=name, **style)
    # An item table's rows also open their rasters.
    if isinstance(data, gpd.GeoDataFrame):
        return VectorLayer(data, name=name, **style)
    raise TypeError(f"cannot explore {type(data).__name__}")


def create_layers(items: Iterable[GeoData | Layer]) -> list[Layer]:
    """Create one layer per raster, frame or layer, in order."""
    layers = []
    for data in items:
        layers.append(create_layer(data))
    return layers


__all__ = [
    "GeoData",
    "Layer",
    "RasterLayer",
    "VectorLayer",
    "create_layer",
    "create_layers",
]
