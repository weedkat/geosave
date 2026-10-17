"""Ordered asset bands and composition of their schema metadata."""

from __future__ import annotations
from typing import Any

import pystac
import xarray as xr
from geosave.geodata.stac.extensions import classification, eo, raster


def write_metadata(asset: pystac.Asset, header: xr.Dataset) -> None:
    """Write Raster, EO and Classification metadata on an Asset with an owner."""
    variables = list(header.data_vars.values())
    stored = [raster.from_dataarray(variable) for variable in variables]
    for entry, variable in zip(stored, variables, strict=True):
        classification.write_band(asset, entry, variable)

    raster.write_asset(asset, stored)
    eo.write_asset(asset, [eo.from_dataarray(variable) for variable in variables])


def read_aliases(asset: pystac.Asset) -> list[tuple[str, ...]]:
    """Return each band's published name and common name, in band order."""
    aliases = []
    for entry in _read_fields(asset):
        names = (
            entry.get("name"),
            entry.get("common_name", entry.get("eo:common_name")),
        )
        aliases.append(tuple(name for name in names if name is not None))
    return aliases


def read_attrs(
    variable: xr.DataArray, asset: pystac.Asset, index: int
) -> dict[str, Any]:
    """Read only the published band's metadata, retaining the variable's context.

    Args:
        variable: Loaded band; pixels and existing attrs are unchanged.
        asset: Asset describing the band.
        index: Zero-based band position.

    Returns:
        Translated source attrs, without the file or loader's existing attrs.
    """
    stated = variable.copy(deep=False)
    stated.attrs = {}

    entries = _read_fields(asset)
    if index >= len(entries):
        return {}
    entry = entries[index]
    stated = raster.read_fields(stated, entry)
    stated = classification.read_fields(stated, entry)
    stated = eo.read_fields(stated, entry)
    return dict(stated.attrs)


def _read_fields(asset: pystac.Asset) -> list[dict]:
    """Combine STAC 1.0 extension lists or STAC 1.1 core bands by index."""
    core = asset.extra_fields.get("bands", [])
    stored = asset.extra_fields.get("raster:bands", [])
    spectral = asset.extra_fields.get("eo:bands", [])
    count = max(len(core), len(stored), len(spectral))
    result = []
    for index in range(count):
        entry = {}
        for listing in (stored, spectral, core):
            if index < len(listing):
                entry.update(listing[index])
        result.append(entry)
    return result
