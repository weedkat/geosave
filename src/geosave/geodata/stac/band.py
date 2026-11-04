"""Ordered asset bands and composition of their schema metadata."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import pystac
import xarray as xr

import geosave.geodata.attrs as attrs
from geosave import __path__ as _package_paths
from geosave.geodata.stac.extensions import classification, eo, raster
from geosave.geodata.warnings import DroppedAttrsWarning


def write_band_fields(asset: pystac.Asset, header: xr.Dataset) -> None:
    """Write Raster, EO and Classification metadata on an Asset with an owner."""
    variables = list(header.data_vars.values())
    stored = [raster.build_band(variable) for variable in variables]
    for entry, variable in zip(stored, variables, strict=True):
        classification.write_band(asset, entry, variable)

    raster.write_asset(asset, stored)
    eo.write_asset(asset, [eo.build_band(variable) for variable in variables])


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
        asset: Asset stating the band.
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
    stated = raster.stamp_fields(stated, entry)
    stated = classification.stamp_fields(stated, entry)
    stated = eo.stamp_fields(stated, entry)
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


def read_shared_attrs(
    variable: xr.DataArray, sources: Sequence[tuple[pystac.Item, str, int]]
) -> dict[str, Any]:
    """Read the published band attrs on which the source Items agree.

    Args:
        variable: Loaded band providing context for metadata interpretation.
        sources: Source Items with their asset keys and zero-based band indices,
            already resolved by the caller. An absent asset states no attrs.

    Returns:
        Agreed source attrs. Pixels and existing variable attrs are unchanged.

    Warns:
        DroppedAttrsWarning: Source Items disagree on a band's metadata.
    """
    published = []
    for entry, key, index in sources:
        asset = entry.assets.get(key)
        published.append({} if asset is None else read_attrs(variable, asset, index))

    kept = attrs.intersect(published)
    stated_keys = set()
    for fields in published:
        stated_keys.update(fields)
    disputed = sorted(stated_keys - kept.keys())
    if disputed:
        values = {}
        for key in disputed:
            values[key] = {}
            for (entry, _, _), fields in zip(sources, published, strict=True):
                values[key][entry.id] = fields.get(key)
        warnings.warn(
            f"STAC items publish {disputed} differently for {variable.name!r}, so "
            f"those source attrs are dropped: {values}",
            DroppedAttrsWarning,
            skip_file_prefixes=tuple(_package_paths),
        )
    return kept
