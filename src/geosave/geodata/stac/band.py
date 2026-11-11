"""Ordered asset bands and composition of their schema metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict, cast

import pystac

import geosave.geodata.attrs as attrs
from geosave.geodata.stac.extensions import classification, eo, raster


BandFields = TypedDict(
    "BandFields",
    {
        "name": str,
        "common_name": str,
        "description": str,
        "eo:common_name": str,
        "center_wavelength": float,
        "full_width_half_max": float,
        "eo:center_wavelength": float,
        "eo:full_width_half_max": float,
        "scale": float,
        "offset": float,
        "unit": str,
        "raster:scale": float,
        "raster:offset": float,
        "raster:unit": str,
        "classification:classes": list[dict[str, Any]],
    },
    total=False,
)


class BandAttrs(
    raster.RasterAttrs, eo.SpectralAttrs, classification.ClassificationAttrs
):
    """CF storage, spectral and classification attrs published for a band."""


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


def read_attrs(asset: pystac.Asset, index: int) -> BandAttrs:
    """Translate one published band into raster attrs.

    Args:
        asset: Asset publishing core bands or extension band lists.
        index: Zero-based band position.

    Returns:
        CF packing, units, legend and spectral attrs; empty if absent.
    """
    entries = _read_fields(asset)
    if index < 0:
        raise ValueError("band index must be nonnegative")
    if index >= len(entries):
        return {}

    entry = entries[index]
    storage_fields = {}
    spectral_fields = {}
    for key, value in entry.items():
        storage_fields[key.removeprefix("raster:")] = value
        spectral_fields[key.removeprefix("eo:")] = value

    packing = raster.read_attrs(raster.RasterBand(storage_fields))
    legend = classification.read_attrs(raster.RasterBand(dict(entry)))
    spectral = eo.read_attrs(eo.Band(spectral_fields))
    return {**packing, **legend, **spectral}


def _read_fields(asset: pystac.Asset) -> list[BandFields]:
    """Combine STAC 1.0 extension lists or STAC 1.1 core bands by index."""
    core = asset.extra_fields.get("bands", [])
    stored = asset.extra_fields.get("raster:bands", [])
    spectral = asset.extra_fields.get("eo:bands", [])
    count = max(len(core), len(stored), len(spectral))
    result = []
    for index in range(count):
        entry: BandFields = {}
        for listing in (stored, spectral, core):
            if index < len(listing):
                entry.update(listing[index])
        result.append(entry)
    return result


def read_shared_attrs(
    sources: Sequence[tuple[pystac.Item, str, int]],
) -> BandAttrs:
    """Combine published band attrs using native xarray drop-conflicts semantics.

    Args:
        sources: Source Items with their asset keys and zero-based band indices,
            already resolved by the caller. An absent asset states no attrs.

    Returns:
        Combined source attrs, retaining keys absent from some Items.
        Pixels and existing variable attrs are unchanged.

    Warns:
        DroppedAttrsWarning: Source Items disagree on a band's metadata.
    """
    published = []
    for entry, key, index in sources:
        asset = entry.assets.get(key)
        published.append({} if asset is None else read_attrs(asset, index))

    return cast("BandAttrs", attrs.combine_attrs(published))
