"""Ordered asset bands and composition of their schema metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict, cast

import odc.stac
import pystac
import xarray as xr

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
    stated: BandAttrs = {**packing, **legend, **spectral}

    # A one-band asset's title names its band, as publication writes it.
    if "long_name" not in stated and len(entries) == 1 and asset.title:
        stated["long_name"] = asset.title
    return stated


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
    """Combine published band attrs with xarray's drop-conflicts policy.

    Args:
        sources: Source Items with their asset keys and zero-based band indices,
            already resolved by the caller. An absent asset states no attrs.

    Returns:
        Combined source attrs, retaining keys absent from some Items.
        Pixels and existing variable attrs are unchanged.
    """
    published = []
    for entry, key, index in sources:
        asset = entry.assets.get(key)
        published.append({} if asset is None else read_attrs(asset, index))

    observations = []
    for stated in published:
        observations.append(xr.Dataset(attrs=stated))
    combined = xr.merge(observations, combine_attrs="drop_conflicts")
    return cast("BandAttrs", combined.attrs)


_COMMON_FIELDS = ("nodata", "data_type", "unit", "statistics")
_BAND_LISTS = ("raster:bands", "eo:bands", "bands")


def read_band(item: pystac.Item, name: str) -> dict[str, Any]:
    """Read the STAC fields of the band ODC loads as `name`.

    Asset-level fields apply to every band of the asset; the band's own entry
    in `raster:bands`, `eo:bands` or `bands` overrides them.

    Args:
        item: Item holding the band's asset.
        name: Band name or alias, resolved as `odc.stac` resolves it.

    Returns:
        Fields with their STAC spelling; empty if `name` does not resolve.
    """
    parsed = odc.stac.parse_item(item)
    try:
        key, position = parsed.collection.band_key(name)
    except ValueError:
        return {}
    if key not in item.assets or (key, position) not in parsed.bands:
        return {}

    extra = item.assets[key].extra_fields
    fields: dict[str, Any] = {}
    for field, value in extra.items():
        if field in _BAND_LISTS:
            continue
        if field in _COMMON_FIELDS or field.startswith(
            ("raster:", "eo:", "classification:")
        ):
            fields[field] = value

    index = position - 1
    for listing in _BAND_LISTS:
        entries = extra.get(listing, [])
        if index < len(entries):
            fields.update(entries[index])
    return fields
