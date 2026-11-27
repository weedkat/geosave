"""Find the STAC asset and metadata for each loaded band."""

from collections.abc import Iterator, Sequence
from typing import Any

import odc.stac
import pystac
import xarray as xr
from odc.stac import ConversionConfig

# STAC 1.1 lets an asset state these fields once for all of its bands.
_ASSET_FIELDS = ("nodata", "data_type", "unit", "statistics")
_ASSET_PREFIXES = ("raster:", "eo:", "classification:")
BAND_LISTS = ("raster:bands", "eo:bands", "bands")


def read_band_fields(asset: pystac.Asset, index: int) -> dict[str, Any]:
    """Read the STAC fields of one band, addressed by its zero-based index.

    An asset can state fields once for all bands. An entry in `raster:bands`,
    `eo:bands` or `bands` overrides the asset's value for that band.
    """
    stated = asset.extra_fields
    fields: dict[str, Any] = {}
    for key, value in stated.items():
        if key in BAND_LISTS:
            continue
        if key in _ASSET_FIELDS or key.startswith(_ASSET_PREFIXES):
            fields[key] = value

    for listing in BAND_LISTS:
        entries = stated.get(listing, [])
        if index < len(entries):
            fields.update(entries[index])

    return fields


def find_band_sources(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> Iterator[tuple[xr.DataArray, list[tuple[pystac.Item, pystac.Asset, int]]]]:
    """Find each loaded variable's source Items, assets and zero-based band indexes.

    ODC resolves asset names, indexed bands and aliases for the supplied
    collection configuration. An Item can lack a band that another Item supplies.
    """
    parsed_items = list(odc.stac.parse_items(items, stac_cfg))

    for name, variable in ds.data_vars.items():
        band_name = str(name)
        sources = []
        for item, parsed in zip(items, parsed_items, strict=True):
            if band_name not in parsed:
                continue

            key, position = parsed.collection.band_key(band_name)
            if key in item.assets:
                sources.append((item, item.assets[key], position - 1))

        yield variable, sources
