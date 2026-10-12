"""STAC objects as xarray ones, one module per level.

A Collection is a Dataset's root ACDD attrs, an Item the dated files of one
scene, an Asset one file and the variables it loads into, a band one
variable's attrs. Each level translates its object into xarray terms and
builds it back with `write`, composing the level below it and the schemas in
`extensions`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import odc.stac

from geosave import __path__ as _package_paths
from geosave.geodata.warnings import DroppedAttrsWarning

import geosave.geodata.attrs as attrs

from . import asset, band, collection, item

if TYPE_CHECKING:
    import pystac
    import xarray as xr

__all__ = ["asset", "band", "collection", "item", "load_attrs"]


def load_attrs(
    items: Sequence[pystac.Item],
    source: pystac.Collection,
    loaded: xr.Dataset,
    *,
    stac_cfg: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Return the loaded Dataset carrying the attrs its Collection and bands state.

    The Collection describes the whole raster, so it lands on the root as
    ACDD; each band describes the variable it loads into, where only what
    every item publishes alike is kept. The loader's own attrs describe the
    pixels it produced, so they win.

    Args:
        items: Items making up one load, in search order.
        source: Collection the items belong to.
        loaded: Dataset odc-stac loaded from them.
        stac_cfg: The same ODC conversion settings used for loading.

    Returns:
        The Dataset, its root and variables carrying their attrs.

    Raises:
        ValueError: `items` is empty.

    Warns:
        DroppedAttrsWarning: The items publish a field differently, or only
            some of them publish it.
    """
    if not items:
        raise ValueError("reading a STAC load's attrs needs at least one item")

    parsed = list(odc.stac.parse_items(items, cfg=stac_cfg))
    described = {}
    for name, variable in loaded.data_vars.items():
        published = []
        for entry, odc_item in zip(items, parsed, strict=True):
            key, index = odc_item.collection.band_key(str(name))
            found = entry.assets.get(key)
            listed = asset.bands(found) if found else []
            # An item may lack the asset, or list fewer bands than the file holds.
            if len(listed) >= index:
                published.append(band.read(*listed[index - 1]))
            else:
                published.append({})
        kept = attrs.shared(published)
        disputed = sorted({key for fields in published for key in fields} - set(kept))
        if disputed:
            values = {
                key: {
                    entry.id: fields.get(key)
                    for entry, fields in zip(items, published, strict=True)
                }
                for key in disputed
            }
            warnings.warn(
                f"STAC items publish {disputed} differently for {str(name)!r}, so "
                f"those source attrs are dropped: {values}",
                DroppedAttrsWarning,
                skip_file_prefixes=tuple(_package_paths),
            )
        stated = variable.copy(deep=False)
        stated.attrs = {**kept, **variable.attrs}
        # odc states its fill as `nodata`; written once, CF's spelling carries it.
        described[name] = attrs.stamp(
            stated, attrs.Nodata, attrs.parse(stated, attrs.Nodata)
        )
    return loaded.assign(described).gs.write_acdd(collection.read(source))
