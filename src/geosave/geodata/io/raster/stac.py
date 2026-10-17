"""Open the raster assets described by native STAC Items."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import pystac
from odc.geo.geobox import GeoBox

from geosave.geodata.stac.translate import band, collection, item
from geosave.geodata.transform.warp import Resampling

from . import assemble
from ..storage import location

if TYPE_CHECKING:
    import xarray as xr

type ChunkSpec = dict[str, int] | int | Literal["auto"] | None


def open_raster(
    items: Sequence[pystac.Item],
    *,
    geobox: GeoBox | None = None,
    resampling: Resampling | Mapping[str, Resampling] = "nearest",
    chunks: ChunkSpec = {},
    collection_fields: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Open data assets and assemble their variables and observations.

    Args:
        items: Items of one collection with resolvable asset hrefs.
        geobox: Optional grid to assemble the assets onto.
        resampling: Kernel or per-variable mapping, with ``"*"`` as fallback.
        chunks: File-opening chunks; an empty mapping uses native lazy chunks.
        collection_fields: Optional serialized Collection describing the raster.

    Returns:
        Dataset carrying file metadata supplemented by STAC metadata.
    """
    rasters = []
    for entry in items:
        names = item.read_variable_names(entry)
        for key, asset in item.select_data_assets(entry).items():
            raster = _open_asset(asset, names[key], chunks, key)
            rasters.append(item.assign_time(raster, entry))

    raster = assemble.from_rasters(rasters, geobox=geobox, resampling=resampling)

    if collection_fields is not None:
        source = pystac.Collection.from_dict(dict(collection_fields))
        raster = raster.assign_attrs(collection.read_attrs(source))

    return raster


def _open_asset(
    asset: pystac.Asset,
    names: Sequence[str],
    chunks: ChunkSpec,
    key: str,
) -> xr.Dataset:
    """Open one asset and apply its band identities and published metadata."""
    from ..openers import open_raster

    href = asset.get_absolute_href() or asset.href
    raster = open_raster(location(href), chunks=chunks)

    raster = assemble.rename_bands(raster, names, source=href, fallback=key)
    for index, name in enumerate(raster.data_vars):
        variable = raster[name]
        metadata = band.read_attrs(variable, asset, index)
        raster[name] = variable.assign_attrs({**metadata, **variable.attrs})

    return raster
