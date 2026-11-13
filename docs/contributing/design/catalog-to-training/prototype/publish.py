# ruff: noqa
# Throwaway prototype for docs/contributing/design/catalog-to-training.md; not library code.
"""Prototype: publish a collection Dataset as COGs plus Items built from its metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pystac
import xarray as xr
from shapely.geometry import mapping

import geosave as gs  # noqa: F401
from geosave.geodata import attrs
from geosave.geodata.stac.extensions import classification, eo, projection, raster
from geosave.geodata.stac.item import create_item, normalize_time


def to_catalog(
    collection: xr.Dataset,
    root: Path,
    *,
    collection_id: str,
    format_id: str = "{time:%Y%m%dT%H%M%S}",
):
    """Write each observation's variables as COGs; return the Item table describing them."""
    from geosave.geodata.stac import create_catalog

    tree = collection.gs.expand_stac(format_id=format_id)
    paths = tree.gs.to_cog(root, overwrite=True)
    written = {}
    for path in paths:
        written[(Path(path).parent.name, Path(path).stem)] = str(path)

    items = []
    for node in tree.children.values():
        coords = node.to_dataset(inherit="all_coords").coords
        instant = None
        if "time" in coords:
            instant = (
                pd.Timestamp(coords["time"].values).tz_localize("UTC").to_pydatetime()
            )
        start = end = None
        if "start_datetime" in coords:
            start = (
                pd.Timestamp(coords["start_datetime"].values)
                .tz_localize("UTC")
                .to_pydatetime()
            )
            end = (
                pd.Timestamp(coords["end_datetime"].values)
                .tz_localize("UTC")
                .to_pydatetime()
            )
        properties = {}
        if "stac_properties" in coords:
            import orjson

            properties = orjson.loads(str(coords["stac_properties"].values))
            for key in ("datetime", "start_datetime", "end_datetime"):
                properties.pop(key, None)
        time = normalize_time(instant, start, end)
        item = create_item(
            id=str(node.name),
            time=time,
            properties=properties,
            collection=collection_id,
        )
        grid = None
        for leaf in node.children.values():
            data = leaf.to_dataset(inherit="all_coords")
            ((name, variable),) = data.data_vars.items()
            grid = data.gs.geobox
            asset = pystac.Asset(
                written[(str(node.name), str(leaf.name))],
                media_type=pystac.MediaType.COG,
                roles=["data"],
            )
            item.add_asset(str(leaf.name), asset)
            projection.write_asset(asset, grid)
            band = raster.build_band(variable)
            classification.write_band(
                asset,
                band,
                attrs.CFFlags.from_attrs(variable.attrs),
                attrs.Display.from_attrs(variable.attrs),
            )
            raster.write_asset(asset, [band])
            eo.write_asset(asset, [eo.build_band(variable)])
        footprint = grid.extent.to_crs("EPSG:4326").geom
        item.geometry = mapping(footprint)
        item.bbox = list(footprint.bounds)
        items.append(item)

    definition = pystac.Collection(
        collection_id,
        f"{collection_id} published by geosave",
        pystac.Extent(
            pystac.SpatialExtent([[-180, -90, 180, 90]]),
            pystac.TemporalExtent([[None, None]]),
        ),
    )
    return create_catalog(items, collections=[definition])
