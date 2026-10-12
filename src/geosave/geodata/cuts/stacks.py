"""One window per anchor item, joined by the items that cover it."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import orjson
import pystac
import shapely
from affine import Affine
from odc.geo.geobox import GeoBox
from pystac.extensions.projection import ProjectionExtension

from geosave import __path__ as _package_paths
from geosave.geodata.io.vector import stac_geoparquet
from geosave.geodata.stac.translate.item import instant
from geosave.geodata.warnings import (
    DuplicateInstantsWarning,
    PartialCoverWarning,
)

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

# Footprints are WGS84 polygons rebuilt from each grid; a shrink this small
# absorbs float noise without letting a real gap pass.
_TOLERANCE = 1e-7

# Item fields a window row does not copy from its anchor.
_STAC_FIELDS = {
    "type",
    "stac_version",
    "stac_extensions",
    "id",
    "collection",
    "assets",
    "links",
    "bbox",
    "geometry",
    "datetime",
    "start_datetime",
    "end_datetime",
}
_TIME_FIELDS = ("datetime", "start_datetime", "end_datetime")


def stacks(items: gpd.GeoDataFrame, *, on: str) -> GeoDataFrame:
    """List one window per item of collection `on`, on that item's grid.

    Every item of another collection whose footprint covers the anchor's
    joins it, so a window holds what was observed over all of it.

    Args:
        items: Item table, local or searched, of any collections and dates,
            its hrefs absolute as `open_vector` and `search` return them.
        on: Collection whose items are the windows, as labels are.

    Returns:
        One row per anchor item, in table order:

            anchor          the anchor Item's id
            parent          None; a cut names the row it came from here
            sources         JSON {collection: [Item dict, ...]}, trimmed to
                            what reading needs
            proj:code, proj:transform, proj:shape   the anchor's grid
            start_datetime, end_datetime            span of the dated members
            geometry        the anchor's footprint, WGS84
            ...             the anchor Item's own properties

    Raises:
        ValueError: `on` names no collection in the table, an item names no
            collection, or an anchor's assets state no grid.

    Warns:
        DuplicateInstantsWarning: Two items of one collection cover an anchor
            at one instant; the first in table order is kept.
        PartialCoverWarning: Items intersect an anchor without covering it.

    Examples:
        >>> windows = stacks(gs.open_vector("data/catalog.parquet"), on="label")
        >>> windows.iloc[0].gs.to_stack().gs.groups
        ('label', 's2', 's1', 'dem')
    """
    # A window groups its sources by collection, so every item must name one.
    unnamed = items.loc[items["collection"].isna(), "id"].tolist()
    if unnamed:
        raise ValueError(
            f"{len(unnamed)} items name no collection, so no window can group "
            f"them: {unnamed[:5]}; set their collection or drop them"
        )
    present = sorted(items["collection"].unique())
    if on not in present:
        raise ValueError(
            f"no item belongs to the collection {on!r}; the table holds {present}"
        )
    # A boolean filter keeps the geometry; pandas-stubs type it as a DataFrame.
    anchors = cast(
        "gpd.GeoDataFrame", items[items["collection"] == on].reset_index(drop=True)
    )
    others = cast(
        "gpd.GeoDataFrame", items[items["collection"] != on].reset_index(drop=True)
    )
    shrunk = gpd.GeoDataFrame(
        {"anchor_row": range(len(anchors))},
        geometry=shapely.buffer(anchors.geometry.values, -_TOLERANCE),
        crs=items.crs,
    )
    covering = gpd.sjoin(others, shrunk, predicate="covers", how="inner")
    touching = gpd.sjoin(others, shrunk, predicate="intersects", how="inner")
    pairs = set(zip(covering.index, covering["anchor_row"], strict=True))
    partial = sorted(
        {
            index
            for index, row in zip(touching.index, touching["anchor_row"], strict=True)
            if (index, row) not in pairs
        }
    )
    if partial:
        warnings.warn(
            f"{len(partial)} items intersect a window without covering it, so "
            f"they do not join it: {others.loc[partial[:5], 'id'].tolist()}",
            PartialCoverWarning,
            skip_file_prefixes=tuple(_package_paths),
        )

    anchor_items = stac_geoparquet.to_item(anchors)
    other_items = [] if others.empty else stac_geoparquet.to_item(others)
    rows = []
    for position, anchor in enumerate(anchor_items):
        members: dict[str, list[dict[str, Any]]] = {on: [_stripped(anchor)]}
        joined = sorted(covering.index[covering["anchor_row"] == position])
        for each in (other_items[index] for index in joined):
            kept = members.setdefault(cast("str", each.collection_id), [])
            fields = _stripped(each)
            when = instant(fields)
            if when is not None and any(instant(other) == when for other in kept):
                warnings.warn(
                    f"{each.collection_id!r} has two items at {when} over "
                    f"{anchor.id!r}; kept {kept[0]['id']!r}, dropped {each.id!r}",
                    DuplicateInstantsWarning,
                    skip_file_prefixes=tuple(_package_paths),
                )
                continue
            kept.append(fields)
        grid = _grid(anchor)
        dated = [
            stamp
            for listed in members.values()
            for fields in listed
            if (stamp := instant(fields)) is not None
        ]
        properties = anchors.iloc[position]
        rows.append(
            {
                "anchor": anchor.id,
                "parent": None,
                "sources": orjson.dumps(members).decode(),
                "proj:code": str(grid.crs),
                "proj:transform": list(grid.transform)[:6],
                "proj:shape": [grid.shape[0], grid.shape[1]],
                "start_datetime": min(dated) if dated else None,
                "end_datetime": max(dated) if dated else None,
                "geometry": properties.geometry,
                **{
                    str(key): value
                    for key, value in properties.items()
                    if key not in _STAC_FIELDS and not str(key).startswith("proj:")
                },
            }
        )
    return cast(
        "GeoDataFrame", gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    )


def _stripped(item: pystac.Item) -> dict[str, Any]:
    """Keep what reading an Item needs, its hrefs absolute."""
    fields = item.to_dict(include_self_link=False, transform_hrefs=False)
    for key, asset in item.assets.items():
        fields["assets"][key]["href"] = asset.get_absolute_href()
    return {
        "type": fields["type"],
        "stac_version": fields["stac_version"],
        "stac_extensions": fields.get("stac_extensions", []),
        "id": fields["id"],
        "collection": fields.get("collection"),
        "geometry": fields["geometry"],
        "bbox": fields.get("bbox"),
        "properties": {
            key: value
            for key, value in fields["properties"].items()
            if key in _TIME_FIELDS and value is not None
        },
        "assets": fields["assets"],
    }


def _grid(item: pystac.Item) -> GeoBox:
    """Return the grid the anchor's first gridded asset states."""
    for asset in item.assets.values():
        stated = ProjectionExtension.ext(asset)
        if stated.shape is not None and stated.transform is not None:
            crs = stated.code if stated.code is not None else stated.wkt2
            height, width = stated.shape
            return GeoBox((height, width), Affine(*stated.transform[:6]), crs)
    raise ValueError(f"{item.id} states no grid to cut windows on")
