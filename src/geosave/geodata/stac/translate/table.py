"""Pure conversion between native Items and GeoDataFrame rows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pystac
import rustac
from stac_geoparquet.arrow import parse_stac_items_to_arrow

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core import GeoCatalog


def from_items(
    items: pystac.Item | Iterable[pystac.Item],
    *,
    collections: Iterable[pystac.Collection] = (),
) -> GeoDataFrame[GeoCatalog]:
    """Build a table with one row per Item.

    Args:
        items: One Item or many, whose assets carry absolute hrefs, or a self
            href that resolves relative ones. They are not changed.
        collections: Definitions copied onto the table, keyed by Collection id.

    Returns:
        GeoDataFrame in WGS84 with the Items' properties as columns and their
        assets as mappings, every href absolute where its Item could resolve
        it. No Items give the empty table a search matching nothing returns.

    Examples:
        >>> root = pystac.Catalog.from_file("data/stac/catalog.json")
        >>> from_items(root.get_items(recursive=True))["id"].tolist()
        ['forest_20250601', 'forest_20250611']
        >>> from_items(next(root.get_items(recursive=True)))["id"].tolist()
        ['forest_20250601']
    """
    if isinstance(items, pystac.Item):
        items = [items]
    records = []
    for item in items:
        # A row keeps no self href, so a relative href is resolved while it can be.
        if item.get_self_href() is not None:
            item = item.clone()
            item.make_asset_hrefs_absolute()
        # Transforming link hrefs reads a searched Item's API root over HTTP.
        records.append(item.to_dict(include_self_link=False, transform_hrefs=False))
    if not records:
        rows = gpd.GeoDataFrame(
            columns=pd.Index(["type", "stac_version", "id", "assets", "geometry"]),
            geometry="geometry",
            crs="EPSG:4326",
        )
    else:
        table = parse_stac_items_to_arrow(records).read_all()
        # geopandas drops this covering column on read, so no item table carries it.
        rows = gpd.GeoDataFrame.from_arrow(table).drop(columns="bbox")

    rows.attrs["stac:collections"] = {
        entry.id: deepcopy(entry.to_dict(transform_hrefs=False))
        for entry in collections
    }
    return cast("GeoDataFrame[GeoCatalog]", rows)


def to_items(rows: gpd.GeoDataFrame) -> list[pystac.Item]:
    """Build the Item each row of an item table states.

    Args:
        rows: Rows of an item table, as `from_items` or `open_vector` returns
            them, filtered or concatenated like any GeoDataFrame.

    Returns:
        One new Item per row, in row order, holding only the assets its row has.

    Examples:
        >>> [item.id for item in to_items(rows[rows["collection"] == "label"])]
        ['s0/label']
    """
    # GeoPandas drops STAC's bbox column; restore it for rustac.
    bounds = rows.geometry.bounds
    bounds.columns = ["xmin", "ymin", "xmax", "ymax"]
    boxes = pd.Series(bounds.to_dict("records"), index=rows.index)

    # A filtered table's index is bookkeeping, not an Item property.
    table = pa.table(rows.assign(bbox=boxes).to_arrow(index=False))
    found = rustac.from_arrow(table)
    return [pystac.Item.from_dict(fields) for fields in found["features"]]


def validate_ids(items: Sequence[pystac.Item]) -> None:
    """Reject repeated Item identities within a Collection."""
    # An id is unique within its collection, so an image and its label may share one.
    seen: set[tuple[str | None, str]] = set()
    for item in items:
        key = (item.collection_id, item.id)
        if key in seen:
            raise ValueError(
                f"two Items of collection {item.collection_id!r} share the id "
                f"{item.id!r}; give each its own, as a template with "
                f"{{start:%Y%m%d}}"
            )
        seen.add(key)
