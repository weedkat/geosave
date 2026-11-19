"""Convert native Items to item-table rows, and rows back to Items."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import pandas as pd

from .collections import read_collections, write_collections
from .hrefs import ABSOLUTE_HREF
from .layout import ITEM_COLUMNS, to_arrow
from .rules import stac_items

if TYPE_CHECKING:
    import pystac

    from geosave.geodata import GeoDataFrame

# One Item as STAC JSON: open-ended fields under the keys the specification names.
type ItemJSON = dict[str, Any]
ABSOLUTE_HREF_PATTERN = re.compile(ABSOLUTE_HREF)


def create_catalog(
    items: Sequence[pystac.Item],
    *,
    collections: Iterable[pystac.Collection] = (),
) -> GeoDataFrame:
    """Lay Items out as an item table, one row per Item, as stac-geoparquet does.

    The stac-geoparquet library builds the columns. Item tables otherwise
    come from `StacClient.search`, `create_catalog` over published Items and reading a
    stac-geoparquet file.

    Args:
        items: Native Items, as `stac.publish_item` returns them. A relative
            asset href is made absolute against its Item's own location.
            Inputs are not changed.
        collections: Collections the table carries beside the ones the Items
            link to, as a stac-geoparquet file's footer metadata does.

    Returns:
        Item table in WGS84, in the order given. No Items give the empty
        table, with the Item columns, that a search matching nothing returns.

    Raises:
        ValueError: Two definitions of one Collection disagree, or an asset
            href is relative and its Item states no location.

    Examples:
        >>> create_catalog(items)["id"].tolist()
        ['20250601T103000', '20250602T103000']
    """
    import pystac
    from stac_geoparquet.arrow import parse_stac_items_to_arrow

    item_dicts = []
    collections = list(collections)
    for item in items:
        collection_link = item.get_single_link(pystac.RelType.COLLECTION)
        if collection_link is not None:
            if isinstance(collection_link.target, pystac.Collection):
                collections.append(collection_link.target)
        item_dicts.append(to_item_dict(item))

    if item_dicts:
        arrow_table = parse_stac_items_to_arrow(
            item_dicts, drop_invalid_properties=False
        ).read_all()
        # Arrow's own column types are kept, as `open_vector` keeps them. The
        # `bbox` covering goes, as geopandas drops it on read: geometry states it.
        catalog = gpd.GeoDataFrame.from_arrow(
            arrow_table, to_pandas_kwargs={"types_mapper": pd.ArrowDtype}
        ).drop(columns="bbox")
    else:
        catalog = gpd.GeoDataFrame(
            columns=pd.Index([*ITEM_COLUMNS, "geometry"]),
            geometry="geometry",
            crs="EPSG:4326",
        )

    write_collections(catalog, collections)
    return cast("GeoDataFrame", catalog)


def to_item_dict(item: pystac.Item) -> ItemJSON:
    """Return an Item's STAC JSON as a table stores it, without changing the Item.

    Asset hrefs are absolute, and links naming no document are left out: the
    table's footer carries the Collection.

    Raises:
        ValueError: An asset href is relative and the Item states no location.
    """
    # Transforming link hrefs reads a searched Item's API root over HTTP.
    item_dict = item.to_dict(include_self_link=False, transform_hrefs=False)

    for key, asset in item.assets.items():
        if ABSOLUTE_HREF_PATTERN.match(asset.href):
            continue
        href = asset.get_absolute_href()
        if href is None:
            raise ValueError(
                f"Item {item.id!r}: asset {key!r} has the relative href "
                f"{asset.href!r} and the Item states no location to resolve it against"
            )
        item_dict["assets"][key]["href"] = href

    located_links = []
    for link in item_dict["links"]:
        if link.get("href"):
            located_links.append(link)
    item_dict["links"] = located_links
    return item_dict


def to_items(rows: gpd.GeoDataFrame) -> list[pystac.Item]:
    """Convert Item rows to native pystac Items; `rows.gs.to_items` calls this.

    Returns:
        One Item per row, in row order, with WGS84 geometry, linked to the
        Collection the table defines for it.

    Raises:
        VectorValidationError: The rows are not STAC Items.
    """
    from stac_geoparquet.arrow import stac_table_to_items

    stac_items(rows)
    if rows.empty:
        return []

    collections_by_id = read_collections(rows.attrs)
    items = []
    for item_dict in stac_table_to_items(to_arrow(rows)):
        items.append(build_item(item_dict, collections_by_id))
    return items


def build_item(
    item_dict: ItemJSON,
    collections_by_id: Mapping[str, pystac.Collection],
) -> pystac.Item:
    """Build a pystac Item from the JSON dict stac-geoparquet makes of one row.

    Args:
        item_dict: One row as `stac_geoparquet.to_dict` or
            `stac_table_to_items` returns it.
        collections_by_id: The table's Collections; the Item links to its own.

    Returns:
        New Item without the properties the row leaves empty.
    """
    from datetime import datetime

    import pystac

    stated_properties = {}
    for key, value in item_dict["properties"].items():
        # A table holds every collection's fields; the ones this Item's
        # collection lacks are empty.
        if value is None:
            continue
        # A row holds dates as Timestamps; a JSON Item states them as text.
        if isinstance(value, datetime):
            value = value.isoformat()
        stated_properties[key] = value
    # stac-geoparquet makes type and version optional; a JSON Item needs them.
    item = pystac.Item.from_dict(
        {
            "type": "Feature",
            "stac_version": pystac.get_stac_version(),
            **item_dict,
            "properties": stated_properties,
        }
    )

    if item.collection_id in collections_by_id:
        item.set_collection(collections_by_id[item.collection_id])
    return item
