"""Encode native catalog records and Collections as STAC GeoParquet represents them."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd
    import pystac

    from geosave.geodata import GeoDataFrame


ITEM_COLUMNS = ("id", "links", "assets", "stac_extensions")
FOOTER_KEY = "stac-geoparquet"


def require_item_columns(columns: Iterable[Any]) -> None:
    """Refuse a table that is not an item table, saying where item tables come from.

    Raises:
        ValueError: A stac-geoparquet column is missing.
    """
    missing = set(ITEM_COLUMNS) - set(columns)
    if missing:
        raise ValueError(
            f"STAC Item operations need columns {sorted(missing)}; item tables come "
            "from to_catalog, StacClient.search, or reading a stac-geoparquet file"
        )


def to_item(row: pd.Series) -> pystac.Item:
    """Convert one Item record to PySTAC, reconstructing its geometry bbox."""
    import shapely
    from stac_geoparquet import to_dict

    require_item_columns(row.index)
    item_json = to_dict(row.to_dict())
    item_json["bbox"] = shapely.bounds(row["geometry"]).tolist()
    return build_item(item_json)


def to_items(rows: gpd.GeoDataFrame) -> list[pystac.Item]:
    """Convert Item rows to PySTAC with WGS84 geometry and reconstructed bboxes."""
    import pandas as pd
    import pyarrow as pa
    from stac_geoparquet.arrow import stac_table_to_items

    rows = rows.to_crs("EPSG:4326")
    # GeoPandas names bounds minx..maxy; the Arrow bbox struct uses xmin..ymax.
    bounds = rows.geometry.bounds.rename(
        columns={"minx": "xmin", "miny": "ymin", "maxx": "xmax", "maxy": "ymax"}
    )
    bbox = pd.Series(bounds.to_dict("records"), index=rows.index)
    table = pa.table(rows.assign(bbox=bbox).to_arrow(index=False))
    items = []
    for fields in stac_table_to_items(table):
        items.append(build_item(fields))
    return items


def build_item(item_json: dict[str, Any]) -> pystac.Item:
    """Build a pystac Item from the dict stac-geoparquet makes of one row.

    Args:
        item_json: One row as `stac_geoparquet.to_dict` or
            `stac_table_to_items` returns it.

    Returns:
        New Item without the properties the row leaves empty.
    """
    from datetime import datetime

    import pystac

    properties = {}
    for key, value in item_json["properties"].items():
        # A table holds every collection's fields; the ones this Item's
        # collection lacks are empty.
        if value is None:
            continue
        # A row holds dates as Timestamps, a list element as datetimes; a JSON
        # Item states them as text.
        if isinstance(value, datetime):
            value = value.isoformat()
        properties[key] = value
    item_json["properties"] = properties
    # stac-geoparquet makes these optional; a JSON Item needs them.
    item_json.setdefault("type", "Feature")
    item_json.setdefault("stac_version", pystac.get_stac_version())
    return pystac.Item.from_dict(item_json)


def from_items(
    items: Iterable[pystac.Item], *, collections: Iterable[pystac.Collection] = ()
) -> GeoDataFrame:
    """Lay Items out as an item table, one row per Item, as stac-geoparquet does.

    Item tables otherwise come from `StacClient.search`, `to_catalog` and
    reading a stac-geoparquet file. Use this for Items built or fetched
    some other way. A GeoJSON file of plain features is not one; `open_vector`
    reads it as an ordinary GeoDataFrame.

    Args:
        items: Items whose assets carry absolute hrefs, or a self href that
            resolves relative ones. They are not changed.
        collections: Collections the table carries, as a stac-geoparquet
            file's footer metadata does.

    Returns:
        Item table in WGS84, in the order given, every href absolute where its
        Item could resolve it. No Items give the empty table, with the Item
        columns, that a search matching nothing returns.

    Raises:
        ValueError: Two Items share an id within one collection.

    Examples:
        >>> from_items(searched_items, collections=[collection])["id"].tolist()
        ['S2A_33TUM_20250115', 'S2B_33TUM_20250125']
    """
    from copy import deepcopy

    from stac_geoparquet.arrow import parse_stac_items_to_arrow

    records = []
    identities = set()
    for item in items:
        identity = (item.collection_id, item.id)
        if identity in identities:
            raise ValueError(
                f"duplicate Item identity {identity!r}; give observations distinct "
                "ids with id_format"
            )
        identities.add(identity)
        # A row keeps no self href, so a relative href is resolved while it can be.
        if item.get_self_href() is not None:
            item = item.clone()
            item.make_asset_hrefs_absolute()
        # Transforming link hrefs reads a searched Item's API root over HTTP.
        records.append(item.to_dict(include_self_link=False, transform_hrefs=False))
    if records:
        table = parse_stac_items_to_arrow(records).read_all()
        # Arrow's own column types are kept, as `open_vector` keeps them. The
        # `bbox` covering goes, as geopandas drops it on read: geometry states it.
        rows = gpd.GeoDataFrame.from_arrow(
            table, to_pandas_kwargs={"types_mapper": pd.ArrowDtype}
        ).drop(columns="bbox")
    else:
        rows = gpd.GeoDataFrame(
            columns=pd.Index([*ITEM_COLUMNS, "geometry"]),
            geometry="geometry",
            crs="EPSG:4326",
        )
    stored_collections = {}
    for collection in collections:
        # A copy, so editing the Collection afterwards leaves the table unchanged.
        fields = deepcopy(collection.to_dict(transform_hrefs=False))
        stored_collections[collection.id] = fields
    rows.attrs[FOOTER_KEY] = {"collections": stored_collections}
    return cast("GeoDataFrame", rows)
