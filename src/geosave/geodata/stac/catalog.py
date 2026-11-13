"""Encode native catalog records and Collections as STAC GeoParquet represents them."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Required, TypedDict, cast

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd
    import pystac

    from geosave.geodata import GeoDataFrame


ITEM_COLUMNS = ("id", "links", "assets", "stac_extensions")
FOOTER_KEY = "stac-geoparquet"
# The file rows were read from, which their relative asset hrefs resolve against.
SOURCE_KEY = "source"


def merge_collections(
    collections: Iterable[pystac.Collection],
) -> dict[str, pystac.Collection]:
    """Copy Collections by ID, rejecting conflicting definitions.

    Navigation links do not define Collection identity. All other fields,
    including extents and descriptive links, must agree for the same ID.
    Inputs and their links are never changed.
    """
    from copy import deepcopy

    import pystac

    merged = {}
    definitions = {}
    navigation = {"self", "root", "parent", "child", "item"}
    for collection in collections:
        fields = deepcopy(collection.to_dict(transform_hrefs=False))
        definition = deepcopy(fields)
        definition["links"] = [
            link for link in fields.get("links", []) if link["rel"] not in navigation
        ]

        previous = definitions.get(collection.id)
        if previous is not None:
            differences = []
            for field in sorted(previous.keys() | definition.keys()):
                if previous.get(field) != definition.get(field):
                    differences.append(field)
            if differences:
                raise ValueError(
                    f"Collection {collection.id!r} disagrees on: {', '.join(differences)}"
                )
            continue

        definitions[collection.id] = definition
        merged[collection.id] = pystac.Collection.from_dict(fields)

    return merged


def concat_catalogs(catalogs: Sequence[gpd.GeoDataFrame]) -> GeoDataFrame:
    """Concatenate item tables, keeping every Collection their footers define.

    Rows concatenate as native GeoPandas does, which refuses differing CRSs;
    pandas drops footers that differ, so the Collections are merged here.

    Args:
        catalogs: Item tables, as `create_catalog` or reading a file returns them.

    Returns:
        One table of every row, with a fresh index and every Collection.

    Raises:
        ValueError: No table is given, or two define one Collection differently.

    Examples:
        >>> concat_catalogs([optical_rows, radar_rows]).gs.collections.keys()
        dict_keys(['optical', 'radar'])
    """
    if not catalogs:
        raise ValueError("concat_catalogs needs at least one table")

    # Relative hrefs resolve against one file; rows from several cannot share it.
    sources = set()
    for catalog in catalogs:
        if SOURCE_KEY in catalog.attrs:
            sources.add(catalog.attrs[SOURCE_KEY])
    if len(sources) > 1:
        raise ValueError(
            f"the tables were read from different files {sorted(sources)}, "
            "against which their asset hrefs resolve; concatenate their Items "
            "instead: create_catalog(rows_a.gs.to_items() + rows_b.gs.to_items())"
        )

    rows = pd.concat(catalogs, ignore_index=True)

    collections = []
    for catalog in catalogs:
        collections.extend(catalog.gs.collections.values())
    stored = {}
    for collection in merge_collections(collections).values():
        stored[collection.id] = collection.to_dict(transform_hrefs=False)
    rows.attrs = {FOOTER_KEY: {"collections": stored}}
    if sources:
        rows.attrs[SOURCE_KEY] = sources.pop()
    return cast("GeoDataFrame", rows)


def resolve_asset_hrefs(item: pystac.Item, source: str | None) -> None:
    """Make an Item's asset hrefs absolute against the file its row was read from.

    Args:
        item: Item whose relative hrefs a stac-geoparquet file stored.
        source: That file's absolute location; None leaves the hrefs as they are.
    """
    if source is None:
        return
    item.set_self_href(source)
    item.make_asset_hrefs_absolute()
    # The row's file is not the Item's own document.
    item.set_self_href(None)


def to_item(row: pd.Series) -> pystac.Item:
    """Convert one WGS84 Item record to PySTAC, reconstructing its bbox.

    A Series does not carry its parent CRS. Use to_items with a one-row
    GeoDataFrame when the source geometry needs reprojection.
    """
    import shapely
    from stac_geoparquet import to_dict

    from geosave.geodata.core.vector.catalog import stac_items

    row.gs.validate(stac_items)
    item_json = to_dict(row.to_dict())
    item_json["bbox"] = shapely.bounds(row["geometry"]).tolist()
    item = build_item(cast("ItemDocument", item_json))
    resolve_asset_hrefs(item, row.attrs.get(SOURCE_KEY))
    return item


def to_items(rows: gpd.GeoDataFrame) -> list[pystac.Item]:
    """Convert Item rows to PySTAC with WGS84 geometry and reconstructed bboxes."""
    import pandas as pd
    import pyarrow as pa
    from stac_geoparquet.arrow import stac_table_to_items

    import pystac

    from geosave.geodata.core.vector.catalog import stac_items

    rows.gs.validate(stac_items)
    if rows.empty:
        return []
    collections = {}
    for key, fields in rows.attrs.get(FOOTER_KEY, {}).get("collections", {}).items():
        collections[key] = pystac.Collection.from_dict(fields)

    rows = rows.to_crs("EPSG:4326")
    if rows.active_geometry_name != "geometry":
        rows = rows.rename_geometry("geometry")
    # GeoPandas names bounds minx..maxy; the Arrow bbox struct uses xmin..ymax.
    bounds = rows.geometry.bounds.rename(
        columns={"minx": "xmin", "miny": "ymin", "maxx": "xmax", "maxy": "ymax"}
    )
    bbox = pd.Series(bounds.to_dict("records"), index=rows.index)
    table = pa.table(rows.assign(bbox=bbox).to_arrow(index=False))
    items = []
    for fields in stac_table_to_items(table):
        item = build_item(cast("ItemDocument", fields))
        resolve_asset_hrefs(item, rows.attrs.get(SOURCE_KEY))
        if item.collection_id in collections:
            item.set_collection(collections[item.collection_id])
        items.append(item)
    return items


class ItemDocument(TypedDict, total=False):
    """JSON Item fields produced by the STAC GeoParquet decoder."""

    type: str
    stac_version: str
    id: str
    geometry: dict[str, Any] | None
    bbox: list[float] | None
    properties: Required[dict[str, Any]]
    assets: dict[str, dict[str, Any]]
    links: list[dict[str, Any]]
    stac_extensions: list[str]
    collection: str


def build_item(item_json: ItemDocument) -> pystac.Item:
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
    item_json = {**item_json, "properties": properties}
    # stac-geoparquet makes these optional; a JSON Item needs them.
    item_json.setdefault("type", "Feature")
    item_json.setdefault("stac_version", pystac.get_stac_version())
    return pystac.Item.from_dict(dict(item_json))


def create_catalog(
    items: pystac.Item | Sequence[pystac.Item],
    *,
    collections: Iterable[pystac.Collection] = (),
) -> GeoDataFrame:
    """Lay Items out as an item table, one row per Item, as stac-geoparquet does.

    Item tables otherwise come from `StacClient.search`, `to_catalog` and
    reading a stac-geoparquet file. Use this for Items built or fetched
    some other way. A GeoJSON file of plain features is not one; `open_vector`
    reads it as an ordinary GeoDataFrame.

    Args:
        items: One native Item or a sequence of Items. Assets must carry
            absolute hrefs, or an Item self href that resolves relative ones.
            Inputs are not changed.
        collections: Collections the table carries, as a stac-geoparquet
            file's footer metadata does.

    Returns:
        Item table in WGS84, in the order given, every href absolute where its
        Item could resolve it. No Items give the empty table, with the Item
        columns, that a search matching nothing returns.

    Raises:
        ValueError: Two Items share an id within one collection.

    Examples:
        >>> create_catalog(searched_items, collections=[collection])["id"].tolist()
        ['S2A_33TUM_20250115', 'S2B_33TUM_20250125']
    """
    from copy import deepcopy

    from stac_geoparquet.arrow import parse_stac_items_to_arrow
    from pystac.utils import is_absolute_href

    import pystac

    if isinstance(items, pystac.Item):
        items = [items]

    records = []
    identities = set()
    definitions = list(collections)
    for item in items:
        link = item.get_single_link(pystac.RelType.COLLECTION)
        if link is not None and isinstance(link.target, pystac.Collection):
            definitions.append(link.target)
        identity = (item.collection_id, item.id)
        if identity in identities:
            raise ValueError(
                f"duplicate Item identity {identity!r}; give observations distinct "
                "ids with format_id"
            )
        identities.add(identity)
        # A row keeps no self href, so a relative href is resolved while it can be.
        if item.get_self_href() is not None:
            item = item.clone()
            item.make_asset_hrefs_absolute()
        for key, asset in item.assets.items():
            if not is_absolute_href(asset.href):
                raise ValueError(
                    f"Item {item.id!r} asset {key!r} has an unresolved relative href"
                )
        # Transforming link hrefs reads a searched Item's API root over HTTP.
        records.append(item.to_dict(include_self_link=False, transform_hrefs=False))
    if records:
        table = parse_stac_items_to_arrow(
            records, drop_invalid_properties=False
        ).read_all()
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
    for collection in merge_collections(definitions).values():
        # A copy, so editing the Collection afterwards leaves the table unchanged.
        fields = deepcopy(collection.to_dict(transform_hrefs=False))
        stored_collections[collection.id] = fields
    rows.attrs[FOOTER_KEY] = {"collections": stored_collections}
    return cast("GeoDataFrame", rows)
