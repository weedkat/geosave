"""Lay native Items out as a stac-geoparquet table, and join such tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    import pystac

    from geosave.geodata import GeoDataFrame


ITEM_COLUMNS = ("id", "links", "assets", "stac_extensions")
FOOTER_KEY = "stac-geoparquet"
# The file rows were read from, which their relative asset hrefs resolve against.
SOURCE_KEY = "source"

# Links that place a Collection in a catalog tree; they do not define it.
NAVIGATION_RELS = ("self", "root", "parent", "child", "item")


def merge_collections(
    collections: Iterable[pystac.Collection],
) -> dict[str, pystac.Collection]:
    """Keep one copy of each Collection by id, refusing definitions that disagree.

    Navigation links do not define a Collection. Every other field, extents
    and descriptive links included, must agree for one id. Inputs are not
    changed.

    Args:
        collections: Collections, possibly repeating an id.

    Returns:
        The first Collection given for each id, copied.

    Raises:
        ValueError: Two Collections of one id differ; the message names the fields.

    Examples:
        >>> list(merge_collections([optical, optical.clone(), radar]))
        ['optical', 'radar']
    """
    import pystac

    collections_by_id: dict[str, pystac.Collection] = {}
    for collection in collections:
        kept_collection = collections_by_id.get(collection.id)
        if kept_collection is None:
            # `to_dict` builds new containers, so the copy shares nothing.
            collection_dict = collection.to_dict(transform_hrefs=False)
            collections_by_id[collection.id] = pystac.Collection.from_dict(collection_dict)
            continue

        kept_definition = _read_definition(kept_collection)
        definition = _read_definition(collection)
        differing_fields = []
        for field in sorted(kept_definition.keys() | definition.keys()):
            if kept_definition.get(field) != definition.get(field):
                differing_fields.append(field)
        if differing_fields:
            raise ValueError(
                f"Collection {collection.id!r} disagrees on: {', '.join(differing_fields)}"
            )
    return collections_by_id


def _read_definition(collection: pystac.Collection) -> dict[str, object]:
    """Return a Collection's JSON fields without its navigation links."""
    definition = collection.to_dict(transform_hrefs=False)
    defining_links = []
    for link in definition.get("links", []):
        if link["rel"] not in NAVIGATION_RELS:
            defining_links.append(link)
    definition["links"] = defining_links
    return definition


def build_item(
    item_dict: dict[str, Any],
    *,
    source_file: str | None,
    collections_by_id: Mapping[str, pystac.Collection],
) -> pystac.Item:
    """Build a pystac Item from the JSON dict stac-geoparquet makes of one row.

    Args:
        item_dict: One row as `stac_geoparquet.to_dict` or
            `stac_table_to_items` returns it.
        source_file: The file the row was read from, which relative asset
            hrefs resolve against; None leaves the hrefs as they are.
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

    if source_file is not None:
        # The file locates the assets; it is not the Item's own document.
        item.set_self_href(source_file)
        item.make_asset_hrefs_absolute()
        item.set_self_href(None)
    if item.collection_id in collections_by_id:
        item.set_collection(collections_by_id[item.collection_id])
    return item


def concat_catalogs(catalogs: Sequence[gpd.GeoDataFrame]) -> GeoDataFrame:
    """Concatenate item tables, keeping every Collection their footers define.

    Rows concatenate as native GeoPandas does, which refuses differing CRSs;
    pandas drops footers that differ, so the Collections are merged here.

    Args:
        catalogs: Item tables, as `create_catalog` or reading a file returns them.

    Returns:
        One table of every row, with a fresh index and every Collection.

    Raises:
        ValueError: No table is given, two define one Collection differently,
            or the tables were read from different files.

    Examples:
        >>> concat_catalogs([optical_rows, radar_rows]).gs.collections.keys()
        dict_keys(['optical', 'radar'])
    """
    if not catalogs:
        raise ValueError("concat_catalogs needs at least one table")

    # Relative hrefs resolve against one file; rows from several cannot share it.
    source_files = set()
    for catalog in catalogs:
        if SOURCE_KEY in catalog.attrs:
            source_files.add(catalog.attrs[SOURCE_KEY])
    if len(source_files) > 1:
        raise ValueError(
            f"the tables were read from different files {sorted(source_files)}, "
            "against which their asset hrefs resolve; concatenate their Items "
            "instead: create_catalog(rows_a.gs.to_items() + rows_b.gs.to_items())"
        )

    collections = []
    for catalog in catalogs:
        collections.extend(catalog.gs.collections.values())
    collection_dicts = {}
    for collection in merge_collections(collections).values():
        collection_dicts[collection.id] = collection.to_dict(transform_hrefs=False)

    combined = pd.concat(catalogs, ignore_index=True)
    combined.attrs = {FOOTER_KEY: {"collections": collection_dicts}}
    if source_files:
        combined.attrs[SOURCE_KEY] = source_files.pop()
    return cast("GeoDataFrame", combined)


def create_catalog(
    items: Sequence[pystac.Item],
    *,
    collections: Iterable[pystac.Collection] = (),
) -> GeoDataFrame:
    """Lay Items out as an item table, one row per Item, as stac-geoparquet does.

    The stac-geoparquet library builds the columns. Item tables otherwise
    come from `StacClient.search`, `Dataset.gs.to_catalog` and reading a
    stac-geoparquet file.

    Args:
        items: Native Items with absolute asset hrefs, as `RasterStac.to_cog`
            returns them. Inputs are not changed.
        collections: Collections the table carries beside the ones the Items
            link to, as a stac-geoparquet file's footer metadata does.

    Returns:
        Item table in WGS84, in the order given. No Items give the empty
        table, with the Item columns, that a search matching nothing returns.

    Raises:
        ValueError: Two definitions of one Collection disagree.

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
        # Transforming link hrefs reads a searched Item's API root over HTTP.
        item_dicts.append(item.to_dict(include_self_link=False, transform_hrefs=False))

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

    collection_dicts = {}
    for collection in merge_collections(collections).values():
        collection_dicts[collection.id] = collection.to_dict(transform_hrefs=False)
    catalog.attrs[FOOTER_KEY] = {"collections": collection_dicts}
    return cast("GeoDataFrame", catalog)
