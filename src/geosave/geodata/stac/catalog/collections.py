"""The Collections an item table carries in its footer: read, reconcile, join."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import pandas as pd

from .layout import FOOTER_KEY

if TYPE_CHECKING:
    import pystac

    from geosave.geodata import GeoDataFrame

# Links that place a Collection in a catalog tree; they do not define it.
NAVIGATION_RELS = ("self", "root", "parent", "child", "item")


def read_collections(attrs: Mapping[Any, Any]) -> dict[str, pystac.Collection]:
    """Return the Collections a table's attrs hold in its footer metadata, by id.

    Args:
        attrs: `rows.attrs` of an item table, or of one of its rows.
    """
    import pystac

    footer = attrs.get(FOOTER_KEY, {})
    collections_by_id = {}
    for collection_id, collection_dict in footer.get("collections", {}).items():
        collections_by_id[collection_id] = pystac.Collection.from_dict(collection_dict)
    return collections_by_id


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

    collections = []
    for catalog in catalogs:
        collections.extend(read_collections(catalog.attrs).values())
    collection_dicts = {}
    for collection in merge_collections(collections).values():
        collection_dicts[collection.id] = collection.to_dict(transform_hrefs=False)

    combined = pd.concat(catalogs, ignore_index=True)
    combined.attrs = {FOOTER_KEY: {"collections": collection_dicts}}
    return cast("GeoDataFrame", combined)
