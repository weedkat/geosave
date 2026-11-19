"""STAC Items as a stac-geoparquet table: its layout, rules, Collections and Items.

    layout.py       columns, footer key, Arrow form
    hrefs.py        absolute asset hrefs
    collections.py  footer Collections; concat_catalogs
    rules.py        stac_items, stac_collections, one_collection
    items.py        create_catalog, to_items
"""

from .collections import (
    concat_catalogs,
    merge_collections,
    read_collections,
    write_collections,
)
from .hrefs import ABSOLUTE_HREF, resolve_hrefs
from .items import build_item, create_catalog, to_items
from .layout import FOOTER_KEY, ITEM_COLUMNS, read_assets, to_arrow
from .rules import one_collection, stac_collections, stac_items

__all__ = [
    "ABSOLUTE_HREF",
    "FOOTER_KEY",
    "ITEM_COLUMNS",
    "build_item",
    "concat_catalogs",
    "create_catalog",
    "merge_collections",
    "one_collection",
    "read_assets",
    "read_collections",
    "resolve_hrefs",
    "stac_collections",
    "stac_items",
    "to_arrow",
    "to_items",
    "write_collections",
]
