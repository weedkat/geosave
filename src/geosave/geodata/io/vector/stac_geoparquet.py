"""Persist Item tables and embedded Collection definitions as STAC GeoParquet."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import orjson
import pyarrow.parquet as pq
import pystac
from pystac.utils import make_relative_href
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

from geosave.geodata.core import GeoCatalog
from geosave.geodata.io import storage
from geosave.geodata.io.storage import absolute_location

from . import geoparquet

if TYPE_CHECKING:
    from upath import UPath

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.io.storage import Location, StorageOptions


def resolve_asset_hrefs(rows: gpd.GeoDataFrame, path: Location) -> GeoDataFrame:
    """Make the asset hrefs of rows read from an item table openable.

    Args:
        rows: Rows as stored, asset hrefs relative to their file.
        path: The file they were read from, as a path, URL, or UPath.

    Returns:
        The rows with every asset href absolute. Rows that are not an item
        table's are returned as given.
    """
    # A table stating no Items is a caller's own, and stays as written.
    if rows.empty or not isinstance(rows.gs, GeoCatalog):
        return cast("GeoDataFrame", rows)
    from geosave.geodata.stac.translate import table as stac_table

    location = absolute_location(path)
    items = stac_table.to_items(rows)
    for item in items:
        item.set_self_href(location)
    result = stac_table.from_items(items)
    result.attrs = rows.attrs.copy()
    return result


def write(
    rows: gpd.GeoDataFrame,
    path: Location,
    *,
    collections: Iterable[pystac.Collection] | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> UPath:
    """Write an item table as one stac-geoparquet file.

    `rows.gs.to_geoparquet` is the accessor spelling. Asset hrefs are stored
    relative to the file, so a table and its assets move together. Every
    Collection the rows name is stored in the file's metadata with the extent
    of its rows. Add a batch by concatenating its rows onto the table's and
    writing again. Attached Collection definitions are retained automatically.

    Args:
        rows: Item table, as `create_catalog` or `open_vector` returns it.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        collections: Explicit definitions overriding attached definitions by id.
            Every referenced Collection must have a definition; unused ones are
            ignored. Their extents are updated on copies.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        Local writes return a PosixUPath; URL writes return a UPath carrying
        its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: `rows` is not an item table, a referenced Collection definition is missing, or the suffix is wrong.

    Examples:
        >>> forest = create_collection("forest", description="Forest samples")
        >>> files = ds.gs.to_cog("samples/forest_{time:%Y%m%d}.tif")
        >>> items = create_items(files, collection=forest)
        >>> write(create_catalog(items), "data/forest.parquet", collections=[forest])
        PosixUPath('data/forest.parquet')
    """
    if not isinstance(rows.gs, GeoCatalog):
        raise ValueError(
            "to_geoparquet writes an item table as stac-geoparquet; this frame "
            "has no STAC Item columns"
        )
    from ._stac import prepare_collections

    from geosave.geodata.stac.translate import table as stac_table

    location = absolute_location(path)
    items = stac_table.to_items(rows)
    from geosave.geodata.stac.translate.table import validate_ids

    validate_ids(items)
    definitions = rows.gs.collections
    definitions.update({c.id: c for c in collections or ()})
    stored = {
        collection_id: collection.to_dict(
            include_self_link=False, transform_hrefs=False
        )
        for collection_id, collection in prepare_collections(
            items, definitions.values()
        ).items()
    }

    for item in items:
        item.set_self_href(location)
        item.make_asset_hrefs_relative()
        item.remove_links(pystac.RelType.COLLECTION)
        # The Item schema requires a link beside a collection id; a table's
        # Collections live in the table, so the link names its own file.
        if item.collection_id is not None:
            item.add_link(
                pystac.Link(
                    pystac.RelType.COLLECTION, make_relative_href(location, location)
                )
            )
    records = [
        item.to_dict(include_self_link=False, transform_hrefs=False) for item in items
    ]
    table = parse_stac_items_to_arrow(records).read_all()

    return storage.write(
        path,
        lambda target: to_parquet(table, target, collections=stored or None),
        suffixes=geoparquet.FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )


def read_collections(
    path: Location, *, storage_options: StorageOptions | None = None
) -> dict[str, pystac.Collection]:
    """Return the Collections an item table carries, keyed by id.

    Args:
        path: Parquet file, as a path, URL, or UPath.
        storage_options: Options for the filesystem a URL names.

    Returns:
        {
            "<collection id>": its Collection, with the extent of its rows,
        }
        Empty where the file carries none.

    Examples:
        >>> read_collections("data/catalog.parquet")["forest"].license
        'CC-BY-4.0'
    """
    with storage.location(path, storage_options).open("rb") as handle:
        metadata = pq.read_metadata(handle).metadata or {}
    return parse_collections(metadata.get(b"stac-geoparquet", b"{}"))


def parse_collections(value: bytes | str) -> dict[str, pystac.Collection]:
    """Decode the Collections one file's `stac-geoparquet` metadata states.

    Args:
        value: The metadata value, as JSON.

    Returns:
        {
            "<collection id>": its Collection,
        }

    Examples:
        >>> parse_collections(b'{"version": "1.0.0"}')
        {}
    """
    stored = orjson.loads(value).get("collections", {})
    return {
        collection_id: pystac.Collection.from_dict(fields)
        for collection_id, fields in stored.items()
    }
