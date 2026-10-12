"""stac-geoparquet: item tables in memory and in one file."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pystac
import rustac
from pystac.utils import make_relative_href
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

from geosave.geodata.core.catalog import ITEM_COLUMNS, is_item_table
from geosave.geodata.io import storage
from geosave.geodata.io.storage import absolute_location

from . import geoparquet

if TYPE_CHECKING:
    from upath import UPath

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.io.storage import Location, StorageOptions


def from_item(items: pystac.Item | Iterable[pystac.Item]) -> GeoDataFrame:
    """Build a table with one row per Item.

    Args:
        items: One Item or many, whose assets carry absolute hrefs, or a self
            href that resolves relative ones. They are not changed.

    Returns:
        GeoDataFrame in WGS84 with the Items' properties as columns and their
        assets as mappings, every href absolute where its Item could resolve
        it. No Items give the empty table a search matching nothing returns.

    Examples:
        >>> root = pystac.Catalog.from_file("data/stac/catalog.json")
        >>> from_item(root.get_items(recursive=True))["id"].tolist()
        ['forest_20250601', 'forest_20250611']
        >>> from_item(next(root.get_items(recursive=True)))["id"].tolist()
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
        return cast(
            "GeoDataFrame",
            gpd.GeoDataFrame(
                columns=pd.Index([*ITEM_COLUMNS, "geometry"]),
                geometry="geometry",
                crs="EPSG:4326",
            ),
        )
    table = parse_stac_items_to_arrow(records).read_all()
    # geopandas drops this covering column on read, so no item table carries it.
    return cast("GeoDataFrame", gpd.GeoDataFrame.from_arrow(table).drop(columns="bbox"))


def to_item(rows: gpd.GeoDataFrame) -> list[pystac.Item]:
    """Build the Item each row of an item table states.

    Args:
        rows: Rows of an item table, as `from_item` or `open_vector` returns
            them, filtered or concatenated like any GeoDataFrame.

    Returns:
        One new Item per row, in row order, holding only the assets its row has.

    Examples:
        >>> [item.id for item in to_item(rows[rows["collection"] == "label"])]
        ['s0/label']
    """
    # rustac reads the table as geopandas left it, including all-null columns.
    found = rustac.from_arrow(_arrow(rows))
    return [pystac.Item.from_dict(fields) for fields in found["features"]]


def resolve(rows: gpd.GeoDataFrame, path: Location) -> GeoDataFrame:
    """Make the asset hrefs of rows read from an item table openable.

    Args:
        rows: Rows as stored, asset hrefs relative to their file.
        path: The file they were read from, as a path, URL, or UPath.

    Returns:
        The rows with every asset href absolute. Rows that are not an item
        table's are returned as given.
    """
    # A table stating no Items is a caller's own, and stays as written.
    if rows.empty or not is_item_table(rows):
        return cast("GeoDataFrame", rows)
    location = absolute_location(path)
    items = to_item(rows)
    for item in items:
        item.set_self_href(location)
    return from_item(items)


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
    writing again, passing the Collections the table already holds.

    Args:
        rows: Item table, as `from_item` or `open_vector` returns it.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        collections: Collections the rows belong to. They are not changed. A
            collection the rows name and this omits is stored with its id as
            its description.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        Local writes return a PosixUPath; URL writes return a UPath carrying
        its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: `rows` is not an item table, a given Collection has no
            rows, or the suffix is wrong.

    Examples:
        >>> forest = create_collection("forest", description="Forest samples")
        >>> files = ds.gs.to_cog("samples/forest_{time:%Y%m%d}.tif")
        >>> write(create_item(files, collection=forest), "data/forest.parquet", collections=[forest])
        PosixUPath('data/forest.parquet')
    """
    if not is_item_table(rows):
        raise ValueError(
            "to_geoparquet writes an item table as stac-geoparquet; this frame "
            "has no STAC Item columns"
        )
    # Deferred: the STAC package imports this module while it loads.
    from geosave.geodata.stac.translate.collection import describe

    location = absolute_location(path)
    items = to_item(rows)
    stored = {
        collection_id: collection.to_dict(
            include_self_link=False, transform_hrefs=False
        )
        for collection_id, collection in describe(items, collections).items()
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


def get_collections(
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
        >>> get_collections("data/catalog.parquet")["forest"].license
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


def _arrow(rows: gpd.GeoDataFrame) -> pa.Table:
    """Spell rows as the Arrow table the format states, bounds and all."""
    # A STAC bbox is its geometry's bounds; geopandas dropped the column on read.
    bounds = rows.geometry.bounds
    bounds.columns = ["xmin", "ymin", "xmax", "ymax"]
    boxes = pd.Series(bounds.to_dict("records"), index=rows.index)
    # A filtered table's index is the caller's bookkeeping, not an Item field.
    return pa.table(rows.assign(bbox=boxes).to_arrow(index=False))
