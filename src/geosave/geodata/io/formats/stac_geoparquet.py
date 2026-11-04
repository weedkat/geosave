"""Persist Item tables and embedded Collection definitions as STAC GeoParquet."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pystac
from pystac.utils import make_absolute_href, make_relative_href
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

from geosave.geodata.io import storage
from geosave.geodata.io.formats import geoparquet
from geosave.geodata.io.storage import absolute_location
from geosave.geodata.stac.catalog import FOOTER_KEY, ITEM_COLUMNS

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
        The rows with every asset href absolute, every other column as read.
        Rows that are not an item table's are returned as given.
    """
    if not all(key in rows for key in ITEM_COLUMNS):
        return cast("GeoDataFrame", rows)
    location = absolute_location(path)
    resolved_assets = []
    for assets in rows["assets"]:
        resolved = {}
        for key, asset in assets.items():
            # The assets column holds every key any row has; a row lacking one holds None.
            if asset is None:
                resolved[key] = None
            else:
                resolved[key] = {
                    **asset,
                    "href": make_absolute_href(asset["href"], location),
                }
        resolved_assets.append(resolved)
    # Rebuilt with the column's own Arrow type, which plain dicts would lose.
    assets_type = cast("pd.ArrowDtype", rows["assets"].dtype).pyarrow_dtype
    resolved_column = pd.Series(
        pd.arrays.ArrowExtensionArray(pa.array(resolved_assets, type=assets_type)),
        index=rows.index,
    )
    return cast("GeoDataFrame", rows.assign(assets=resolved_column))


def write_stac_geoparquet(
    rows: gpd.GeoDataFrame,
    path: Location,
    *,
    collections: Iterable[pystac.Collection] | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Any,
) -> UPath:
    """Write an item table as one stac-geoparquet file.

    Selected by `rows.gs.to_geoparquet` from its catalog metadata. Asset hrefs are stored relative to
    the file, so a table and its assets move together. The table's
    Collections, overridden by any given, go in the file's footer metadata as
    stac-geoparquet stores them, each with the extent of its rows.

    Args:
        rows: Item table, as `from_items` or `open_vector` returns it.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        collections: Collections replacing the table's own of the same id.
            Ones no row names are ignored; extents are updated on copies.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Shared GeoParquet options, with STAC representation
            constraints checked before passing codec and Parquet options on.

    Returns:
        Local writes return a PosixUPath; URL writes return a UPath carrying
        its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: `rows` is not an item table, or the suffix is wrong.

    Examples:
        >>> catalog = ds.gs.to_catalog("samples", path_format="forest_{time:%Y%m%d}.tif")
        >>> write_stac_geoparquet(catalog, "data/forest.parquet")
        PosixUPath('data/forest.parquet')
    """
    missing = set(ITEM_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"an item table needs STAC columns {sorted(missing)}")
    footer = rows.attrs.get(FOOTER_KEY)
    if not isinstance(footer, Mapping):
        raise ValueError("catalog stac-geoparquet metadata must be a mapping")
    if not isinstance(footer.get("collections", {}), Mapping):
        raise ValueError("catalog metadata collections must be a mapping")
    if write_options.pop("index", None) not in (None, False):
        raise ValueError("STAC GeoParquet does not serialize a pandas index")
    if write_options.pop("geometry_encoding", "WKB") != "WKB":
        raise ValueError("STAC GeoParquet requires geometry_encoding='WKB'")
    if not write_options.pop("write_covering_bbox", True):
        raise ValueError("STAC GeoParquet requires write_covering_bbox=True")
    schema_version = write_options.pop("schema_version", None)
    if schema_version is not None:
        if schema_version not in ("1.0.0", "1.1.0"):
            raise ValueError("STAC GeoParquet schema_version must be 1.0.0 or 1.1.0")
        write_options["schema_version"] = schema_version
    parquet_options = dict(write_options.pop("parquet_options", {}))
    reserved = {
        "collections",
        "collection_metadata",
        "filesystem",
        "index",
        "geometry_encoding",
        "write_covering_bbox",
    }
    conflicts = set(parquet_options) & (
        set(write_options) | reserved | {"schema_version"}
    )
    if conflicts:
        raise TypeError(
            f"Parquet options repeat or override managed settings: {sorted(conflicts)}"
        )
    write_options.update(parquet_options)
    if not write_options.get("store_schema", True):
        raise ValueError(
            "STAC GeoParquet requires store_schema=True to retain metadata"
        )
    row_group_size = write_options.pop("row_group_size", None)

    catalog = rows.gs
    location = absolute_location(path)
    items = catalog.to_items()
    collections_by_id = catalog.collections
    for collection in collections or ():
        collections_by_id[collection.id] = collection
    stored = {}
    prepared = prepare_collections(items, collections_by_id.values())
    for collection_id, collection in prepared.items():
        stored[collection_id] = collection.to_dict(
            include_self_link=False, transform_hrefs=False
        )

    for item in items:
        item.set_self_href(location)
        item.make_asset_hrefs_relative()
        item.remove_links(pystac.RelType.COLLECTION)
        # STAC requires an Item stating a collection to link to it; the
        # Collection lives in this file's footer, so the link names the file.
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
    # The dependency writes each Arrow batch as a row group; keep its footer encoder.
    batches = (
        table
        if row_group_size is None
        else table.to_reader(max_chunksize=row_group_size)
    )

    return storage.write_file(
        path,
        lambda target: to_parquet(
            batches, target, collections=stored or None, **write_options
        ),
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


def prepare_collections(
    items: list[pystac.Item], collections: Iterable[pystac.Collection] | None
) -> dict[str, pystac.Collection]:
    """Copy the given Collections that Items name, each with the extent of its Items.

    Args:
        items: Items naming their collection ids.
        collections: Collections to copy fields from, by id. They are not
            changed. Ones no Item names are ignored.

    Returns:
        Collection id mapped to a copy with updated extent and absolute asset
        hrefs. Catalog hierarchy links are removed; descriptive links remain.
    """
    import pystac

    by_id = {collection.id: collection for collection in collections or ()}
    grouped: dict[str, list[pystac.Item]] = {}
    for item in items:
        if item.collection_id is not None:
            grouped.setdefault(item.collection_id, []).append(item)

    built = {}
    for collection_id, members in grouped.items():
        if collection_id not in by_id:
            continue
        # Client Collection subclasses require an API root; publish native values.
        fields = by_id[collection_id].to_dict(transform_hrefs=False)
        collection = pystac.Collection.from_dict(fields)
        collection.extent = pystac.Extent.from_items(members)

        collection.make_asset_hrefs_absolute()
        for link in collection.links:
            if link.rel not in ("root", "parent", "self", "item", "child"):
                link.target = link.get_absolute_href() or link.target
        for rel in ("root", "parent", "self", "item", "child"):
            collection.remove_links(rel)

        built[collection_id] = collection

    return built
