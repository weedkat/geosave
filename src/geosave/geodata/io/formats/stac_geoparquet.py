"""Persist Item tables and embedded Collection definitions as STAC GeoParquet."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import orjson
import pyarrow.parquet as pq
import pystac
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

from geosave.geodata.io import storage
from geosave.geodata.io.formats import geoparquet
from geosave.geodata.stac.catalog import (
    FOOTER_KEY,
    merge_collections,
    to_items,
)

if TYPE_CHECKING:
    from upath import UPath

    from geosave.geodata.io.storage import Location, StorageOptions


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
    Collections, reconciled with any supplied definitions, go in the file's footer metadata as
    stac-geoparquet stores their supplied definitions without recalculating extents.

    Args:
        rows: Item table, as `create_catalog` or `open_vector` returns it.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        collections: Additional definitions; conflicting definitions of the same ID fail.
            Ones no row names are ignored; supplied definitions are preserved.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Shared GeoParquet options, with STAC representation
            constraints checked before passing codec and Parquet options on.

    Returns:
        The written file, carrying its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: `rows` is not an item table, or the suffix is wrong.

    Examples:
        >>> catalog = ds.gs.to_catalog("samples", format_paths="forest_{time:%Y%m%d}_{asset}.tif")
        >>> write_stac_geoparquet(catalog, "data/forest.parquet")
        PosixUPath('data/forest.parquet')
    """
    from geosave.geodata.core.vector.catalog import stac_collections, stac_items

    rows.gs.validate(stac_items)
    footer = rows.attrs.get(FOOTER_KEY)
    if not isinstance(footer, Mapping):
        raise ValueError("catalog stac-geoparquet metadata must be a mapping")
    if not isinstance(footer.get("collections", {}), Mapping):
        raise ValueError("catalog metadata collections must be a mapping")

    # The Collections given here join the ones the table states; together
    # they must define every collection a row names.
    stated = list(rows.gs.collections.values())
    stated.extend(collections or ())
    collections_by_id = merge_collections(stated)
    rows = rows.copy()
    rows.attrs[FOOTER_KEY] = {
        **footer,
        "collections": {
            key: collection.to_dict(transform_hrefs=False)
            for key, collection in collections_by_id.items()
        },
    }
    rows.gs.validate(stac_collections)

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

    destination = storage.create_upath(path, storage_options)
    storage.check_destination(
        destination, geoparquet.FILE_SUFFIXES, overwrite=overwrite
    )
    location = str(destination.absolute())
    items = to_items(rows)
    stored = {}
    prepared = prepare_collections(items, collections_by_id.values())
    for collection_id, collection in prepared.items():
        stored[collection_id] = collection.to_dict(transform_hrefs=False)

    for item in items:
        item.set_self_href(location)
        item.make_asset_hrefs_relative()
        link = item.get_single_link(pystac.RelType.COLLECTION)
        if link is not None and link.get_href() is None:
            # The footer carries the definition; no JSON document URL is invented.
            item.remove_links(pystac.RelType.COLLECTION)
    records = [
        item.to_dict(include_self_link=False, transform_hrefs=False) for item in items
    ]
    table = parse_stac_items_to_arrow(records, drop_invalid_properties=False).read_all()
    # The dependency writes each Arrow batch as a row group; keep its footer encoder.
    batches = (
        table
        if row_group_size is None
        else table.to_reader(max_chunksize=row_group_size)
    )

    storage.create_parent(destination)
    to_parquet(
        batches,
        destination.path,
        collections=stored or None,
        filesystem=destination.fs,
        **write_options,
    )
    return destination


def read_collections(
    path: Location, *, storage_options: StorageOptions | None = None
) -> dict[str, pystac.Collection]:
    """Return the Collections an item table carries, keyed by id.

    Args:
        path: Parquet file, as a path, URL, or UPath.
        storage_options: Options for the filesystem a URL names.

    Returns:
        {
            "<collection id>": its supplied Collection definition,
        }
        Empty where the file carries none.

    Examples:
        >>> read_collections("data/catalog.parquet")["forest"].license
        'CC-BY-4.0'
    """
    with storage.create_upath(path, storage_options).open("rb") as handle:
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
    """Copy referenced Collections without changing their authoritative extents."""
    definitions = merge_collections(collections or ())
    referenced = {item.collection_id for item in items}
    return {
        key: collection for key, collection in definitions.items() if key in referenced
    }
