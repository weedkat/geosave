"""STAC GeoParquet codec: an Item table as Arrow rows and footer Collections.

The stac-geoparquet library writes the Arrow table and its footer. This
module hands it GeoSave's item table and the Collection definitions its rows
name; `geoparquet.open_geoparquet` reads the file back.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import orjson
from stac_geoparquet.arrow import to_parquet

from geosave.geodata.io import storage
from geosave.geodata.io.vector import geoparquet
from geosave.geodata.stac.catalog import (
    FOOTER_KEY,
    merge_collections,
    read_collections,
    stac_collections,
    stac_items,
    to_arrow,
)

if TYPE_CHECKING:
    import pystac
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

    Selected by `rows.gs.to_geoparquet` from its catalog metadata. Rows are
    written as they are, with their absolute asset hrefs, so any
    stac-geoparquet reader locates the assets. The table's Collections,
    reconciled with any supplied definitions, go in the file's footer
    metadata with their extents as given.

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
        >>> write_stac_geoparquet(catalog, "data/forest.parquet")
        PosixUPath('data/forest.parquet')
    """
    stac_items(rows)
    definitions = merge_stated_collections(rows, collections)
    parquet_options = resolve_parquet_options(**write_options)
    row_group_size = parquet_options.pop("row_group_size", None)
    destination = storage.create_upath(path, storage_options)
    storage.check_destination(destination, geoparquet.SUFFIXES, overwrite=overwrite)

    named = set()
    if "collection" in rows:
        named = set(rows["collection"].dropna())
    stored = {}
    for collection_id, collection in definitions.items():
        if collection_id in named:
            stored[collection_id] = collection.to_dict(transform_hrefs=False)

    table = to_arrow(rows)
    # The dependency writes each Arrow batch as a row group; keep its footer encoder.
    batches = table
    if row_group_size is not None:
        batches = table.to_reader(max_chunksize=row_group_size)

    storage.create_parent(destination)
    to_parquet(
        batches,
        destination.path,
        collections=stored or None,
        filesystem=destination.fs,
        **parquet_options,
    )
    return destination


def merge_stated_collections(
    rows: gpd.GeoDataFrame, collections: Iterable[pystac.Collection] | None
) -> dict[str, pystac.Collection]:
    """Join the Collections a table states with the ones supplied for it.

    Together they must define every collection a row names.

    Args:
        rows: Item table carrying its catalog metadata.
        collections: Additional definitions.

    Returns:
        {
            "<collection id>": its one reconciled definition,
        }

    Raises:
        ValueError: The table's catalog metadata is malformed, two
            definitions of one id conflict, or a row names an undefined one.
    """
    footer = rows.attrs.get(FOOTER_KEY)
    if not isinstance(footer, Mapping):
        raise ValueError("catalog stac-geoparquet metadata must be a mapping")
    if not isinstance(footer.get("collections", {}), Mapping):
        raise ValueError("catalog metadata collections must be a mapping")

    stated = list(read_collections(rows.attrs).values())
    stated.extend(collections or ())
    definitions = merge_collections(stated)
    fields = {}
    for collection_id, collection in definitions.items():
        fields[collection_id] = collection.to_dict(transform_hrefs=False)
    merged = rows.copy()
    merged.attrs[FOOTER_KEY] = {**footer, "collections": fields}
    stac_collections(merged)
    return definitions


# Settings the STAC encoding fixes, and the only value each may be given as.
_FIXED_OPTIONS = {
    "index": False,
    "geometry_encoding": "WKB",
    "write_covering_bbox": True,
}
# Keywords this writer supplies to the dependency itself.
_MANAGED_OPTIONS = {
    "collections",
    "collection_metadata",
    "filesystem",
    "schema_version",
}


def resolve_parquet_options(**write_options: Any) -> dict[str, Any]:
    """Check GeoParquet options against the STAC encoding and flatten them.

    Args:
        **write_options: Shared GeoParquet write options.

    Returns:
        Keywords for `stac_geoparquet.arrow.to_parquet`, a dependency API
        taking open-ended Parquet keywords, plus `row_group_size` when given.

    Raises:
        ValueError: An option contradicts the STAC encoding.
        TypeError: `parquet_options` repeats or overrides a managed setting.

    Examples:
        >>> resolve_parquet_options(compression="zstd", index=False)
        {'compression': 'zstd'}
    """
    options = dict(write_options)
    for key, required in _FIXED_OPTIONS.items():
        given = options.pop(key, None)
        if given is not None and given != required:
            raise ValueError(f"STAC GeoParquet requires {key}={required!r}")
    if options.get("schema_version", "1.1.0") not in ("1.0.0", "1.1.0"):
        raise ValueError("STAC GeoParquet schema_version must be 1.0.0 or 1.1.0")
    if options.get("schema_version") is None:
        options.pop("schema_version", None)
    parquet_options = dict(options.pop("parquet_options", {}))
    conflicts = set(parquet_options) & (
        set(options) | set(_FIXED_OPTIONS) | _MANAGED_OPTIONS
    )
    if conflicts:
        raise TypeError(
            f"Parquet options repeat or override managed settings: {sorted(conflicts)}"
        )
    options.update(parquet_options)
    if not options.get("store_schema", True):
        raise ValueError(
            "STAC GeoParquet requires store_schema=True to retain metadata"
        )
    return options


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
    return read_collections({FOOTER_KEY: orjson.loads(value)})
