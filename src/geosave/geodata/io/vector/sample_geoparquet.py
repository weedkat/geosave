"""Persist sample tables as GeoParquet, their Collection definitions in the file's metadata.

A sample row is not a STAC Item, so the file is plain GeoParquet; its
Collections sit under the `geosave:collections` key rather than the
`stac-geoparquet` key a STAC GeoParquet file states them under.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson
import pyarrow.parquet as pq

from geosave.geodata.io import storage

from . import geoparquet

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd
    from upath import UPath

    from geosave.geodata.io.storage import Location, StorageOptions

# The file metadata key holding {collection id: Collection dict}.
COLLECTIONS_KEY = b"geosave:collections"


def write(
    rows: gpd.GeoDataFrame,
    path: Location,
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> UPath:
    """Write a sample table as one GeoParquet file.

    `rows.gs.to_geoparquet` is the accessor spelling.

    Args:
        rows: Sample table; `attrs["stac:collections"]` is stored with it.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        Local writes return a PosixUPath; URL writes return a UPath carrying
        its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong.
    """
    collections_json = orjson.dumps(rows.attrs.get("stac:collections", {}))

    def save(target: Path) -> None:
        rows.to_parquet(target, index=False)
        table = pq.read_table(target)
        metadata = {**(table.schema.metadata or {}), COLLECTIONS_KEY: collections_json}
        pq.write_table(table.replace_schema_metadata(metadata), target)

    return storage.write(
        path,
        save,
        suffixes=geoparquet.FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )


def read_collections(
    path: Location, *, storage_options: StorageOptions | None = None
) -> dict[str, dict]:
    """Return the Collection dicts a sample table's file states, keyed by id.

    Args:
        path: Parquet file, as a path, URL, or UPath.
        storage_options: Options for the filesystem a URL names.

    Returns:
        {
            "<collection id>": its Collection as a dict,
        }
        Empty where the file states none.

    Examples:
        >>> sorted(read_collections("data/chips.parquet"))
        ['label', 's2']
    """
    with storage.location(path, storage_options).open("rb") as handle:
        metadata = pq.read_metadata(handle).metadata or {}
    return orjson.loads(metadata.get(COLLECTIONS_KEY, b"{}"))
