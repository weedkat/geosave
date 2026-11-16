"""Open and write GeoParquet vector files.

GeoPandas writes and reads the table through PyArrow on the UPath's
filesystem. A STAC Item table is written by `stac_geoparquet`; reading
restores its footer metadata here, since one reader opens both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd
import orjson
import pandas as pd
import pyarrow.parquet as pq
import typing_extensions
from upath import UPath

from geosave.geodata.io import storage
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

SUFFIXES = (".parquet", ".geoparquet")


@dataclass(frozen=True)
class ParquetFooter:
    """What a GeoParquet file's footer states, read without its rows.

    Attributes:
        columns: Column names, in stored order.
        geometry_column: The primary geometry column, or None for a table
            without GeoParquet metadata.
        stac: The `stac-geoparquet` metadata of an Item table, or None.
    """

    columns: tuple[str, ...]
    geometry_column: str | None
    stac: dict[str, Any] | None

    @property
    def holds_items(self) -> bool:
        """Return whether the columns are those of a STAC Item table."""
        from geosave.geodata.stac.catalog import ITEM_COLUMNS

        return all(key in self.columns for key in ITEM_COLUMNS)


def read_footer(path: UPath) -> ParquetFooter:
    """Read a GeoParquet file's footer through its filesystem.

    Args:
        path: The Parquet file.

    Returns:
        The footer's columns, geometry column and STAC metadata.

    Examples:
        >>> read_footer(create_upath("plots.parquet")).geometry_column
        'geometry'
    """
    with path.open("rb") as handle:
        footer = pq.read_metadata(handle)
    metadata = footer.metadata or {}
    geometry_column = None
    if b"geo" in metadata:
        geometry_column = orjson.loads(metadata[b"geo"])["primary_column"]
    stac = None
    if b"stac-geoparquet" in metadata:
        stac = orjson.loads(metadata[b"stac-geoparquet"])
    return ParquetFooter(
        columns=tuple(footer.schema.to_arrow_schema().names),
        geometry_column=geometry_column,
        stac=stac,
    )


class GeoParquetOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when opening GeoParquet.

    Omitted keys use GeoPandas or PyArrow defaults. GeoSave supplies the
    UPath's filesystem explicitly instead of asking PyArrow to choose one.

    Attributes:
        columns: Attribute columns to read; the primary geometry column is
            retained. None reads all columns; [] reads only geometry.
        bbox: Feature filter `(minx, miny, maxx, maxy)` in the stored CRS;
            requires covering-bbox metadata written with `write_covering_bbox`.
        to_pandas_kwargs: Options for converting non-geometry Arrow columns to
            pandas, such as dtype-conversion settings.
        parquet_options: Extra `pyarrow.parquet.read_table` keywords. Do not
            repeat direct options or the filesystem GeoSave supplies.
    """

    columns: Sequence[str] | None
    bbox: tuple[float, float, float, float] | None
    to_pandas_kwargs: Mapping[str, object] | None
    parquet_options: dict[str, object]


def open_geoparquet(
    source: Location,
    *,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[GeoParquetOpenOptions],
) -> GeoDataFrame:
    """Open one local or fsspec-backed GeoParquet file.

    Args:
        source: Local path, fsspec URL, or UPath to GeoParquet.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported GeoPandas and Parquet read options.

    Returns:
        Native GeoDataFrame with its stable `GeoCatalog` accessor. Item
        tables mirror the file's `stac-geoparquet` footer metadata in
        `attrs` and state every asset by absolute href: one a file stores
        relative is resolved against that file.

    Raises:
        TypeError: An option is unsupported, or repeated in `parquet_options`.

    Examples:
        >>> open_geoparquet("plots.parquet", columns=["name"]).columns
        Index(['name', 'geometry'], dtype='object')
    """
    from geosave.geodata.stac.catalog import FOOTER_KEY, resolve_hrefs

    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_options = cast(
        "dict[str, Any]", dict(open_options.pop("parquet_options", {}))
    )
    path = storage.create_upath(source, storage_options)
    footer = read_footer(path)
    columns = open_options.get("columns")
    if columns is not None:
        open_options["columns"] = select_columns(columns, footer)
    if footer.holds_items:
        # Keep Arrow's own column types, so the table goes back to Arrow, and to
        # stac-geoparquet, as it was stored.
        caller_kwargs = open_options.get("to_pandas_kwargs") or {}
        open_options["to_pandas_kwargs"] = {
            "types_mapper": pd.ArrowDtype,
            **caller_kwargs,
        }
    frame = gpd.read_parquet(
        path.path, filesystem=path.fs, **open_options, **parquet_options
    )
    frame = cast("GeoDataFrame", frame)
    if footer.stac is not None:
        frame.attrs[FOOTER_KEY] = footer.stac
    if footer.holds_items and "assets" in frame:
        frame = resolve_hrefs(frame, str(path.absolute()))
    return frame


def select_columns(columns: Sequence[str], footer: ParquetFooter) -> list[str]:
    """Add the geometry column to a column selection that leaves it out.

    Args:
        columns: Attribute columns the caller selects.
        footer: The file's footer.

    Returns:
        The selection, ending in the primary geometry column when it was absent.
    """
    selected = list(columns)
    geometry = footer.geometry_column
    if geometry is not None and geometry not in selected:
        selected.append(geometry)
    return selected


# Closed: a keyword these options do not name is a type error, not a pass-through.
class GeoParquetWriteOptions(typing_extensions.TypedDict, total=False, closed=True):
    """Optional GeoPandas behavior when writing GeoParquet.

    Omitted keys use GeoPandas or PyArrow defaults. These options apply to
    ordinary vectors and STAC catalogs; catalogs require WKB geometry, a bbox
    column and no serialized pandas index.

    Attributes:
        index: Serialize the pandas index when True; False omits it. None
            follows GeoPandas' policy, keeping a RangeIndex in metadata only.
            Catalogs always omit the index and reject True.
        compression: Named Parquet compression codec, or None for no compression.
        geometry_encoding: `"WKB"` for binary geometry or `"geoarrow"` for
            native single-geometry-type encoding in GeoParquet 1.1. Catalogs
            require WKB.
        write_covering_bbox: Write each feature's bounding box and covering
            metadata, enabling filtered reads with `bbox=`. Catalogs require True.
        schema_version: GeoParquet specification version to write. None delegates
            to the selected encoder; encoding features must be supported by
            that version. Catalogs support 1.0.0 and 1.1.0.
        parquet_options: Extra `pyarrow.parquet.write_table` keywords. Do not
            repeat options already supplied directly. Disabling store_schema
            is refused because GeoParquet requires its geometry metadata.
    """

    index: bool | None
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
    geometry_encoding: Literal["WKB", "geoarrow"]
    write_covering_bbox: bool
    schema_version: Literal["0.1.0", "0.4.0", "1.0.0-beta.1", "1.0.0", "1.1.0"] | None
    parquet_options: dict[str, object]


def write_geoparquet(
    gdf: gpd.GeoDataFrame,
    path: Location,
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[GeoParquetWriteOptions],
) -> UPath:
    """Write one feature table to local or fsspec-backed GeoParquet.

    A STAC Item table is written by `write_stac_geoparquet`, which
    `gdf.gs.to_geoparquet` selects from the frame's catalog metadata.

    Args:
        gdf: Feature table to write.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported GeoPandas and Parquet write options.

    Returns:
        The written file, carrying its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported, or repeated in `parquet_options`.
        ValueError: The rows are STAC Items, `store_schema` is disabled, or
            the suffix is wrong.

    Examples:
        >>> write_geoparquet(plots, "plots.parquet", compression="zstd")
        PosixUPath('plots.parquet')
    """
    from geosave.geodata.stac.catalog import FOOTER_KEY, ITEM_COLUMNS

    if FOOTER_KEY in gdf.attrs or all(key in gdf for key in ITEM_COLUMNS):
        raise ValueError(
            "Item rows are written as STAC GeoParquet; write them with "
            "rows.gs.to_geoparquet or write_stac_geoparquet, after "
            "stac.catalog.create_catalog gives them their catalog metadata"
        )
    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_options = cast(
        "dict[str, Any]", dict(write_options.pop("parquet_options", {}))
    )
    if not parquet_options.get("store_schema", True):
        raise ValueError(
            "GeoParquet requires store_schema=True to retain geometry metadata"
        )
    destination = storage.create_upath(path, storage_options)
    storage.check_destination(destination, SUFFIXES, overwrite=overwrite)
    storage.create_parent(destination)
    gdf.to_parquet(
        destination.path,
        filesystem=destination.fs,
        **write_options,
        **parquet_options,
    )
    return destination
