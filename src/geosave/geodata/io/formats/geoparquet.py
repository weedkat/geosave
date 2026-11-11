"""Open and write GeoParquet vector files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
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
    import pystac

    from geosave.geodata import GeoDataFrame

FILE_SUFFIXES = (".parquet", ".geoparquet")


class GeoParquetOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when opening GeoParquet.

    Omitted keys use GeoPandas or PyArrow defaults. GeoSave supplies the
    UPath's filesystem explicitly instead of asking PyArrow to choose one.

    Attributes:
        columns: Attribute columns to read; the primary geometry column is
            retained. None reads all columns; [] reads only geometry.
        bbox: Feature filter `(minx, miny, maxx, maxy)` in the stored CRS;
            requires covering-bbox metadata written with `write_covering_bbox`.
        storage_options: Credentials and connection options merged into the
            source UPath and used by its filesystem.
        to_pandas_kwargs: Options for converting non-geometry Arrow columns to
            pandas, such as dtype-conversion settings.
        parquet_options: Extra `pyarrow.parquet.read_table` keywords. Do not
            repeat direct options or the filesystem GeoSave supplies.
    """

    columns: Sequence[str] | None
    bbox: tuple[float, float, float, float] | None
    storage_options: StorageOptions | None
    to_pandas_kwargs: Mapping[str, object] | None
    parquet_options: dict[str, object]


def open_geoparquet(
    source: Location,
    **open_options: Unpack[GeoParquetOpenOptions],
) -> GeoDataFrame:
    """Open one local or fsspec-backed GeoParquet file.

    Args:
        source: Local path, fsspec URL, or UPath to GeoParquet.
        **open_options: Supported GeoPandas and Parquet read options.

    Returns:
        Native GeoDataFrame with its stable `GeoCatalog` accessor. Item
        tables restore absolute asset hrefs, and mirror the file's
        `stac-geoparquet` footer metadata in `attrs`.

    Raises:
        TypeError: An option is unsupported, or repeated in `parquet_options`.
    """
    from geosave.geodata.stac.catalog import FOOTER_KEY, ITEM_COLUMNS

    storage_options = open_options.pop("storage_options", None)
    parquet_options = dict(open_options.pop("parquet_options", {}))
    parquet = storage.location(source, storage_options)
    with parquet.open("rb") as handle:
        footer = pq.read_metadata(handle)
        metadata = footer.metadata or {}
        item_columns = all(
            key in footer.schema.to_arrow_schema().names for key in ITEM_COLUMNS
        )
    columns = open_options.get("columns")
    if columns is not None and b"geo" in metadata:
        geometry = orjson.loads(metadata[b"geo"])["primary_column"]
        if geometry not in columns:
            open_options["columns"] = [*columns, geometry]
    if item_columns:
        # Keep Arrow's own column types, so the table goes back to Arrow, and to
        # stac-geoparquet, as it was stored.
        caller_kwargs = open_options.pop("to_pandas_kwargs", None) or {}
        open_options["to_pandas_kwargs"] = {
            "types_mapper": pd.ArrowDtype,
            **caller_kwargs,
        }
    frame = cast(
        "GeoDataFrame",
        gpd.read_parquet(
            parquet.path,
            filesystem=parquet.fs,
            **open_options,
            **parquet_options,
        ),
    )
    if b"stac-geoparquet" in metadata:
        frame.attrs[FOOTER_KEY] = orjson.loads(metadata[b"stac-geoparquet"])
        if all(key in frame for key in ITEM_COLUMNS):
            from geosave.geodata.io.formats import stac_geoparquet

            frame = stac_geoparquet.resolve_asset_hrefs(frame, parquet)
    return frame


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
        storage_options: Credentials and connection options for the destination
            filesystem, including uploads after local file creation.
        parquet_options: Extra `pyarrow.parquet.write_table` keywords. Do not
            repeat options already supplied directly. Disabling store_schema
            is refused because GeoParquet requires its geometry metadata.
    """

    index: bool | None
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
    geometry_encoding: Literal["WKB", "geoarrow"]
    write_covering_bbox: bool
    schema_version: Literal["0.1.0", "0.4.0", "1.0.0-beta.1", "1.0.0", "1.1.0"] | None
    storage_options: StorageOptions | None
    parquet_options: dict[str, object]


def write_geoparquet(
    gdf: gpd.GeoDataFrame,
    path: Location,
    *,
    collections: Iterable[pystac.Collection] | None = None,
    overwrite: bool = False,
    **write_options: Unpack[GeoParquetWriteOptions],
) -> UPath:
    """Write one GeoDataFrame to local or fsspec-backed GeoParquet.

    Args:
        gdf: Frame to write. The stac-geoparquet attrs key selects catalog encoding.
        collections: Catalog definitions overriding the frame's by id.
            Ordinary features take none.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        overwrite: Replace an existing file when true.
        **write_options: Supported GeoPandas and Parquet write options.

    Returns:
        Local writes return a PosixUPath; URL writes return a UPath carrying
        its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported, or repeated in `parquet_options`.
        ValueError: Catalog metadata is missing, catalog fields/options are
            invalid, or the suffix is wrong.
    """
    from geosave.geodata.stac.catalog import FOOTER_KEY, ITEM_COLUMNS

    storage_options = write_options.pop("storage_options", None)
    if FOOTER_KEY in gdf.attrs:
        from geosave.geodata.io.formats import stac_geoparquet

        return stac_geoparquet.write_stac_geoparquet(
            gdf,
            path,
            collections=collections,
            overwrite=overwrite,
            storage_options=storage_options,
            **cast("dict[str, Any]", write_options),
        )
    if all(key in gdf for key in ITEM_COLUMNS):
        raise ValueError(
            "Item-shaped rows lack stac-geoparquet metadata; construct a catalog "
            "with stac.catalog.create_catalog before writing"
        )
    if collections is not None:
        raise ValueError("Collection definitions require a catalog with STAC metadata")
    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_options = cast(
        "dict[str, Any]", dict(write_options.pop("parquet_options", {}))
    )
    if not parquet_options.get("store_schema", write_options.get("store_schema", True)):
        raise ValueError(
            "GeoParquet requires store_schema=True to retain geometry metadata"
        )

    def save(target: Path) -> None:
        gdf.to_parquet(target, **write_options, **parquet_options)

    return storage.write_file(
        path,
        save,
        suffixes=FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )
