"""Open and write GeoParquet vector files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from upath import UPath
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd
import typing_extensions

from .. import storage
from ..storage import Location, StorageOptions

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

FILE_SUFFIXES = (".parquet", ".geoparquet")


class GeoParquetOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when opening GeoParquet.

    Omitted keys use GeoPandas or PyArrow defaults. GeoSave supplies the
    UPath's filesystem explicitly instead of asking PyArrow to choose one.

    Attributes:
        columns: Columns to read. Include a geometry column; selecting none
            raises in GeoPandas. None reads all columns.
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


def open_vector(
    source: Location,
    **open_options: Unpack[GeoParquetOpenOptions],
) -> GeoDataFrame:
    """Open one local or fsspec-backed GeoParquet file.

    Args:
        source: Local path, fsspec URL, or UPath to GeoParquet.
        **open_options: Supported GeoPandas and Parquet read options.

    Returns:
        Native GeoDataFrame with its schema-selected `gs` accessor. Item
        tables restore absolute asset hrefs and Collection definitions;
        sample tables restore their Collection definitions.

    Raises:
        TypeError: An option is unsupported, or repeated in `parquet_options`.
    """
    storage_options = open_options.pop("storage_options", None)
    parquet_options = dict(open_options.pop("parquet_options", {}))
    parquet = storage.location(source, storage_options)
    frame = cast(
        "GeoDataFrame",
        gpd.read_parquet(
            parquet.path,
            filesystem=parquet.fs,
            **open_options,
            **parquet_options,
        ),
    )
    from geosave.geodata.core import GeoCatalog, GeoSamples

    if isinstance(frame.gs, GeoSamples):
        from . import sample_geoparquet

        frame.attrs["stac:collections"] = sample_geoparquet.read_collections(parquet)
    elif isinstance(frame.gs, GeoCatalog):
        from . import stac_geoparquet

        frame = stac_geoparquet.resolve_asset_hrefs(frame, parquet)
        frame.attrs["stac:collections"] = {
            key: value.to_dict(transform_hrefs=False)
            for key, value in stac_geoparquet.read_collections(parquet).items()
        }
    return frame


# Closed: a keyword these options do not name is a type error, not a pass-through.
class GeoParquetWriteOptions(typing_extensions.TypedDict, total=False, closed=True):
    """Optional GeoPandas behavior when writing GeoParquet.

    Omitted keys use GeoPandas or PyArrow defaults. These options apply to
    ordinary vectors; STAC Item tables use their own fixed writer encoding.

    Attributes:
        index: Serialize the pandas index when True; False omits it. None
            follows GeoPandas' policy, keeping a RangeIndex in metadata only.
        compression: Named Parquet compression codec, or None for no compression.
        geometry_encoding: `"WKB"` for binary geometry or `"geoarrow"` for
            native single-geometry-type encoding in GeoParquet 1.1.
        write_covering_bbox: Write each feature's bounding box and covering
            metadata, enabling filtered reads with `bbox=`.
        schema_version: GeoParquet specification version to write. None delegates
            to GeoPandas; encoding features must be supported by that version.
        storage_options: Credentials and connection options for the destination
            filesystem, including uploads after local file creation.
        parquet_options: Extra `pyarrow.parquet.write_table` keywords. Do not
            repeat options already supplied directly.
    """

    index: bool | None
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
    geometry_encoding: Literal["WKB", "geoarrow"]
    write_covering_bbox: bool
    schema_version: Literal["0.1.0", "0.4.0", "1.0.0-beta.1", "1.0.0", "1.1.0"] | None
    storage_options: StorageOptions | None
    parquet_options: dict[str, object]


def write(
    gdf: gpd.GeoDataFrame,
    path: Location,
    *,
    overwrite: bool = False,
    **write_options: Unpack[GeoParquetWriteOptions],
) -> UPath:
    """Write one GeoDataFrame to local or fsspec-backed GeoParquet.

    Args:
        gdf: Frame to write.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        overwrite: Replace an existing file when true.
        **write_options: Supported GeoPandas and Parquet write options.

    Returns:
        Local writes return a PosixUPath; URL writes return a UPath carrying
        its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported, or repeated in `parquet_options`.
        ValueError: The suffix is wrong.
    """
    storage_options = write_options.pop("storage_options", None)
    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_options = cast(
        "dict[str, Any]", dict(write_options.pop("parquet_options", {}))
    )

    def save(target: Path) -> None:
        gdf.to_parquet(target, **write_options, **parquet_options)

    return storage.write(
        path,
        save,
        suffixes=FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )
