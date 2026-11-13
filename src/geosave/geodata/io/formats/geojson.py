"""Open and write GeoJSON vector files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import geopandas as gpd
from upath import UPath

from geosave.geodata.io import storage
from geosave.geodata.io.formats.utils.geopandas import unpack_engine_options
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

_FILE_SUFFIXES = (".geojson", ".json")


class GeoJSONOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when opening GeoJSON.

    Omitted keys use GeoPandas and the selected engine's defaults. Spatial
    filters select features; they do not clip their geometries.

    Attributes:
        columns: Property columns to read; an empty list keeps only geometry.
            None reads all properties.
        bbox: Feature filter `(minx, miny, maxx, maxy)` in the file's CRS.
            Mutually exclusive with `mask`.
        mask: Geometry, GeoSeries, GeoDataFrame, or GeoJSON geometry selecting
            intersecting features. GeoPandas reprojects objects carrying a CRS;
            bare geometries must use the file's CRS. Cannot combine with `bbox`.
        rows: Read the first N features with an integer, or a feature-position
            slice. None reads all matching features.
        engine_options: Extra read keywords for the selected GeoPandas engine,
            such as pyogrio's `use_arrow`. Do not repeat direct options or set
            `engine` or `ignore_geometry` here.
    """

    columns: list[str] | None
    bbox: tuple[float, float, float, float] | None
    mask: Any
    rows: int | slice | None
    engine_options: dict[str, Any]


def open_geojson(
    source: Location,
    *,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[GeoJSONOpenOptions],
) -> GeoDataFrame:
    """Open the features of one GeoJSON file.

    Args:
        source: Local path, URL, or UPath, read through its filesystem.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported GeoPandas and file-driver read options.

    Returns:
        GeoDataFrame on the CRS the file names.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: A driver option selects an engine or disables geometry.
    """
    options = unpack_engine_options(open_options)
    with storage.create_upath(source, storage_options).open("rb") as handle:
        frame = gpd.read_file(
            handle, engine="pyogrio", ignore_geometry=False, **options
        )
    return cast("GeoDataFrame", frame)


class GeoJSONWriteOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when writing GeoJSON.

    Omitted keys use GeoPandas and engine defaults. Geometry is transformed
    to WGS84 before writing. Engine-specific options need that engine's support.

    Attributes:
        index: Write the pandas index as property columns when True; False
            omits it. None lets GeoPandas choose based on the index type.
        promote_to_multi: Pyogrio option converting single geometries to their
            multi equivalents. None lets pyogrio choose for the format.
        nan_as_null: Pyogrio option writing floating-property NaNs as JSON null.
            False requests format-specific NaN handling instead.
        metadata: String metadata for engines/drivers supporting it. GeoJSON
            does not provide GeoPandas' GeoPackage metadata support.
        engine_options: Extra write keywords for the selected engine or GDAL
            driver. Do not set `engine` or repeat direct options here.
    """

    index: bool | None
    promote_to_multi: bool | None
    nan_as_null: bool
    metadata: dict[str, str] | None
    engine_options: dict[str, Any]


def write_geojson(
    gdf: gpd.GeoDataFrame,
    path: Location,
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[GeoJSONWriteOptions],
) -> UPath:
    """Write one GeoDataFrame as a GeoJSON file.

    Geometry is serialized in WGS84.

    Args:
        gdf: Frame to write.
        path: Output path or fsspec URL ending in `.geojson` or `.json`.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported GeoPandas and file-driver write options.

    Returns:
        The written file, carrying its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported or supplied twice.
        ValueError: The frame carries STAC Item metadata, the suffix is wrong,
            or `engine_options` names an engine.
    """

    engine_options = dict(write_options.pop("engine_options", {}))
    if "engine" in engine_options:
        raise ValueError("engine_options must not contain engine; pyogrio writes")
    destination = storage.create_upath(path, storage_options)
    storage.check_destination(destination, _FILE_SUFFIXES, overwrite=overwrite)
    features = gdf.to_crs("EPSG:4326")

    with storage.stage_file(destination) as local:
        features.to_file(
            local,
            driver="GeoJSON",
            engine="pyogrio",
            **write_options,
            **engine_options,
        )
    return destination
