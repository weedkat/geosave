"""Open and write GeoJSON and GeoPackage through GeoPandas' OGR engine.

Both formats are one pyogrio read and one pyogrio write. A GeoPackage layer
keeps the frame's own CRS and carries its `attrs` as layer metadata; GeoJSON
is WGS84 by specification and has no slot for `attrs`.
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd
import orjson
import pyogrio
from upath import UPath

from geosave.geodata.io import storage
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

GEOJSON_SUFFIXES = (".geojson", ".json")
GEOPACKAGE_SUFFIXES = (".gpkg",)

# The layer metadata item holding the frame's attrs, as JSON.
ATTRS_ITEM = "geosave_attrs"


class OGROpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when opening GeoJSON or a GeoPackage layer.

    Omitted keys use GeoPandas and pyogrio defaults. Spatial filters select
    features; they do not clip their geometries.

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
        engine_options: Extra pyogrio read keywords, such as `use_arrow`. Do
            not repeat direct options or set `engine` or `ignore_geometry` here.
    """

    columns: list[str] | None
    bbox: tuple[float, float, float, float] | None
    mask: Any
    rows: int | slice | None
    engine_options: dict[str, Any]


class OGRWriteOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when writing GeoJSON or a GeoPackage layer.

    Omitted keys use GeoPandas and pyogrio defaults.

    Attributes:
        index: Write the pandas index as property columns when True; False
            omits it. None lets GeoPandas choose based on the index type.
        promote_to_multi: Convert single geometries to their multi
            equivalents. None lets pyogrio choose for the format.
        nan_as_null: Write floating-property NaNs as missing values; exact
            handling depends on the format's driver.
        metadata: Layer metadata with string keys and values. A GeoPackage
            stores it; GeoJSON has no slot for it.
        engine_options: Extra pyogrio write keywords or GDAL driver options.
            Do not set `engine` or repeat direct options here.
    """

    index: bool | None
    promote_to_multi: bool | None
    nan_as_null: bool
    metadata: dict[str, str] | None
    engine_options: dict[str, Any]


def open_geojson(
    source: Location,
    *,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[OGROpenOptions],
) -> GeoDataFrame:
    """Open the features of one GeoJSON file.

    Args:
        source: Local path, URL, or UPath, read through its filesystem.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported GeoPandas and pyogrio read options.

    Returns:
        GeoDataFrame on the CRS the file names.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: An engine option selects an engine or disables geometry.

    Examples:
        >>> open_geojson("plots.geojson", bbox=(110.0, -8.0, 111.0, -7.0))
    """
    options = merge_engine_options(**open_options)
    with storage.create_upath(source, storage_options).open("rb") as handle:
        frame = gpd.read_file(handle, engine="pyogrio", **options)
    return cast("GeoDataFrame", frame)


def write_geojson(
    gdf: gpd.GeoDataFrame,
    path: Location,
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[OGRWriteOptions],
) -> UPath:
    """Write one GeoDataFrame as a GeoJSON file.

    GeoJSON states coordinates in WGS84, so the frame is reprojected. Its
    `attrs` are not stored; a GeoPackage or GeoParquet keeps the CRS, and a
    GeoPackage the attrs.

    Args:
        gdf: Frame to write.
        path: Output path or fsspec URL ending in `.geojson` or `.json`.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported GeoPandas and pyogrio write options.

    Returns:
        The written file, carrying its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported or supplied twice.
        ValueError: The suffix is wrong, or `engine_options` names an engine.

    Examples:
        >>> write_geojson(plots, "plots.geojson")
        PosixUPath('plots.geojson')
    """
    options = merge_engine_options(**write_options)
    destination = storage.create_upath(path, storage_options)
    storage.check_destination(destination, GEOJSON_SUFFIXES, overwrite=overwrite)
    return _write_layer(gdf.to_crs("EPSG:4326"), destination, "GeoJSON", **options)


def open_geopackage(
    source: Location,
    *,
    layer: str | None = None,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[OGROpenOptions],
) -> GeoDataFrame:
    """Open one GeoPackage layer as a GeoDataFrame.

    Args:
        source: Local path, URL, or UPath, read through its filesystem.
        layer: Layer name. None selects the layer only when exactly one exists.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported GeoPandas and pyogrio read options.

    Returns:
        GeoDataFrame on the CRS the selected layer names, with the `attrs`
        `write_geopackage` stored.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: `layer` is ambiguous, or an engine option selects an
            engine or disables geometry.

    Examples:
        >>> open_geopackage("survey.gpkg", layer="plots").attrs
        {'title': 'Plots'}
    """
    options = merge_engine_options(**open_options)
    path = storage.create_upath(source, storage_options)
    with path.open("rb") as handle:
        if layer is None:
            layer = select_layer(handle, path.name)
        handle.seek(0)
        frame = gpd.read_file(handle, layer=layer, engine="pyogrio", **options)
        handle.seek(0)
        frame.attrs.update(read_layer_attrs(handle, layer))
    return cast("GeoDataFrame", frame)


def write_geopackage(
    gdf: gpd.GeoDataFrame,
    path: Location,
    *,
    layer: str | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[OGRWriteOptions],
) -> UPath:
    """Write one GeoDataFrame as a layer in a GeoPackage.

    A GeoPackage holds the frame in its own CRS, so nothing is reprojected.
    The frame's `attrs` are stored as layer metadata and read back by
    `open_geopackage`.

    Args:
        gdf: Frame to write.
        path: Output path or fsspec URL ending in `.gpkg`.
        layer: Layer name. None names the layer after the file stem.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported GeoPandas and pyogrio write options.

    Returns:
        The written file, carrying its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported or supplied twice, or an attr
            cannot be spelled as JSON.
        ValueError: The suffix is wrong, or `engine_options` names an engine.

    Examples:
        >>> write_geopackage(plots, "survey.gpkg", layer="plots")
        PosixUPath('survey.gpkg')
    """
    options = merge_engine_options(**write_options)
    destination = storage.create_upath(path, storage_options)
    storage.check_destination(destination, GEOPACKAGE_SUFFIXES, overwrite=overwrite)
    options["metadata"] = format_layer_metadata(gdf, options.get("metadata"))
    options["layer"] = layer if layer is not None else destination.stem
    return _write_layer(gdf, destination, "GPKG", **options)


def _write_layer(
    gdf: gpd.GeoDataFrame,
    destination: UPath,
    driver: Literal["GeoJSON", "GPKG"],
    **options: Any,
) -> UPath:
    """Write one layer through pyogrio; GDAL and SQLite need a local file."""
    with storage.stage_file(destination) as local:
        gdf.to_file(local, driver=driver, engine="pyogrio", **options)
    return destination


def select_layer(handle: IO[bytes], filename: str) -> str:
    """Name the one layer a GeoPackage holds.

    Args:
        handle: The open GeoPackage.
        filename: File name, for the error message.

    Returns:
        The layer's name.

    Raises:
        ValueError: The file holds no layer, or several.
    """
    names = [str(name) for name in gpd.list_layers(handle)["name"]]
    if len(names) != 1:
        raise ValueError(
            f"GeoPackage {filename!r} contains layers {names}; select one with layer="
        )
    return names[0]


def format_layer_metadata(
    gdf: gpd.GeoDataFrame, metadata: dict[str, str] | None
) -> dict[str, str] | None:
    """Add the frame's attrs, as one JSON item, to the layer metadata.

    Args:
        gdf: Frame about to be written.
        metadata: Layer metadata the caller supplies.

    Returns:
        The caller's metadata with the attrs under `geosave_attrs`, or the
        metadata unchanged for a frame without attrs.

    Raises:
        TypeError: An attr cannot be spelled as JSON.
    """
    if not gdf.attrs:
        return metadata
    text = orjson.dumps(gdf.attrs, option=orjson.OPT_SERIALIZE_NUMPY).decode()
    return {**(metadata or {}), ATTRS_ITEM: text}


def read_layer_attrs(handle: IO[bytes], layer: str) -> dict[str, Any]:
    """Read the frame attrs `format_layer_metadata` stored on a layer.

    Args:
        handle: The open GeoPackage.
        layer: The layer read.

    Returns:
        The stored attrs, empty for a layer that states none.
    """
    metadata = pyogrio.read_info(handle, layer=layer)["layer_metadata"] or {}
    if ATTRS_ITEM not in metadata:
        return {}
    return orjson.loads(metadata[ATTRS_ITEM])


def merge_engine_options(**options: Any) -> dict[str, Any]:
    """Join typed options with `engine_options`, refusing any given twice.

    Args:
        **options: Read or write options, possibly holding `engine_options`.

    Returns:
        Keywords for `gpd.read_file` or `GeoDataFrame.to_file`; a dependency
        API taking open-ended keywords.

    Raises:
        TypeError: A key is both a typed option and an engine option.
        ValueError: An engine option selects an engine or disables geometry.

    Examples:
        >>> merge_engine_options(rows=5, engine_options={"use_arrow": True})
        {'rows': 5, 'use_arrow': True}
    """
    engine_options = dict(options.pop("engine_options", {}))
    for managed in ("engine", "ignore_geometry"):
        if managed in engine_options:
            raise ValueError(f"engine_options must not contain {managed}")
    if repeated := sorted(options.keys() & engine_options.keys()):
        raise TypeError(
            f"{repeated} given both directly and in engine_options; pass each once"
        )
    return {**options, **engine_options}
