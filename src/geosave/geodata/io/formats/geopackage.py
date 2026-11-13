"""Open and write GeoPackage vector layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import geopandas as gpd
from upath import UPath

from geosave.geodata.io import storage
from geosave.geodata.io.formats.utils.geopandas import unpack_engine_options
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

_FILE_SUFFIXES = (".gpkg",)


class GeoPackageOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when opening a GeoPackage.

    Omitted keys use GeoPandas and engine defaults. The reader's explicit
    `layer` selects a layer; spatial filters select features without clipping.

    Attributes:
        columns: Property columns to read; an empty list keeps only geometry.
            None reads all properties.
        bbox: Feature filter `(minx, miny, maxx, maxy)` in the selected layer's
            CRS. Mutually exclusive with `mask`.
        mask: Geometry, GeoSeries, GeoDataFrame, or GeoJSON geometry selecting
            intersecting features. GeoPandas reprojects objects carrying a CRS;
            bare geometries must use the layer's CRS. Cannot combine with `bbox`.
        rows: Read the first N features with an integer, or a feature-position
            slice. None reads all matching features.
        engine_options: Extra read keywords for the selected engine. Do not
            repeat direct options or set `engine` or `ignore_geometry` here.
    """

    columns: list[str] | None
    bbox: tuple[float, float, float, float] | None
    mask: Any
    rows: int | slice | None
    engine_options: dict[str, Any]


def open_geopackage(
    source: Location,
    *,
    layer: str | None = None,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[GeoPackageOpenOptions],
) -> GeoDataFrame:
    """Open one GeoPackage layer as a GeoDataFrame.

    Args:
        source: Local path, URL, or UPath, read through its filesystem.
        layer: Layer name. None selects the layer only when exactly one exists.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported GeoPandas and file-driver read options.

    Returns:
        GeoDataFrame on the CRS the selected layer names.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: `layer` is ambiguous, or a driver option selects an engine
            or disables geometry.
    """
    path = storage.create_upath(source, storage_options)
    options = unpack_engine_options(open_options)
    with path.open("rb") as handle:
        selected_layer = layer
        if selected_layer is None:
            layer_names = [str(name) for name in gpd.list_layers(handle)["name"]]
            if len(layer_names) != 1:
                raise ValueError(
                    f"GeoPackage {path.name!r} contains layers {layer_names}; "
                    "select one with layer="
                )
            selected_layer = layer_names[0]
            handle.seek(0)

        frame = gpd.read_file(
            handle,
            layer=selected_layer,
            engine="pyogrio",
            ignore_geometry=False,
            **options,
        )
    return cast("GeoDataFrame", frame)


class GeoPackageWriteOptions(TypedDict, total=False):
    """Optional GeoPandas behavior when writing a GeoPackage.

    Omitted keys use GeoPandas and engine defaults. The writer retains the
    table's CRS and writes one layer into a new or replaced file.

    Attributes:
        index: Write the pandas index as property columns when True; False
            omits it. None lets GeoPandas choose based on the index type.
        promote_to_multi: Pyogrio option converting single geometries to their
            multi equivalents. None lets pyogrio choose for the format.
        nan_as_null: Pyogrio option writing floating-property NaNs as missing
            values. Exact handling depends on the GeoPackage driver.
        metadata: Layer metadata with string keys and values, supported by the
            selected GeoPandas engine's GeoPackage writer.
        engine_options: Extra write keywords for the selected engine or GDAL
            driver. Do not set `engine` or repeat direct options here.
    """

    index: bool | None
    promote_to_multi: bool | None
    nan_as_null: bool
    metadata: dict[str, str] | None
    engine_options: dict[str, Any]


def write_geopackage(
    gdf: gpd.GeoDataFrame,
    path: Location,
    *,
    layer: str | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[GeoPackageWriteOptions],
) -> UPath:
    """Write one GeoDataFrame as a layer in a GeoPackage.

    A GeoPackage holds the frame in its own CRS, so nothing is reprojected.

    Args:
        gdf: Frame to write.
        path: Output path or fsspec URL ending in `.gpkg`.
        layer: Layer name. None names the layer after the file stem.
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

    with storage.stage_file(destination) as local:
        gdf.to_file(
            local,
            layer=layer if layer is not None else destination.stem,
            driver="GPKG",
            engine="pyogrio",
            **write_options,
            **engine_options,
        )
    return destination
