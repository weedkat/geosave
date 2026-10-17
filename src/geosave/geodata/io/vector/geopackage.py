"""Open and write GeoPackage vector layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd

from pathlib import Path

from upath import UPath

from .. import storage
from ..storage import Location, StorageOptions
from .geojson import driver_read_options

if TYPE_CHECKING:
    from os import PathLike
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
        driver_options: Extra read keywords for the selected engine. Do not
            repeat direct options or set `engine` or `ignore_geometry` here.
    """

    columns: list[str] | None
    bbox: tuple[float, float, float, float] | None
    mask: Any
    rows: int | slice | None
    driver_options: dict[str, Any]


def open_vector(
    source: Location,
    *,
    layer: str | None = None,
    engine: Literal["fiona", "pyogrio"] = "pyogrio",
    **open_options: Unpack[GeoPackageOpenOptions],
) -> GeoDataFrame:
    """Open one GeoPackage layer as a GeoDataFrame.

    Args:
        source: Local path, `file://` URL, or a URL GDAL reads itself, as
            `s3://` or `https://`, with credentials from `configure_gdal`.
        layer: Layer name. None selects the layer only when exactly one exists.
        engine: Concrete GeoPandas file-reading engine.
        **open_options: Supported GeoPandas and file-driver read options.

    Returns:
        GeoDataFrame on the CRS the selected layer names.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: `layer` is ambiguous, or a driver option selects an engine
            or disables geometry.
    """
    path = storage.absolute_location(storage.location(source))
    selected_layer = layer
    if selected_layer is None:
        layer_names = [str(name) for name in gpd.list_layers(path)["name"]]
        if len(layer_names) != 1:
            raise ValueError(
                f"GeoPackage {Path(str(source)).name!r} contains layers "
                f"{layer_names}; select one with layer="
            )
        selected_layer = layer_names[0]

    options = driver_read_options(open_options)
    frame = gpd.read_file(
        path,
        layer=selected_layer,
        engine=engine,
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
        driver_options: Extra write keywords for the selected engine or GDAL
            driver. Do not set `engine` or repeat direct options here.
        storage_options: Filesystem credentials and connection options for
            uploading the completed temporary file to a remote destination.
    """

    index: bool | None
    promote_to_multi: bool | None
    nan_as_null: bool
    metadata: dict[str, str] | None
    driver_options: dict[str, Any]
    storage_options: StorageOptions | None


def write(
    gdf: gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    layer: str | None = None,
    engine: Literal["fiona", "pyogrio"] = "pyogrio",
    overwrite: bool = False,
    **write_options: Unpack[GeoPackageWriteOptions],
) -> UPath:
    """Write one GeoDataFrame as a layer in a GeoPackage.

    A GeoPackage holds the frame in its own CRS, so nothing is reprojected.

    Args:
        gdf: Frame to write.
        path: Output path or fsspec URL ending in `.gpkg`.
        layer: Layer name. None names the layer after the file stem.
        engine: Concrete GeoPandas file-writing engine.
        overwrite: Replace an existing file when true.
        **write_options: Supported GeoPandas and file-driver write options.

    Returns:
        Local writes return a PosixUPath; URL writes return a UPath carrying
        its storage options.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported or supplied twice.
        ValueError: The suffix is wrong, or an option conflicts with `engine`.
    """
    storage_options = write_options.pop("storage_options", None)
    driver_options = dict(write_options.pop("driver_options", {}))
    if "engine" in driver_options:
        raise ValueError("driver_options must not contain engine; use engine=")
    # The layer is named after the file asked for, not the one being staged.
    name = layer if layer is not None else storage.location(path).stem

    def save(target: Path) -> None:
        gdf.to_file(
            target,
            layer=name,
            driver="GPKG",
            engine=engine,
            **write_options,
            **driver_options,
        )

    return storage.write(
        path,
        save,
        suffixes=_FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )
