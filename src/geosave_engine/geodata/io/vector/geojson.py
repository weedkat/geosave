"""Open and write GeoJSON vector files."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd


from pathlib import Path

from upath import UPath

from .. import storage
from ..storage import Location, StorageOptions

if TYPE_CHECKING:
    from os import PathLike

_FILE_SUFFIXES = (".geojson", ".json")


class GeoJSONOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when opening GeoJSON."""

    columns: list[str] | None
    bbox: tuple[float, float, float, float] | None
    mask: Any
    rows: int | slice | None
    driver_options: dict[str, Any]


def read(
    source: Location,
    *,
    engine: Literal["fiona", "pyogrio"] = "pyogrio",
    **open_options: Unpack[GeoJSONOpenOptions],
) -> gpd.GeoDataFrame:
    """Open the features of one GeoJSON file.

    Args:
        source: Local path, `file://` URL, or a URL GDAL reads itself, as
            `s3://` or `https://`, with credentials from `configure_gdal`.
        engine: Concrete GeoPandas file-reading engine.
        **open_options: Supported GeoPandas and file-driver read options.

    Returns:
        GeoDataFrame on the CRS the file names.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: A driver option selects an engine or disables geometry.
    """
    options = driver_read_options(open_options)
    frame = gpd.read_file(
        storage.openable(storage.location(source)),
        engine=engine,
        ignore_geometry=False,
        **options,
    )
    return cast("gpd.GeoDataFrame", frame)


def driver_read_options(open_options: Mapping[str, Any]) -> dict[str, Any]:
    """Join typed read options with `driver_options`, refusing any given twice.

    Args:
        open_options: Read options, possibly holding `driver_options`.

    Returns:
        One mapping of keywords for `gpd.read_file`.

    Raises:
        TypeError: A key is both a typed option and a driver option.
        ValueError: A driver option selects an engine or disables geometry.
    """
    typed = dict(open_options)
    driver_options = dict(typed.pop("driver_options", {}))
    if "engine" in driver_options:
        raise ValueError("driver_options must not contain engine; use engine=")
    if "ignore_geometry" in driver_options:
        raise ValueError("driver_options must not contain ignore_geometry")
    if repeated := sorted(typed.keys() & driver_options.keys()):
        raise TypeError(
            f"{repeated} given both directly and in driver_options; pass each once"
        )
    return {**typed, **driver_options}


class GeoJSONWriteOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when writing GeoJSON."""

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
    engine: Literal["fiona", "pyogrio"] = "pyogrio",
    overwrite: bool = False,
    **write_options: Unpack[GeoJSONWriteOptions],
) -> UPath:
    """Write one GeoDataFrame as a GeoJSON file.

    Geometry is serialized in WGS84.

    Args:
        gdf: Frame to write.
        path: Output path or fsspec URL ending in `.geojson` or `.json`.
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
    features = gdf.to_crs("EPSG:4326")

    def save(target: Path) -> None:
        features.to_file(
            target, driver="GeoJSON", engine=engine, **write_options, **driver_options
        )

    return storage.write(
        path,
        save,
        suffixes=_FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )
