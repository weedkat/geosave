"""Read native rasters, stacks, and vectors through their format modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from os import PathLike
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from upath import UPath


from . import storage
from .storage import Location
from .raster import cogs, gdal, netcdf, safe, zarr
from .raster.cogs import CogFile
from .vector import geojson, geopackage, geoparquet

if TYPE_CHECKING:
    from geosave.geodata import Dataset, DataTree, GeoDataFrame

_NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")

# Stores holding named variables.
_DATASET_SUFFIXES = (".zarr", *_NETCDF_SUFFIXES)

# Files holding one banded array, which GDAL reads.
_ARRAY_SUFFIXES = (".tif", ".tiff", ".jp2", ".png")

_VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".parquet", ".geoparquet")

# Vendor product directories, whose leaves name themselves by path.
_PRODUCT_SUFFIXES = (".safe",)


def open_raster(source: Location | Sequence[CogFile], **options: Any) -> Dataset:
    """Open one raster file or store, or the records one `to_cog` wrote.

    `options` are forwarded untouched; each format module types its own.

    Args:
        source: Local path or URI ending in a recognised raster suffix, or
            the `CogFile` records a `to_cog` returned.
        **options: Forwarded to the reader the suffix selects.

    Returns:
        Dataset holding one variable per band or stored variable.

    Raises:
        TypeError: A sequence holds anything but `CogFile` records.
        ValueError: The suffix names a vector format or no supported format,
            or no record is given.

    Examples:
        >>> open_raster("scene.zarr").gs.variables
        ('red', 'nir')
        >>> open_raster(scene.gs.to_cog("out/s_{time:%Y%m%d}.tif")).gs.variables
        ('red', 'nir')
    """
    if not isinstance(source, (str, PathLike, UPath)):
        files = list(source)
        if not all(isinstance(file, CogFile) for file in files):
            raise TypeError(
                "open_raster reads one path, or the CogFile records to_cog "
                "returned; open other files one at a time"
            )
        return cogs.read(files, **options)

    # A signed URL carries its token in the query; the path names the format.
    text = str(source)
    path = urlsplit(text).path if "://" in text else text
    suffix = PurePath(path).suffix.lower()

    if suffix == ".zarr":
        return zarr.read(source, **options)

    if suffix in _NETCDF_SUFFIXES:
        return netcdf.read(source, **options)

    if suffix in _ARRAY_SUFFIXES:
        return gdal.read(source, **options)

    if suffix in _PRODUCT_SUFFIXES:
        return safe.read(source, **options)

    if suffix in _VECTOR_SUFFIXES:
        raise ValueError(
            f"{source} holds vector features rather than pixels; read it with "
            f"open_vector"
        )

    raise ValueError(
        f"{source} ends in {suffix!r}, which names no supported raster format; "
        f"read a store ending in {_DATASET_SUFFIXES} or a raster ending in "
        f"{_ARRAY_SUFFIXES}"
    )


def open_stack(
    source: Location | Mapping[str, Location] | Sequence[CogFile],
    **options: Any,
) -> DataTree:
    """Open the layers of a saved stack as one stack.

    Args:
        source: The `CogFile` records a stack's `to_cog` returned; layer names
            mapped to the file or store each is; a local folder holding one
            file or store per layer, each named by its stem; or a vendor
            product such as a `.SAFE` folder.
        **options: Forwarded to the reader of each layer.

    Returns:
        DataTree holding one group per layer: records' and a mapping's in
        their own order, a folder's in name order.

    Raises:
        ValueError: A record names no layer, the folder holds no layer or two
            entries naming one layer, or the source is one raster.

    Examples:
        >>> open_stack(sample.gs.to_cog("out/{collection}.tif")).gs.groups
        ('optical', 'label')
        >>> open_stack(sample.gs.to_zarr("prepared/s1")).gs.groups
        ('sentinel_2_l2a', 'label')
    """
    from geosave.geodata.core.stack import stack

    if not isinstance(source, (str, PathLike, UPath, Mapping)):
        layers: dict[str, list[CogFile]] = {}
        for file in source:
            if file.collection is None:
                raise ValueError(
                    "these files name no collection; open_raster reads a lone raster"
                )
            layers.setdefault(file.collection, []).append(file)
        return stack(
            {name: cogs.read(files, **options) for name, files in layers.items()}
        )

    if isinstance(source, Mapping):
        with ExitStack() as opened:
            rasters = {}
            for name, path in source.items():
                rasters[name] = open_raster(path, **options)
                opened.callback(rasters[name].close)
            tree = stack(rasters)
            tree.set_close(opened.pop_all().close)
            return tree

    suffix = PurePath(str(source)).suffix.lower()

    # A folder of one file or store per layer, as a stack's `to_zarr` writes.
    folder = storage.location(source, options.get("storage_options"))
    if suffix not in (".zarr", *_PRODUCT_SUFFIXES) and folder.is_dir():
        entries = [
            entry
            for entry in sorted(folder.iterdir())
            if not entry.name.startswith(".")
            and entry.suffix.lower() in (*_ARRAY_SUFFIXES, *_DATASET_SUFFIXES)
        ]
        if not entries:
            raise ValueError(f"{source} holds no layer file or store to read")
        # A layer is named by its stem, so `dem.tif` beside `dem.zarr` is ambiguous.
        layers_by_stem: dict[str, UPath] = {}
        repeats = set()
        for entry in entries:
            if entry.stem in layers_by_stem:
                repeats.add(entry.stem)
            layers_by_stem[entry.stem] = entry
        if repeats:
            raise ValueError(
                f"{source} holds several entries naming the layers {sorted(repeats)}"
            )
        return open_stack(layers_by_stem, **options)

    if suffix in _PRODUCT_SUFFIXES:
        return safe.read_stack(source, **options)
    raise ValueError(
        f"{source} ends in {suffix!r}, which holds one raster; open it with "
        f"open_raster, or open the folder holding one raster per group"
    )


def open_vector(source: Location, **options: Any) -> GeoDataFrame:
    """Open a vector file through the reader its suffix names.

    GeoParquet accepts local paths or fsspec URLs; GeoJSON and GeoPackage are
    local only. A GeoParquet file holding STAC Items, an item table, is read
    with its stored relative hrefs made openable.

    Args:
        source: Local path, or fsspec URL for GeoParquet, ending in a
            recognised vector suffix.
        **options: Forwarded to the reader the suffix selects.

    Returns:
        GeoDataFrame holding the file's features in one CRS, read through `gs`.

    Raises:
        ValueError: The suffix names a raster format or no supported format.

    Examples:
        >>> open_vector("plantations.geojson").gs.crs.to_epsg()
        4326
    """
    suffix = PurePath(str(source)).suffix.lower()

    if suffix in (".geojson", ".json"):
        return cast("GeoDataFrame", geojson.read(source, **options))

    if suffix == ".gpkg":
        return cast("GeoDataFrame", geopackage.read(source, **options))

    if suffix in (".parquet", ".geoparquet"):
        frame = cast("GeoDataFrame", geoparquet.read(source, **options))
        from geosave.geodata.core.catalog import is_item_table

        if not is_item_table(frame):
            return frame
        # Deferred: only an item table needs the STAC dependencies.
        from .vector import stac_geoparquet

        return stac_geoparquet.resolve(frame, source)

    if suffix in _DATASET_SUFFIXES or suffix in _ARRAY_SUFFIXES:
        raise ValueError(
            f"{source} holds pixels rather than vector features; open it with "
            f"open_raster"
        )

    raise ValueError(
        f"{source} ends in {suffix!r}, which names no supported vector format; "
        f"read a vector ending in {_VECTOR_SUFFIXES}"
    )
