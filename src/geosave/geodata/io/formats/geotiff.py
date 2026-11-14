"""Write a Dataset as a COG or a plain GeoTIFF, one band per variable.

Raster preparation preserves per-variable CF encoding. GDAL metadata translation
lives in io.gdal_metadata; rioxarray writes the pixels and grid.
"""

from __future__ import annotations

from geosave.geodata.io import rasterio as raster_reader
from geosave.geodata.io.rasterio import RasterioOpenOptions

from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr

# rasterio ships this as a compiled module, which no type checker can resolve.
from rasterio.shutil import (
    copy,  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]
)
from upath import UPath
from xarray.core.utils import equivalent
from xarray.conventions import encode_cf_variable

import geosave.geodata.attrs as attrs
from geosave.geodata.core.conventions import (
    TIME_COORDINATE,
)
from geosave.geodata.io import storage
from geosave.geodata.io.gdal_metadata import format_time_tags, write_metadata
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dask.delayed import Delayed


type CreationOptionValue = str | int | float | bool

FILE_SUFFIXES = (".tif", ".tiff")


class GeoTIFFWriteOptions(TypedDict, total=False):
    """Declared creation options passed to GDAL's TIFF drivers.

    Omitted keys use the selected driver's defaults. Driver and GDAL build
    determine which codecs and settings are available.

    Attributes:
        tiff_tags: Write ACDD attrs as baseline TIFF tags, which any TIFF
            reader shows, instead of GDAL-only metadata: `title` as
            `TIFFTAG_DOCUMENTNAME`, `summary` as `TIFFTAG_IMAGEDESCRIPTION`,
            `license` as `TIFFTAG_COPYRIGHT`, `institution` as
            `TIFFTAG_ARTIST`, and `geosave <version>` as `TIFFTAG_SOFTWARE`.
            Each attr is written once, as its tag. Reading maps the four tags
            back to their ACDD attrs either way. False by default; GeoSave's
            own option, not forwarded to GDAL.
        compress: Compression codec name, such as `"DEFLATE"` or `"LZW"`.
        predictor: Compression predictor: 1 disables prediction, 2 differences
            adjacent samples, and 3 predicts floating-point samples.
        zlevel: GTiff's DEFLATE compression effort. The COG driver calls its
            corresponding setting `LEVEL`; GeoSave does not translate this key.
        num_threads: GDAL worker-thread count or `"ALL_CPUS"`.
        bigtiff: `"YES"` forces BigTIFF, `"NO"` forces classic TIFF;
            `"IF_NEEDED"` and `"IF_SAFER"` delegate size heuristics to GDAL.
        interleave: `"PIXEL"` stores bands together per pixel;
            `"BAND"` stores band planes separately.
        sparse_ok: Omit all-zero or all-nodata blocks from the TIFF allocation.
        creation_options: Additional GDAL driver settings, unpacked alongside the explicit
            options. Duplicate or GeoSave-managed settings are refused.
    """

    compress: str
    predictor: int
    zlevel: int
    num_threads: int | Literal["ALL_CPUS"]
    bigtiff: Literal["YES", "NO", "IF_NEEDED", "IF_SAFER"]
    interleave: Literal["PIXEL", "BAND"]
    sparse_ok: bool
    creation_options: dict[str, CreationOptionValue]
    tiff_tags: bool


class COGWriteOptions(GeoTIFFWriteOptions, total=False):
    """Creation options passed to GDAL's COG driver.

    Omitted keys use GDAL defaults. This includes the inherited TIFF options;
    the writer does not change pixel resolution when creating overviews.

    Attributes:
        compress: Compression codec name, such as `"DEFLATE"` or `"LZW"`.
        predictor: Compression predictor: 1 disables prediction, 2 differences
            adjacent samples, and 3 predicts floating-point samples.
        zlevel: Inherited GTiff effort option, passed unchanged. COG names its
            compression-effort setting `LEVEL` rather than `ZLEVEL`.
        num_threads: GDAL worker-thread count or `"ALL_CPUS"`.
        bigtiff: Force or prohibit BigTIFF with `"YES"` or `"NO"`, or ask
            GDAL's `"IF_NEEDED"`/`"IF_SAFER"` size heuristics.
        interleave: `"PIXEL"` stores bands together per pixel;
            `"BAND"` stores band planes separately.
        sparse_ok: Omit all-zero or all-nodata blocks from the TIFF allocation.
        creation_options: Additional GDAL driver settings; duplicate or managed keys
            are refused.
        blocksize: Square tile side in pixels; GDAL requires a multiple of 16.
        overview_resampling: GDAL kernel used to build reduced-resolution
            overviews, such as `"NEAREST"`, `"AVERAGE"`, or `"BILINEAR"`.
        overview_count: Number of overview levels to include; 0 omits them.
    """

    blocksize: int
    overview_resampling: str
    overview_count: int


class GTiffWriteOptions(GeoTIFFWriteOptions, total=False):
    """Creation options passed to GDAL's GTiff driver.

    Omitted keys use GDAL defaults. These include the inherited TIFF options
    and GTiff-specific strip, tile, and sample-layout settings.

    Attributes:
        compress: Compression codec name, such as `"DEFLATE"` or `"LZW"`.
        predictor: Compression predictor: 1 disables prediction, 2 differences
            adjacent samples, and 3 predicts floating-point samples.
        zlevel: DEFLATE compression effort, validated against GDAL's supported
            level range for its compression library.
        num_threads: GDAL worker-thread count or `"ALL_CPUS"`.
        bigtiff: Force or prohibit BigTIFF with `"YES"` or `"NO"`, or ask
            GDAL's `"IF_NEEDED"`/`"IF_SAFER"` size heuristics.
        interleave: `"PIXEL"` stores bands together per pixel;
            `"BAND"` stores band planes separately.
        sparse_ok: Omit all-zero or all-nodata blocks from the TIFF allocation.
        creation_options: Additional GDAL driver settings; duplicate or managed keys
            are refused.
        tiled: Write tiled blocks when True; False uses strips.
        blockxsize: Tile width in pixels when tiled; must be a multiple of 16.
        blockysize: Block height in pixels; tiled blocks need a multiple of 16.
        copy_src_overviews: Ask GDAL to copy existing source overviews. GeoSave
            creates this file from xarray pixels and supplies no source overviews.
        photometric: GDAL color interpretation, such as `"RGB"` or `"MINISBLACK"`.
        nbits: Stored bits per sample, subject to the pixel dtype and GDAL's
            supported bit depths; values must fit that storage representation.
    """

    tiled: bool
    blockxsize: int
    blockysize: int
    copy_src_overviews: bool
    photometric: str
    nbits: int


def validate(ds: xr.Dataset) -> None:
    """Check that one raster can be written as a GeoTIFF.

    Args:
        ds: Raster whose pixel dimensions, time and fill values are checked.

    Raises:
        ValueError: The grid, dimensions, dates or band fill values cannot be
            represented by one GeoTIFF. No pixels are computed.
    """
    ds.gs.validate()
    spatial = ds.gs.spatial_dims
    dimensions = set()
    for variable in ds.data_vars.values():
        dimensions.update(variable.dims)
    extra = dimensions - {*spatial, TIME_COORDINATE}
    if extra:
        raise ValueError(
            f"a COG holds one instant of one grid, but the raster also spans {sorted(extra)}"
        )
    count = ds.sizes.get(TIME_COORDINATE, 1)
    if count != 1:
        raise ValueError(
            f"one GeoTIFF holds one instant, but the raster spans {count} dates; "
            "select one observation before writing"
        )
    cube = ds.squeeze(TIME_COORDINATE) if TIME_COORDINATE in ds.dims else ds
    time = cube.coords.get(TIME_COORDINATE)
    if time is not None and np.issubdtype(time.dtype, np.datetime64):
        if np.isnat(time.values).any():
            raise ValueError("a GeoTIFF observation time cannot be NaT")
    if time is not None and not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError(
            "the time coordinate must be a datetime to be stored in a GeoTIFF"
        )
    fill_values = {}
    for name, variable in ds.data_vars.items():
        value = variable.gs.nodata
        if attrs.CFPacking.is_stated_in(variable.encoding):
            value = variable.encoding.get("_FillValue", value)
        fill_values[name] = value
    if not fill_values:
        raise ValueError("a GeoTIFF needs at least one data variable")
    first = next(iter(fill_values.values()))
    if not all(equivalent(value, first) for value in fill_values.values()):
        raise ValueError(
            f"one GeoTIFF holds one fill value, but these bands declare {fill_values}; "
            "set a shared nodata value or split bands into separate assets"
        )


def write_files(
    files: Mapping[UPath, xr.Dataset],
    *,
    driver: Literal["COG", "GTiff"] = "COG",
    compute: bool = True,
    overwrite: bool = False,
    **options: Any,
) -> list[UPath] | Delayed:
    """Write several GeoTIFFs, each destination and raster checked before any is written.

    Args:
        files: Destinations, each ending in `.tif` or `.tiff`, mapped to the
            raster one GeoTIFF holds there.
        driver: `"COG"` for Cloud Optimized GeoTIFFs, `"GTiff"` for plain ones.
        compute: False returns one native `dask.Delayed` writing every file.
        overwrite: Replace existing destinations.
        **options: Creation options for the driver.

    Returns:
        Written paths in destination order, or one delayed result.

    Raises:
        TypeError: An option is not one the driver takes.
        ValueError: Two destinations resolve to one file, a suffix is wrong,
            or a raster cannot be one GeoTIFF: more than one instant, band
            fill values that differ, no grid.
        FileExistsError: A destination exists and overwrite is false.

    Examples:
        >>> write_files({root / "red.tif": scene[["red"]], root / "nir.tif": scene[["nir"]]})
        [PosixUPath('/abs/red.tif'), PosixUPath('/abs/nir.tif')]
    """
    options = _normalize_options(driver, options)
    prepared = {}

    addresses = set()
    for path, raster in files.items():
        address = str(path.absolute())
        if address in addresses:
            raise ValueError(f"path selections collide at {path}")
        addresses.add(address)
        storage.check_destination(path, FILE_SUFFIXES, overwrite=overwrite)
        prepared[path] = _prepare_raster(raster)

    if compute:
        return _write_files(prepared, driver, options, overwrite)
    from dask.delayed import delayed

    # A partial keeps the rasters' lazy arrays out of the graph's dependencies.
    return delayed(partial(_write_files, prepared, driver, options, overwrite))()


def _write_files(
    files: Mapping[UPath, xr.Dataset],
    driver: str,
    options: Mapping[str, Any],
    overwrite: bool,
) -> list[UPath]:
    """Write checked rasters, one GeoTIFF each."""
    for path in files:
        storage.check_destination(path, FILE_SUFFIXES, overwrite=overwrite)
    written = []
    for path, raster in files.items():
        written.append(_write_file(raster, path, driver, options, overwrite))
    return written


def write_cog(
    ds: xr.Dataset,
    path: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[COGWriteOptions],
) -> UPath | Delayed:
    """Write one Dataset as a Cloud Optimized GeoTIFF, a band per variable.

    Each variable names itself in its band's metadata. A scalar `time`
    coordinate becomes ``TIFFTAG_DATETIME``, in whole seconds; its period
    becomes ACDD ``time_coverage_start`` and ``time_coverage_end`` attrs.

    Args:
        ds: Dataset on one grid, optionally with one time step.
        path: Output path, fsspec URL, or UPath, ending in ``.tif`` or ``.tiff``.
        compute: False defers the complete write and returns a delayed path.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        **options: COG creation options, and `tiff_tags=True` to write ACDD
            attrs as baseline TIFF tags (see `GeoTIFFWriteOptions`).

    Returns:
        The written file, carrying its storage options, or a delayed path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube holds several time steps,
            or the variables declare more than one fill value.

    Examples:
        >>> write_cog(ds, "scene.tif", compress="DEFLATE")
        PosixUPath('scene.tif')
        >>> write_cog(ds.assign_attrs(title="Forest"), "scene.tif", tiff_tags=True)
        PosixUPath('scene.tif')
    """
    destination = storage.create_upath(path, storage_options)
    settings = _normalize_options("COG", options)
    storage.check_destination(destination, FILE_SUFFIXES, overwrite=overwrite)
    raster = _prepare_raster(ds)
    operation = partial(_write_file, raster, destination, "COG", settings, overwrite)
    if compute:
        return operation()
    from dask.delayed import delayed

    return delayed(operation)()


def write_geotiff(
    ds: xr.Dataset,
    path: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[GTiffWriteOptions],
) -> UPath | Delayed:
    """Write one Dataset as a plain GeoTIFF, a band per variable.

    Reach for `write_cog` unless a consumer needs a striped or otherwise
    non-COG file. Coordinates map to tags exactly as `write_cog` maps them.

    Args:
        ds: Dataset on one grid, optionally with one time step.
        path: Output path, fsspec URL, or UPath, ending in ``.tif`` or ``.tiff``.
        compute: False defers the complete write and returns a delayed path.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        **options: GTiff creation options, and `tiff_tags=True` to write ACDD
            attrs as baseline TIFF tags (see `GeoTIFFWriteOptions`).

    Returns:
        The written file, carrying its storage options, or a delayed path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube holds several time steps,
            or the variables declare more than one fill value.

    Examples:
        >>> write_geotiff(ds, "scene.tif", tiled=True, blockxsize=512)
        PosixUPath('scene.tif')
    """
    destination = storage.create_upath(path, storage_options)
    settings = _normalize_options("GTiff", options)
    storage.check_destination(destination, FILE_SUFFIXES, overwrite=overwrite)
    raster = _prepare_raster(ds)
    operation = partial(_write_file, raster, destination, "GTiff", settings, overwrite)
    if compute:
        return operation()
    from dask.delayed import delayed

    return delayed(operation)()


def _normalize_options(driver: str, options: Mapping[str, Any]) -> dict[str, Any]:
    """Unpack driver settings once, refusing duplicate and managed keywords."""
    if driver not in ("COG", "GTiff"):
        raise ValueError(f"unsupported TIFF driver: {driver!r}")
    declared = COGWriteOptions if driver == "COG" else GTiffWriteOptions
    unknown = options.keys() - declared.__annotations__.keys()
    if unknown:
        raise TypeError(f"unsupported raster write options: {sorted(unknown)}")

    options = dict(options)
    extra = {}
    for key, value in options.pop("creation_options", {}).items():
        name = key.lower()
        if name in extra:
            raise TypeError(f"creation_options repeat setting: {name!r}")
        extra[name] = value
    managed = {
        "driver",
        "compute",
        "windowed",
        "lock",
        "tags",
        "tiff_tags",
        "dtype",
        "nodata",
        "crs",
        "transform",
        "count",
        "width",
        "height",
    }
    conflicts = extra.keys() & (options.keys() | managed)
    if conflicts:
        raise TypeError(
            f"creation_options repeat or override managed settings: {sorted(conflicts)}"
        )
    return {**options, **extra}


def _prepare_raster(ds: xr.Dataset) -> xr.Dataset:
    """Validate and encode one observation without computing its pixels."""
    validate(ds)
    rgb = ds.gs.rgb
    # Dataset.rio.to_raster stacks bands and loses per-variable CF encoding.
    variables = {}
    for name in ds.gs.variables:
        array = ds[name]
        dtype = np.dtype(array.encoding.get("dtype", array.dtype))
        # Encoding raw floats would invent a NaN fill over their missing-value code.
        if (
            attrs.CFPacking.is_stated_in(array.encoding)
            or "_Unsigned" in array.encoding
            or dtype != array.dtype
        ):
            variables[name] = encode_cf_variable(
                array.variable.copy(deep=False), name=name
            )
    ds = ds.assign(variables)

    # One file holds one instant, however the axis spells it.
    if TIME_COORDINATE in ds.dims:
        ds = ds.squeeze(TIME_COORDINATE)

    if rgb is not None:
        ds = ds.assign_attrs(attrs.Display(rgb=list(rgb)).to_attrs())
    return ds.assign_attrs(format_time_tags(ds))


def _write_file(
    raster: xr.Dataset,
    path: UPath,
    driver: str,
    options: Mapping[str, Any],
    overwrite: bool,
) -> UPath:
    """Write prepared pixels and metadata, then transfer the completed file."""
    storage.check_destination(path, FILE_SUFFIXES, overwrite=overwrite)
    options = dict(options)
    tiff_tags = bool(options.pop("tiff_tags", False))

    # Keep metadata on raster; rioxarray receives only the pixel representation.
    dimensions = set().union(*(variable.dims for variable in raster.data_vars.values()))
    auxiliary = [
        name
        for name, coordinate in raster.coords.items()
        if name == TIME_COORDINATE or not set(coordinate.dims) <= dimensions
    ]
    pixels = raster.drop_vars(auxiliary).copy(deep=False)
    pixels.attrs = {}
    for name in pixels.data_vars:
        pixels[name].attrs = {}

    with storage.stage_file(path) as built:
        # Finish metadata before COG conversion, preserving its optimized layout.
        source = (
            built.with_name(f"{built.stem}.source.tif") if driver == "COG" else built
        )
        source_options = {"tiled": True} if driver == "COG" else options
        pixels.rio.to_raster(
            source, driver="GTiff", windowed=bool(pixels.chunks), **source_options
        )
        with rasterio.open(source, "r+") as dst:
            write_metadata(dst, raster, tiff_tags=tiff_tags)
        if driver == "COG":
            copy(source, built, driver="COG", **options)
    return path


def open_geotiff(
    source: Location,
    *,
    chunks: Any = {},
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[RasterioOpenOptions],
) -> xr.Dataset:
    """Open GeoTIFF bands with their grid and metadata, retaining lazy pixels.

    Args:
        source: GeoTIFF path or URL.
        chunks: Dask chunks; {} uses native tiles, None uses backend indexing.
        mask_and_scale: Decode packing and missing values when True.
        storage_options: Filesystem credentials and connection settings.
        **options: Rasterio read options.

    Returns:
        Dataset with one variable per band.
    """
    return raster_reader.open_raster(
        source,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        storage_options=storage_options,
        **options,
    )
