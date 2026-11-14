"""Write named raster bands through Rasterio; convert disk TIFFs to COGs."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import numpy as np
import xarray as xr
from rasterio.windows import Window
from rasterio.shutil import copy  # type: ignore[import-untyped]
from upath import UPath
from xarray.conventions import encode_cf_variable
from xarray.core.utils import equivalent

from geosave.geodata import attrs
from geosave.geodata.io import gdal_metadata, storage
from geosave.geodata.io import rasterio as raster_reader
from geosave.geodata.io.rasterio import RasterioOpenOptions
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from dask.delayed import Delayed
    from rasterio.io import DatasetWriter

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
    """

    compress: str
    predictor: int
    zlevel: int
    num_threads: int | Literal["ALL_CPUS"]
    bigtiff: Literal["YES", "NO", "IF_NEEDED", "IF_SAFER"]
    interleave: Literal["PIXEL", "BAND"]
    sparse_ok: bool
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


def write_geotiff(
    ds: xr.Dataset,
    path: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    tiff_tags: bool = False,
    **options: Any,
) -> UPath | Delayed:
    """Write one observation as named TIFF bands using native GDAL options.

    ``compute=False`` defers the entire write and returns a delayed UPath.
    ``overwrite`` protects existing files. ``tiff_tags`` maps ACDD fields to
    baseline TIFF tags; all other options go directly to Rasterio.
    """
    path = storage.create_upath(path, storage_options)
    storage.check_destination(path, FILE_SUFFIXES, overwrite=overwrite)
    observation = single_observation(ds)
    bands = encode_bands(observation)
    profile = raster_profile(bands)
    tags = gdal_metadata.file_tags(observation, tiff_tags=tiff_tags)
    if not compute:
        from dask.delayed import delayed

        return delayed(
            partial(
                write_geotiff,
                ds,
                path,
                overwrite=overwrite,
                tiff_tags=tiff_tags,
                **options,
            )
        )()

    with raster_reader.open_writer(path, **profile, **options) as dst:
        dst.update_tags(**tags)
        write_pixels(dst, bands)
        gdal_metadata.write_band_tags(dst, bands)
        gdal_metadata.write_descriptions(dst, bands)
        gdal_metadata.write_packing(dst, bands)
        gdal_metadata.write_colorinterp(dst, observation)
        gdal_metadata.write_colormaps(dst, bands)
    return path


def write_cog(
    ds: xr.Dataset,
    path: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    tiff_tags: bool = False,
    **options: Any,
) -> UPath | Delayed:
    """Convert a disk-backed TIFF to COG without buffering the raster in RAM.

    GDAL owns COG conversion and creation options. The completed local file is
    transferred through UPath. ``compute=False`` defers the complete operation.
    """
    path = storage.create_upath(path, storage_options)
    storage.check_destination(path, FILE_SUFFIXES, overwrite=overwrite)
    if not compute:
        # Check representation before scheduling; pixels stay lazy.
        observation = single_observation(ds)
        raster_profile(encode_bands(observation))
        gdal_metadata.file_tags(observation, tiff_tags=tiff_tags)
        from dask.delayed import delayed

        return delayed(
            partial(
                write_cog, ds, path, overwrite=overwrite, tiff_tags=tiff_tags, **options
            )
        )()

    with storage.stage_file(path) as destination:
        source = destination.with_name(f"{destination.stem}.source.tif")
        write_geotiff(ds, source, tiff_tags=tiff_tags, tiled=True)
        copy(source, destination, driver="COG", **options)
    return path


def single_observation(ds: xr.Dataset) -> xr.Dataset:
    """Select a singleton time axis; require one grid with 2-D bands."""
    ds.gs.validate()
    if "time" in ds.dims:
        if ds.sizes["time"] != 1:
            raise ValueError("one GeoTIFF holds one instant; select one observation")
        ds = ds.squeeze("time")
    spatial = set(ds.gs.spatial_dims)
    for name, band in ds.data_vars.items():
        if set(band.dims) != spatial:
            raise ValueError(
                f"GeoTIFF band {name!r} must have spatial dimensions {spatial}"
            )
    return ds


def encode_band(band: xr.DataArray) -> xr.DataArray:
    """Apply a band's stated CF storage recipe, keeping raw pixels unchanged."""
    dtype = np.dtype(band.encoding.get("dtype", band.dtype))
    if not (
        attrs.CFPacking.is_stated_in(band.encoding)
        or "_Unsigned" in band.encoding
        or dtype != band.dtype
    ):
        return band
    variable = encode_cf_variable(band.variable.copy(deep=False), name=band.name)
    return xr.DataArray(variable, coords=band.coords, name=band.name)


def encode_bands(ds: xr.Dataset) -> xr.Dataset:
    """Encode variables independently, retaining their order and metadata."""
    return ds.assign({name: encode_band(band) for name, band in ds.data_vars.items()})


def raster_profile(bands: xr.Dataset) -> dict[str, Any]:
    """Describe the common TIFF grid, stored dtype, band count and nodata."""
    if not bands.data_vars:
        raise ValueError("a GeoTIFF needs at least one data variable")
    fills = {name: band.gs.nodata for name, band in bands.data_vars.items()}
    nodata = next(iter(fills.values()))
    if not all(equivalent(value, nodata) for value in fills.values()):
        raise ValueError(f"one GeoTIFF holds one fill value; bands declare {fills}")
    grid = bands.odc.geobox
    return dict(
        width=grid.width,
        height=grid.height,
        count=len(bands.data_vars),
        crs=str(grid.crs),
        transform=grid.transform,
        nodata=nodata,
        dtype=np.result_type(*(band.dtype for band in bands.data_vars.values())).name,
    )


def write_pixels(dst: DatasetWriter, bands: xr.Dataset) -> None:
    """Stream each band by source chunk; use TIFF windows for unchunked sources."""
    from dask.array.core import slices_from_chunks

    for index, band in enumerate(bands.data_vars.values(), start=1):
        band = band.transpose(*bands.gs.spatial_dims)
        if band.chunks is None:
            windows = (window for _, window in dst.block_windows(index))
        else:
            windows = (
                Window.from_slices(slices[0], slices[1])
                for slices in slices_from_chunks(band.chunks)
            )
        for window in windows:
            rows, columns = window.toslices()
            y, x = band.dims
            pixels = band.isel({y: rows, x: columns}).values
            dst.write(pixels, index, window=window)


def write_files(
    files: Mapping[UPath, xr.Dataset],
    *,
    driver: Literal["COG", "GTiff"] = "COG",
    compute: bool = True,
    overwrite: bool = False,
    **options: Any,
) -> list[UPath] | Delayed:
    """Preflight an asset batch, then call the one-file writer for each asset.

    Deferred batches repeat all checks before writing their first file.
    """
    writer = {"COG": write_cog, "GTiff": write_geotiff}[driver]
    addresses = set()
    for path, ds in files.items():
        address = str(path.absolute())
        if address in addresses:
            raise ValueError(f"path selections collide at {path}")
        addresses.add(address)
        storage.check_destination(path, FILE_SUFFIXES, overwrite=overwrite)
        observation = single_observation(ds)
        raster_profile(encode_bands(observation))
        gdal_metadata.file_tags(observation, tiff_tags=options.get("tiff_tags", False))
    if not compute:
        from dask.delayed import delayed

        return delayed(
            partial(write_files, files, driver=driver, overwrite=overwrite, **options)
        )()
    return [
        cast(UPath, writer(ds, path, overwrite=overwrite, **options))
        for path, ds in files.items()
    ]


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
