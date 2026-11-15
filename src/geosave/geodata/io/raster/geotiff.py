"""Open GDAL rasters and write GeoTIFFs and COGs as one variable per band.

rioxarray writes and reads pixels, grid, nodata, packing, descriptions and
tags. The GDAL codec stacks a Dataset into the band array rioxarray takes,
adds the properties rioxarray has no slot for, and reads all of it back.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import rioxarray
import xarray as xr
from fsspec.core import split_protocol
from rasterio.shutil import copy  # type: ignore[import-untyped]
from upath import UPath

from geosave.geodata.io import storage
from geosave.geodata.io.raster.codecs import gdal
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from dask.delayed import Delayed

    from geosave.geodata import Dataset

SUFFIXES = (".tif", ".tiff")
# Single-raster files GDAL reads into the same band stack.
READ_SUFFIXES = (*SUFFIXES, ".jp2", ".png")
# Filesystems GDAL reads itself, without an fsspec opener.
GDAL_PROTOCOLS = ("", "s3", "gs", "gcs", "az", "abfs", "http", "https", "hf")


class GeoTIFFOpenOptions(TypedDict, total=False):
    """Optional rioxarray behavior when reading one raster.

    Omitted keys use rioxarray or rasterio defaults. These are passed to
    `rioxarray.open_rasterio`, with rasterio open options forwarded beneath it.

    Attributes:
        parse_coordinates: Generate spatial coordinates from the file's
            transform. None lets rioxarray decide for the stored grid.
        cache: Cache accessed pixels as NumPy arrays. None lets rioxarray
            choose based on chunking; this is separate from GDAL's block cache.
        lock: Lock used for threaded lazy reads. False opens a separate file
            handle per request; a lock object serializes access to a handle.
        decode_times: Decode CF-encoded numeric dates instead of leaving them
            as numbers. GeoTIFF acquisition tags are handled separately.
        decode_timedelta: Decode CF duration variables as timedeltas; False
            retains numbers, and None delegates the decoding policy.
        overview_level: Zero-based rasterio overview index to open instead of
            full-resolution pixels, when the file has that reduced-resolution
            image. Omit to read the base image; -1 also selects the base image.
    """

    parse_coordinates: bool | None
    cache: bool | None
    lock: Any
    decode_times: bool
    decode_timedelta: bool | None
    overview_level: int


class GeoTIFFWriteOptions(TypedDict, total=False):
    """Declared creation options passed to GDAL's TIFF drivers.

    Omitted keys use the selected driver's defaults. Driver and GDAL build
    determine which codecs and settings are available.

    Attributes:
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
    **options: Unpack[GTiffWriteOptions],
) -> UPath | Delayed:
    """Write one observation as named TIFF bands using native GDAL options.

    Args:
        ds: Raster with at most one instant along `time`.
        path: Output path or URL ending in `.tif` or `.tiff`.
        compute: False returns one native `dask.Delayed`; computing it writes
            the file and returns its path.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        tiff_tags: Write ACDD attrs as baseline TIFF tags, which any TIFF
            reader shows, instead of GDAL-only metadata: `title` as
            `TIFFTAG_DOCUMENTNAME`, `summary` as `TIFFTAG_IMAGEDESCRIPTION`,
            `license` as `TIFFTAG_COPYRIGHT`, `institution` as
            `TIFFTAG_ARTIST`, and `geosave <version>` as `TIFFTAG_SOFTWARE`.
            Reading maps the four tags back to their ACDD attrs either way.
        **options: GTiff creation options, passed to GDAL unchanged.

    Returns:
        The written file, or one delayed result.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, or the raster holds several instants.
        RasterValidationError: Variables state different pixel meaning.

    Examples:
        >>> write_geotiff(ds, "scene.tif", compress="DEFLATE")
        PosixUPath('scene.tif')
    """
    path = storage.create_upath(path, storage_options)
    storage.check_destination(path, SUFFIXES, overwrite=overwrite)
    stack = gdal.encode_dataset(ds, tiff_tags=tiff_tags)
    write = partial(_store_geotiff, stack, path, overwrite=overwrite, **options)
    return storage.defer(write, compute=compute)


def write_cog(
    ds: xr.Dataset,
    path: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    tiff_tags: bool = False,
    **options: Unpack[COGWriteOptions],
) -> UPath | Delayed:
    """Write one observation as a Cloud Optimized GeoTIFF.

    A tiled GeoTIFF is written first, then GDAL's COG driver converts it,
    building overviews and ordering the file for range reads.

    Args:
        ds: Raster with at most one instant along `time`.
        path: Output path or URL ending in `.tif` or `.tiff`.
        compute: False returns one native `dask.Delayed`; computing it writes
            the file and returns its path.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.
        tiff_tags: Write ACDD attrs as baseline TIFF tags; see `write_geotiff`.
        **options: COG creation options, passed to GDAL unchanged.

    Returns:
        The written file, or one delayed result.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, or the raster holds several instants.
        RasterValidationError: Variables state different pixel meaning.

    Examples:
        >>> write_cog(ds, "s3://bucket/scene.tif", overview_resampling="AVERAGE")
        S3Path('s3://bucket/scene.tif')
    """
    path = storage.create_upath(path, storage_options)
    storage.check_destination(path, SUFFIXES, overwrite=overwrite)
    stack = gdal.encode_dataset(ds, tiff_tags=tiff_tags)
    write = partial(_store_cog, stack, path, overwrite=overwrite, **options)
    return storage.defer(write, compute=compute)


def write_files(
    files: Mapping[UPath, xr.Dataset],
    *,
    driver: Literal["COG", "GTiff"] = "COG",
    compute: bool = True,
    overwrite: bool = False,
    tiff_tags: bool = False,
    **options: Any,
) -> list[UPath] | Delayed:
    """Check and encode an asset batch, then write each file with one driver.

    Nothing is written unless every file passes its checks. A deferred batch
    checks its destinations again before writing its first file.

    Args:
        files: Each destination and the raster written there.
        driver: `"COG"` writes as `write_cog`, `"GTiff"` as `write_geotiff`.
        compute: False returns one native `dask.Delayed` writing the batch.
        overwrite: Replace existing files when true.
        tiff_tags: Write ACDD attrs as baseline TIFF tags; see `write_geotiff`.
        **options: The driver's creation options.

    Returns:
        The written files in order, or one delayed result.

    Raises:
        FileExistsError: A destination exists and `overwrite` is false.
        ValueError: Two destinations collide, or a raster is not one GeoTIFF.
    """
    stacks: dict[UPath, xr.DataArray] = {}
    addresses = set()
    for path, ds in files.items():
        address = str(path.absolute())
        if address in addresses:
            raise ValueError(f"path selections collide at {path}")
        addresses.add(address)
        storage.check_destination(path, SUFFIXES, overwrite=overwrite)
        stacks[path] = gdal.encode_dataset(ds, tiff_tags=tiff_tags)
    store = STORES[driver]
    write = partial(_store_files, stacks, store, overwrite=overwrite, **options)
    return storage.defer(write, compute=compute)


def _store_files(
    stacks: Mapping[UPath, xr.DataArray],
    store: Callable[..., UPath],
    *,
    overwrite: bool,
    **options: Any,
) -> list[UPath]:
    """Write every encoded stack, refusing all if one destination now exists."""
    for path in stacks:
        storage.refuse_existing(path, overwrite=overwrite)
    written = []
    for path, stack in stacks.items():
        written.append(store(stack, path, overwrite=overwrite, **options))
    return written


def _store_geotiff(
    stack: xr.DataArray, path: UPath, *, overwrite: bool, **options: Any
) -> UPath:
    """Write an encoded stack as a GeoTIFF at its destination."""
    storage.refuse_existing(path, overwrite=overwrite)
    # rioxarray stores each Dask chunk by reopening the file by name, which
    # needs a file GDAL reaches without an fsspec opener.
    with storage.stage_file(path) as local:
        _write_gtiff(stack, local, **options)
    return path


def _store_cog(
    stack: xr.DataArray, path: UPath, *, overwrite: bool, **options: Any
) -> UPath:
    """Write an encoded stack as a tiled GeoTIFF, then convert it to a COG."""
    storage.refuse_existing(path, overwrite=overwrite)
    with TemporaryDirectory() as folder, storage.stage_file(path) as local:
        source = Path(folder) / "source.tif"
        _write_gtiff(stack, source, tiled=True)
        copy(source, local, driver="COG", **options)
    return path


def _write_gtiff(stack: xr.DataArray, local: Path, **options: Any) -> None:
    """Write pixels and tags through rioxarray, then the extras it lacks."""
    lock = Lock() if stack.chunks is not None else None
    stack.rio.to_raster(str(local), driver="GTiff", lock=lock, **options)
    gdal.write_extras(local, stack)


# The writer of each driver `write_files` offers.
STORES: dict[str, Callable[..., UPath]] = {"COG": _store_cog, "GTiff": _store_geotiff}


def open_geotiff(
    source: Location,
    *,
    chunks: Any = {},
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[GeoTIFFOpenOptions],
) -> Dataset:
    """Read a GeoTIFF, or any single raster GDAL reads, as one variable per band.

    A band naming itself in a `variable_name` tag becomes that variable,
    spending the tag; the rest keep rasterio's `band_1`, `band_2`. Its metadata
    stays on the variable, the description as `long_name` unless it is the name.

    Args:
        source: Local path or URI to a raster GDAL can open. An
            `hf://buckets/` URL is read through the bucket's S3 gateway.
        chunks: Chunk configuration for the opened variables. The default,
            `{}`, opens Dask arrays chunked by the file's own tiles, so
            arithmetic on them stays lazy; None opens rioxarray's lazy arrays,
            which read on the first computation and keep the pixels.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        storage_options: Options for the filesystem a URL names. GDAL cannot
            see them, so Rasterio uses the UPath filesystem as its opener;
            without them, GDAL reads remote URLs lazily itself with its
            configured environment (see `read_gdal_address`).
        **open_options: Supported rioxarray and rasterio open options.

    Returns:
        Dataset holding one variable per band, each `(y, x)`; a dated file
        carries its instant as a scalar `time` coordinate, not as attrs.
        ``TIFFTAG_DATETIME`` supplies whole seconds, then the filename stem
        supplies a fallback. ACDD coverage start and end restore the
        ``start_datetime`` and ``end_datetime`` coordinates. A stem naming a
        whole period places the raster at its first instant.

    Raises:
        RasterioIOError: The file cannot be opened or read.
        ValueError: The file holds subdatasets or names one variable on two bands.

    Examples:
        >>> open_geotiff("scene.tif").gs.variables
        ('B04', 'B08')
        >>> open_geotiff("20190507_red.tif").time.item()
        datetime.datetime(2019, 5, 7, 0, 0)
    """
    path = storage.create_upath(source, storage_options)
    options: dict[str, Any] = dict(open_options)
    address = read_gdal_address(path)
    if address is None:
        address = path.path
        options["opener"] = path.fs

    properties = gdal.read_properties(address, **options)
    if chunks == {} and properties.is_tiled:
        chunks = {"y": properties.block_height, "x": properties.block_width}
    stack = _open_stack(address, chunks=chunks, **options)

    with ExitStack() as handles:
        handles.callback(stack.close)
        dataset = gdal.decode_dataset(stack, properties, filename=path.name)
        if mask_and_scale:
            dataset = dataset.gs.mask_and_scale()
        dataset.encoding["source"] = str(path.absolute())
        dataset.set_close(handles.pop_all().close)
        return cast("Dataset", dataset)


def _open_stack(address: str, *, chunks: Any, **options: Any) -> xr.DataArray:
    """Open stored pixels as rioxarray's `(band, y, x)` array.

    Raises:
        ValueError: The file holds subdatasets rather than one band stack.
    """
    # GeoSave reads each band's packing from GDAL, so rioxarray's warning
    # that one stack attr cannot state several scales does not apply.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="(Scales|Offsets) differ")
        opened = rioxarray.open_rasterio(
            address, chunks=chunks, mask_and_scale=False, **options
        )
    if isinstance(opened, xr.DataArray):
        return opened
    for raster in opened if isinstance(opened, list) else [opened]:
        raster.close()
    raise ValueError(f"{address} holds subdatasets; open one by its own URI")


def read_gdal_address(path: UPath) -> str | None:
    """Return the address GDAL reads a UPath by itself, or None where it cannot.

    GDAL reads local files and s3, gs, az, http and Hugging Face bucket URLs
    itself, lazily, with the credentials `configure_gdal` sets. It cannot see
    a UPath's `storage_options`, nor reach filesystems such as `memory://`.

    Args:
        path: The raster file.

    Returns:
        The path or URL to hand GDAL, or None.

    Examples:
        >>> read_gdal_address(create_upath("s3://bucket/scene.tif"))
        's3://bucket/scene.tif'
        >>> read_gdal_address(create_upath("s3://bucket/scene.tif", {"anon": True})) is None
        True
    """
    if path.storage_options or path.protocol not in GDAL_PROTOCOLS:
        return None
    return hf_to_s3(path)


def hf_to_s3(location: Location) -> str:
    """Spell a Hugging Face bucket URL as the S3 URL GDAL opens.

    GDAL reads local paths and `s3://`, `gs://`, `az://` and `https://` URLs
    itself. A Hugging Face bucket is reached through its S3 gateway, which
    needs S3 keys and `configure_gdal` pointed at `s3.hf.co` beforehand.

    Args:
        location: Local path or URL of a raster file.

    Returns:
        `s3://<namespace>/<bucket>/<key>` for `hf://buckets/<namespace>/<bucket>/<key>`,
        else the location unchanged.

    Raises:
        ValueError: An `hf://` URL names something other than a bucket.

    Examples:
        >>> hf_to_s3("hf://buckets/me/samples/forest/a.tif")
        's3://me/samples/forest/a.tif'
    """
    protocol, path = split_protocol(str(location))
    if protocol != "hf":
        return str(location)
    if not path.startswith("buckets/"):
        raise ValueError(
            f"{location} is not a Hugging Face bucket; only hf://buckets/ URLs "
            f"are served by the S3 gateway GDAL reads through"
        )
    return f"s3://{path.removeprefix('buckets/')}"
