"""Everything GDAL-specific: its environment, its paths, and reading one file.

`configure_gdal` sets the process environment remote reads use. `read` opens
one GDAL-supported raster file as a Dataset of its bands: a band is one
variable, named by the `variable_name` tag it carries. Tags and native
band properties become CF attrs here, leaving rioxarray the pixels and the
grid.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from datetime import datetime
from enum import Enum, auto
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import numpy as np
import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr
from fsspec.core import split_protocol

import geosave.geodata.attrs as attrs
from geosave.geodata.core.conventions import (
    END_DATETIME,
    START_DATETIME,
    TIME_COORDINATE,
)
from geosave.geodata.io.gdal_metadata import GDAL_DATETIME_FORMAT, read_metadata
from geosave.geodata.io.storage import (
    Location,
    StorageOptions,
    create_upath,
    stage_file,
    create_parent,
)
from geosave.geodata.utils.datetime import parse_stem_dates

# Protocols GDAL reaches by itself, with the credentials `configure_gdal` sets;
# a Hugging Face bucket through its S3 gateway.
GDAL_PROTOCOLS = ("", "s3", "gs", "gcs", "az", "abfs", "http", "https", "hf")

if TYPE_CHECKING:
    from rasterio.io import DatasetWriter
    from upath import UPath

    from geosave.geodata import Dataset


class RasterioOpenOptions(TypedDict, total=False):
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


@contextmanager
def open_writer(path: UPath, **profile: Any) -> Generator[DatasetWriter]:
    """Open a GTiff writer; stage object-store writes that cannot seek in w+b."""
    if path.protocol in {"s3", "s3a", "gs", "gcs", "az", "abfs", "abfss", "hf"}:
        with stage_file(path) as local:
            with rasterio.open(local, "w", driver="GTiff", **profile) as dst:
                yield dst
    else:
        create_parent(path)
        with rasterio.open(
            path.path, "w", opener=path.fs, driver="GTiff", **profile
        ) as dst:
            yield dst


def open_raster(
    source: Location,
    *,
    chunks: Any = {},
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[RasterioOpenOptions],
) -> Dataset:
    """Read any GDAL-readable raster as one variable per band.

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
        ACDD ``time_coverage_start`` names the instant; without it
        ``TIFFTAG_DATETIME`` supplies whole seconds, then the filename stem
        supplies a fallback. A differing ``time_coverage_end`` restores the
        ``start_datetime`` and ``end_datetime`` coordinates of a composite. A stem naming a
        whole period places the raster at its first instant.

    Raises:
        RasterioIOError: The file cannot be opened or read.
        ValueError: The file holds subdatasets or names one variable on two bands.

    Examples:
        >>> open_raster("scene.tif").gs.variables
        ('B04', 'B08')
        >>> open_raster("20190507_red.tif").time.item()
        datetime.datetime(2019, 5, 7, 0, 0)
    """
    path = create_upath(source, storage_options)
    with ExitStack() as handles:
        options: dict[str, Any] = dict(open_options)
        address = read_gdal_address(path)
        if address is None:
            address = path.path
            options["opener"] = path.fs
        with rasterio.open(address, **options) as src:
            # `{}` asks for the backend's preferred chunks, as Zarr's are its own;
            # a tiled file's are its tiles, so a window reads only what it overlaps.
            block_y, block_x = src.block_shapes[0]
            if chunks == {} and block_x < src.width:
                chunks = {"y": block_y, "x": block_x}
            root, bands = read_metadata(src)
        cube = rioxarray.open_rasterio(
            address,
            band_as_variable=False,
            chunks=chunks,
            mask_and_scale=mask_and_scale,
            **options,
        )

        if isinstance(cube, list):
            raise ValueError(
                f"{source} holds subdatasets; open one of them by its own URI instead"
            )
        if not isinstance(cube, xr.DataArray):
            raise ValueError(
                f"{source} did not open as an array of "
                f"band; open it with the reader for its own format"
            )

        opened = cube
        handles.callback(opened.close)
        try:
            cube = cube.to_dataset(dim="band")
            cube = cube.rename_vars(dict(zip(cube.data_vars, bands, strict=True)))
            cube.attrs = root
            for number, (name, band_attrs) in enumerate(bands.items(), start=1):
                cube[name].attrs = band_attrs
                # A reader of the file's own tiles finds the variable by its band
                cube[name].encoding["band"] = number
                if mask_and_scale:
                    packing = attrs.CFPacking.from_attrs(band_attrs)
                    if packing is not None:
                        cube[name].encoding.update(packing.to_attrs())
                    cube[name].encoding.pop("scales", None)
                    cube[name].encoding.pop("offsets", None)
            if mask_and_scale:
                # Decoded pixels are physical, so the file's packing lives in encoding alone.
                cube = cube.assign(
                    {
                        name: attrs.clear_attrs(band, attrs.CFPacking)
                        for name, band in cube.data_vars.items()
                    }
                )

            # The TIFF datetime tag states the time; ACDD attrs state its period.
            coverage = attrs.ACDD.from_attrs(root)
            instant = None
            if "TIFFTAG_DATETIME" in root:
                try:
                    instant = np.datetime64(
                        datetime.strptime(
                            root["TIFFTAG_DATETIME"], GDAL_DATETIME_FORMAT
                        )  # noqa: DTZ007
                    )
                except ValueError:
                    pass
            if instant is None:
                # A file dating itself only in its name still places itself in time.
                timespan = parse_stem_dates(PurePath(str(source)).stem)
                if timespan is not None:
                    instant = np.datetime64(timespan[0])
            # One file holds one instant, so its date is a scalar coordinate; the
            # attrs only carried it there, and a write states them again from it.
            if instant is not None:
                cube = cube.assign_coords(
                    {
                        TIME_COORDINATE: xr.DataArray(
                            instant, attrs={"standard_name": "time", "axis": "T"}
                        )
                    }
                )
            if (
                coverage is not None
                and coverage.time_coverage_start is not None
                and coverage.time_coverage_end is not None
            ):
                cube = cube.assign_coords(
                    {
                        START_DATETIME: np.datetime64(
                            coverage.time_coverage_start, "ns"
                        ),
                        END_DATETIME: np.datetime64(coverage.time_coverage_end, "ns"),
                    }
                )
            for key in ("TIFFTAG_DATETIME", "time_coverage_start", "time_coverage_end"):
                cube.attrs.pop(key, None)
        except BaseException:
            opened.close()
            raise

        # Xarray's derived Datasets do not retain the backend closer they still use.
        if cube is not opened:
            cube.set_close(handles.pop_all().close)

        cube.encoding["source"] = str(path.absolute())
        return cast("Dataset", cube)


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


class _Unset(Enum):
    """Sentinel type for a keyword argument that was not passed.

    Distinct from an explicit None, and single-member so that `is not _UNSET`
    narrows a `X | None | _Unset` union down to `X | None`.
    """

    TOKEN = auto()


_UNSET = _Unset.TOKEN


def _bool_env(value: bool | _Unset) -> str | _Unset:
    return value if value is _UNSET else ("TRUE" if value else "FALSE")


def configure_gdal(
    *,
    aws_no_sign_request: bool | _Unset = _UNSET,
    aws_access_key_id: str | _Unset = _UNSET,
    aws_secret_access_key: str | _Unset = _UNSET,
    aws_session_token: str | _Unset = _UNSET,
    aws_default_region: str | _Unset = _UNSET,
    aws_s3_endpoint: str | _Unset = _UNSET,
    aws_virtual_hosting: bool | _Unset = _UNSET,
    aws_https: bool | _Unset = _UNSET,
    gdal_disable_readdir_on_open: bool | Literal["EMPTY_DIR"] | _Unset = _UNSET,
    gdal_http_max_retry: int | _Unset = _UNSET,
    gdal_http_retry_delay: float | _Unset = _UNSET,
    gdal_http_merge_consecutive_ranges: bool | _Unset = _UNSET,
    gdal_num_threads: int | Literal["ALL_CPUS"] | _Unset = _UNSET,
    cpl_vsil_curl_allowed_extensions: list[str] | _Unset = _UNSET,
    vsi_cache: bool | _Unset = _UNSET,
    vsi_cache_size: int | _Unset = _UNSET,
    omp_num_threads: int | _Unset = _UNSET,
) -> None:
    """Set the environment used by remote GDAL-backed raster reads.

    This is the single configuration adapter for rasterio and odc-stac
    reads. Settings are process-wide, so call it once during application
    startup before creating lazy reads.

    Args:
        aws_no_sign_request: Skip request signing for public AWS buckets.
        aws_access_key_id: AWS access key.
        aws_secret_access_key: AWS secret key.
        aws_session_token: Temporary AWS session token.
        aws_default_region: Default AWS region.
        aws_s3_endpoint: Host of an S3-compatible service, such as
            ``s3.hf.co`` for Hugging Face buckets.
        aws_virtual_hosting: False addresses buckets as path segments, which
            S3-compatible gateways need.
        aws_https: False reaches an S3-compatible service over plain HTTP,
            as a local one usually is.
        gdal_disable_readdir_on_open: True skips listing a remote folder
            before opening a file in it. `"EMPTY_DIR"` also skips probing
            for sidecar files, which makes remote COG reads far faster but
            hides `.aux.xml` statistics and `.ovr` overviews, local ones too.
        gdal_http_max_retry: Maximum HTTP retry attempts.
        gdal_http_retry_delay: Delay between HTTP retries in seconds.
        gdal_http_merge_consecutive_ranges: Merge adjacent byte-range reads.
        gdal_num_threads: GDAL worker count or ``"ALL_CPUS"``.
        cpl_vsil_curl_allowed_extensions: Remote file extensions GDAL may
            inspect.
        vsi_cache: Enable GDAL's in-memory VSI cache.
        vsi_cache_size: VSI cache size in bytes.
        omp_num_threads: OpenMP worker count for linked codecs.

    Examples:
        >>> configure_gdal(
        ...     aws_no_sign_request=True,
        ...     gdal_disable_readdir_on_open="EMPTY_DIR",
        ...     gdal_http_max_retry=3,
        ... )
    """
    values: dict[str, str | _Unset] = {
        "AWS_NO_SIGN_REQUEST": _bool_env(aws_no_sign_request),
        "AWS_ACCESS_KEY_ID": aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        "AWS_SESSION_TOKEN": aws_session_token,
        "AWS_DEFAULT_REGION": aws_default_region,
        "AWS_S3_ENDPOINT": aws_s3_endpoint,
        "AWS_VIRTUAL_HOSTING": _bool_env(aws_virtual_hosting),
        # GDAL spells this one YES or NO.
        "AWS_HTTPS": aws_https
        if aws_https is _UNSET
        else ("YES" if aws_https else "NO"),
        "GDAL_DISABLE_READDIR_ON_OPEN": (
            gdal_disable_readdir_on_open
            if isinstance(gdal_disable_readdir_on_open, str)
            else _bool_env(gdal_disable_readdir_on_open)
        ),
        "GDAL_HTTP_MAX_RETRY": (
            gdal_http_max_retry
            if gdal_http_max_retry is _UNSET
            else str(gdal_http_max_retry)
        ),
        "GDAL_HTTP_RETRY_DELAY": (
            gdal_http_retry_delay
            if gdal_http_retry_delay is _UNSET
            else str(gdal_http_retry_delay)
        ),
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": _bool_env(
            gdal_http_merge_consecutive_ranges
        ),
        "GDAL_NUM_THREADS": (
            gdal_num_threads if gdal_num_threads is _UNSET else str(gdal_num_threads)
        ),
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": (
            cpl_vsil_curl_allowed_extensions
            if cpl_vsil_curl_allowed_extensions is _UNSET
            else ",".join(cpl_vsil_curl_allowed_extensions)
        ),
        "VSI_CACHE": _bool_env(vsi_cache),
        "VSI_CACHE_SIZE": vsi_cache_size
        if vsi_cache_size is _UNSET
        else str(vsi_cache_size),
        "OMP_NUM_THREADS": (
            omp_num_threads if omp_num_threads is _UNSET else str(omp_num_threads)
        ),
    }
    for key, value in values.items():
        if value is not _UNSET:
            os.environ[key] = value
