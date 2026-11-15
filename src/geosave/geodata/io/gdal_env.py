"""Document and configure the process-wide GDAL environment."""

from __future__ import annotations

import os
from typing import Literal


def format_gdal_option(value: bool | int | float | str | list[str]) -> str:
    """Spell one setting as the text GDAL reads from its environment.

    Args:
        value: The setting as Python holds it.

    Returns:
        `TRUE` or `FALSE` for a flag, a comma-joined list, else the value's text.

    Examples:
        >>> format_gdal_option(False), format_gdal_option([".tif", ".jp2"])
        ('FALSE', '.tif,.jp2')
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, list):
        return ",".join(value)
    return str(value)


def configure_gdal(
    *,
    aws_no_sign_request: bool | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    aws_default_region: str | None = None,
    aws_s3_endpoint: str | None = None,
    aws_virtual_hosting: bool | None = None,
    aws_https: bool | None = None,
    gdal_disable_readdir_on_open: bool | Literal["EMPTY_DIR"] | None = None,
    gdal_http_max_retry: int | None = None,
    gdal_http_retry_delay: float | None = None,
    gdal_http_merge_consecutive_ranges: bool | None = None,
    gdal_num_threads: int | Literal["ALL_CPUS"] | None = None,
    cpl_vsil_curl_allowed_extensions: list[str] | None = None,
    vsi_cache: bool | None = None,
    vsi_cache_size: int | None = None,
    omp_num_threads: int | None = None,
) -> None:
    """Set the environment used by remote GDAL-backed raster reads.

    This is the single configuration adapter for rasterio and odc-stac
    reads. Settings are process-wide, so call it once during application
    startup before creating lazy reads. An argument left as None leaves its
    variable as the environment already has it.

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
    # Named as the arguments are, so each variable is its argument in capitals.
    settings = {
        "aws_no_sign_request": aws_no_sign_request,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_session_token": aws_session_token,
        "aws_default_region": aws_default_region,
        "aws_s3_endpoint": aws_s3_endpoint,
        "aws_virtual_hosting": aws_virtual_hosting,
        "aws_https": aws_https,
        "gdal_disable_readdir_on_open": gdal_disable_readdir_on_open,
        "gdal_http_max_retry": gdal_http_max_retry,
        "gdal_http_retry_delay": gdal_http_retry_delay,
        "gdal_http_merge_consecutive_ranges": gdal_http_merge_consecutive_ranges,
        "gdal_num_threads": gdal_num_threads,
        "cpl_vsil_curl_allowed_extensions": cpl_vsil_curl_allowed_extensions,
        "vsi_cache": vsi_cache,
        "vsi_cache_size": vsi_cache_size,
        "omp_num_threads": omp_num_threads,
    }
    for name, value in settings.items():
        if value is not None:
            os.environ[name.upper()] = format_gdal_option(value)
