"""Open a raster or vector file by the format its suffix names."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlsplit

from geosave.geodata.io.raster import geotiff, netcdf, zarr
from geosave.geodata.io.vector import geoparquet, ogr

from .storage import Location, StorageOptions

if TYPE_CHECKING:
    from geosave.geodata import Dataset, DataTree, GeoDataFrame

# The reader of each suffix, built from the suffixes each format module states.
RASTER_OPENERS: dict[str, Callable[..., Dataset]] = {}
RASTER_TREE_OPENERS: dict[str, Callable[..., DataTree]] = {}
for _suffix in zarr.SUFFIXES:
    RASTER_OPENERS[_suffix] = zarr.open_zarr
    RASTER_TREE_OPENERS[_suffix] = zarr.open_zarr_tree
for _suffix in netcdf.SUFFIXES:
    RASTER_OPENERS[_suffix] = netcdf.open_netcdf
    RASTER_TREE_OPENERS[_suffix] = netcdf.open_netcdf_tree
for _suffix in geotiff.READ_SUFFIXES:
    RASTER_OPENERS[_suffix] = geotiff.open_geotiff

VECTOR_OPENERS: dict[str, Callable[..., Any]] = {}
for _suffix in ogr.GEOJSON_SUFFIXES:
    VECTOR_OPENERS[_suffix] = ogr.open_geojson
for _suffix in ogr.GEOPACKAGE_SUFFIXES:
    VECTOR_OPENERS[_suffix] = ogr.open_geopackage
for _suffix in geoparquet.SUFFIXES:
    VECTOR_OPENERS[_suffix] = geoparquet.open_geoparquet


def open_raster(
    source: Location,
    *,
    group: str | None = None,
    chunks: int | Mapping[str, int] | Literal["auto"] | None = {},
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Any,
) -> Dataset:
    """Open one raster file or one stored group as a Dataset.

    Args:
        source: Path, URL, or UPath ending in .tif, .tiff, .jp2, .png, .zarr,
            .nc, .nc4, or .cdf. Signed URL queries are preserved.
        group: Zarr or NetCDF group path; None opens the root group.
            Child groups are not opened. Other formats do not accept a group.
        chunks: Dask chunk sizes: {} (default) uses backend-preferred chunks,
            "auto" uses Dask's target sizes, an integer or dimension mapping
            requests sizes, and None uses backend lazy indexing without Dask.
            Coordinate indexes may still be read eagerly.
        mask_and_scale: False (default) retains stored numbers; True decodes
            missing values, scale_factor and add_offset into physical values.
        storage_options: Filesystem configuration listed below. None uses the
            UPath's options and backend credential discovery.
        **options: Format-specific opening options listed below. Omitted keys
            use the installed backend's defaults unless explicitly stated.

    Returns:
        Dataset for the file or selected group. Use `open_raster_tree` to open
        a Zarr or NetCDF hierarchy. File handles close with the result;
        `close()` releases them early, such as before overwriting the file.
        Opening does not reproject.

    Raises:
        ValueError: Unsupported format or incompatible backend option.
        TypeError: A sequence is supplied instead of one location.

    GeoTIFF / COG, JPEG2000 and PNG options:
        parse_coordinates (bool | None): Generate spatial coordinates from the file's
            transform. None lets rioxarray decide for the stored grid.
        cache (bool | None): Cache accessed pixels as NumPy arrays. None lets rioxarray
            choose based on chunking; this is separate from GDAL's block cache.
        lock (Any): Lock used for threaded lazy reads. False opens a separate file
            handle per request; a lock object serializes access to a handle.
        decode_times (bool): Decode CF-encoded numeric dates instead of leaving them
            as numbers. GeoTIFF acquisition tags are handled separately.
        decode_timedelta (bool | None): Decode CF duration variables as timedeltas; False
            retains numbers, and None delegates the decoding policy.
        overview_level (int): Zero-based rasterio overview index to open instead of
            full-resolution pixels, when the file has that reduced-resolution
            image. Omit to read the base image; -1 also selects the base image.

    Zarr options:
        synchronizer (Any): Zarr synchronization object where the backend supports
            it. The current Zarr 3 library warns that this is unimplemented.
        drop_variables (str | Sequence[str]): One variable name or a sequence to exclude while opening.
        consolidated (bool | None): Require consolidated metadata when True; False reads
            ordinary metadata. None delegates metadata discovery to xarray.
        overwrite_encoded_chunks (bool): Discard stored chunk sizes from each variable's
            encoding when opening with a different requested chunk layout.
        chunk_store (Any): Separate store for chunk data where the backend supports
            it. The current Zarr 3 library warns that this is unimplemented.
        decode_timedelta (bool | None): Decode CF duration variables as timedeltas; False
            retains numbers, and None delegates the decoding policy.
        use_zarr_fill_value_as_mask (bool | None): Use the Zarr fill value as missing-value
            metadata for CF decoding with `mask_and_scale=True`. False ignores
            it; None delegates to the store-format policy.
        chunked_array_type (str | None): Registered xarray chunk manager, such as `"dask"`.
        from_array_kwargs (dict[str, Any] | None): Extra keywords for the chunk manager's `from_array`
            operation, such as options for `dask.array.from_array`.

    NetCDF options:
        cache (bool | None): Cache accessed pixels as NumPy arrays; None delegates the
            choice to xarray based on whether chunking is enabled.
        decode_cf (bool | None): Decode CF conventions. False can leave time and grid
            metadata encoded rather than usable as geospatial coordinates.
        decode_times (bool | CFDatetimeCoder | Mapping[str, bool | CFDatetimeCoder] | None): Decode CF dates, use a CFDatetimeCoder, or choose a
            policy per variable with a mapping. False retains numeric dates.
        decode_timedelta (bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None): Decode CF durations using a boolean, CFTimedeltaCoder,
            or per-variable mapping; None delegates the decoding policy.
        use_cftime (bool | Mapping[str, bool] | None): Use cftime dates when True or NumPy dates when False,
            optionally per variable. None lets xarray select a representation.
            Xarray prefers configuring this on `decode_times`' coder.
        concat_characters (bool | Mapping[str, bool] | None): Combine character-array axes into strings, as one
            boolean or a per-variable mapping.
        drop_variables (str | Sequence[str]): One variable name or a sequence to exclude while opening.
        create_default_indexes (bool): Create pandas indexes for dimension coordinates,
            which reads those coordinate values into memory.
        inline_array (bool): Place backend arrays directly in Dask tasks instead of
            separate graph nodes; applies when Dask chunking is used.
        chunked_array_type (str | None): Registered xarray chunk manager, such as `"dask"`.
        from_array_kwargs (dict[str, Any] | None): Extra keywords for the chunk manager's `from_array`
            operation, such as options for `dask.array.from_array`.
        backend_kwargs (dict[str, Any] | None): Extra keywords for h5netcdf's open function.

    Storage options:
        Options are merged with a source UPath's options. Omitted values use the
        installed filesystem backend's defaults and credential discovery. Install
        the corresponding fsspec backend for cloud protocols. These are synchronous
        openers: leave asynchronous=False. protocol and chain_parser are forbidden.
        Every format reads through the UPath's filesystem: NetCDF, GeoJSON and
        GeoPackage from an open file handle, Zarr and GeoParquet by filesystem
        reads. A GDAL raster given storage_options is read through the filesystem
        as Rasterio's opener; without options GDAL reads the URL itself, with the
        environment configure_gdal sets. HF bucket URLs without options go through
        the S3 gateway that environment names.

        Common filesystem keys (where supported by the selected backend):
            use_listings_cache (bool): Cache directory listings.
            listings_expiry_time (float | None): Listing lifetime in seconds.
            max_paths (int | None): Maximum cached directory listings.
            skip_instance_cache (bool): Construct a fresh filesystem instance.
            asynchronous (bool): Must remain False for these synchronous readers.
            loop (AbstractEventLoop | None): Event loop for asynchronous backends.
            batch_size (int | None): Maximum concurrent bulk filesystem operations.

        S3 / S3-compatible (s3://):
            anon (bool): Use anonymous requests rather than discovered credentials.
            key, secret, token (str | None): Access key, secret, session token.
            username, password (str | None): Aliases for key and secret.
            endpoint_url (str | None): Custom S3 service endpoint.
            use_ssl (bool): Use HTTPS unless overridden by the endpoint URL.
            client_kwargs (dict | None): Arguments to the botocore client, including
                region_name (str), endpoint_url (str), verify (bool | str CA path),
                aws_access_key_id, aws_secret_access_key, aws_session_token (str).
            config_kwargs (dict | None): Botocore Config arguments, including
                connect_timeout, read_timeout (float seconds), max_pool_connections
                (int), signature_version (str), retries (dict: mode (str),
                max_attempts or total_max_attempts (int)), s3 (dict:
                addressing_style ("auto", "virtual", "path"), payload_signing_enabled
                (bool), use_accelerate_endpoint (bool)).
            s3_additional_kwargs (dict | None): S3 operation arguments such as
                RequestPayer ("requester") and ExpectedBucketOwner (str).
            session (AioSession | None): Existing aiobotocore session.
            requester_pays (bool): Acknowledge requester-pays access.
            default_block_size (int | None): Read-ahead block size in bytes.
            default_fill_cache (bool): Fill intervening cache blocks on seeking.
            default_cache_type (str): fsspec cache, for example "readahead" or "none".
            max_concurrency (int): Concurrent multipart transfers.
            fixed_upload_size (bool): Use fixed multipart upload sizes; write-only.
            version_aware (bool): Enable object-version access.
            cache_regions (bool): Cache bucket regions.

        Google Cloud Storage (gs://, gcs://):
            project (str): Billing project.
            access (str): "read_only", "read_write", or "full_control" OAuth scope.
            token (str | dict | None): Credential path/dictionary, or "anon",
                "google_default", "cache", "browser", "cloud"; None discovers it.
            block_size (int | None): Read block size in bytes.
            consistency (str): "none", "size", or "md5" transfer checking.
            cache_timeout (float | None): Listing cache lifetime in seconds.
            requests_timeout, timeout (float | None): Backend request timeouts.
            requester_pays (bool | str): Enable requester pays, optionally a project.
            session_kwargs (dict | None): aiohttp ClientSession options, including
                headers (dict[str, str]), auth (BasicAuth), timeout (ClientTimeout),
                trust_env (bool), connector (BaseConnector).
            endpoint_url (str | None): Custom storage endpoint.
            check_connection (bool | None): Backend initial connectivity check.
            default_location (str | None): New bucket location; irrelevant to reads.
            version_aware (bool): Enable object generation access.

        Azure (az://, abfs://):
            account_name, account_key (str | None): Storage account credentials.
            connection_string (str | None): Complete Azure connection string.
            credential (str | AsyncTokenCredential | None): SAS or token credential.
            sas_token (str | None): Shared access signature.
            anon (bool | None): Anonymous access or backend credential discovery.
            client_id, client_secret, tenant_id (str | None): Service principal.
            account_host (str | None): Custom account host.
            location_mode (str): "primary" or "secondary" service endpoint.
            blocksize (int): Transfer block size in bytes.
            default_fill_cache (bool): Fill cache blocks between seeks.
            default_cache_type (str): fsspec read cache implementation.
            max_concurrency (int | None): Concurrent transfers.
            timeout (int | None): Server-side timeout in seconds.
            connection_timeout, read_timeout (int | None): Client timeouts in seconds.
            version_aware (bool): Enable blob version access.
            assume_container_exists (bool | None): Skip container existence checks.

        HTTP / HTTPS:
            block_size (int | None): Read block size; zero streams without seeking.
                Random-access raster/Parquet readers need a seekable source.
            simple_links (bool): Include bare URL-like links in directory listings.
            same_scheme (bool): Restrict listed links to the source URL scheme.
            cache_type (str): fsspec read cache, such as "bytes" or "readahead".
            cache_options (dict | None): Selected cache's constructor options;
                the bytes cache supports trim (bool). Other keys
                depend on cache_type and are backend extensions.
            client_kwargs (dict | None): aiohttp ClientSession options, including
                headers (dict[str, str]) for authorization, auth (BasicAuth),
                timeout (ClientTimeout), trust_env (bool), connector (BaseConnector).
            get_client (callable | None): Async client factory.
            encoded (bool): Preserve an already encoded URL, including signed URLs.

        Hugging Face (hf://):
            token (str | None): Hub access token; None uses configured credentials.
            endpoint (str | None): Hub endpoint.
            block_size (int | None): Read block size in bytes; zero streams.

        Local (plain paths, file://, local://):
            auto_mkdir (bool): Backend option for creating write parent directories;
                irrelevant to reads. Local paths use native filesystem handling.

        Memory (memory://):
            No protocol-specific keys; storage lives in the current process.

        Nested dictionaries remain backend extension points, not closed GeoSave
        schemas. The named nested keys above cover common connection settings;
        additional SDK keys depend on the installed backend version. A filesystem
        may reject options unsupported by its implementation.
    """
    if isinstance(source, (list, tuple)):
        raise TypeError(
            "open_raster reads one file or store; files laid out across dates or bands "
            "read back through the catalog published with them "
            "(stac.load_dataset(rows.gs.to_items(), ...)), or open each file on its own"
        )
    opener = find_opener(source, RASTER_OPENERS)
    if group is not None:
        if opener is geotiff.open_geotiff:
            raise ValueError("group is only supported for Zarr and NetCDF")
        options["group"] = group
    return opener(
        source,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        storage_options=storage_options,
        **options,
    )


def open_raster_tree(
    source: Location,
    *,
    chunks: int | Mapping[str, int] | Literal["auto"] | None = {},
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Any,
) -> DataTree:
    """Open a whole Zarr or NetCDF hierarchy as a DataTree.

    Args:
        source: Path, URL, or UPath ending in .zarr, .nc, .nc4, or .cdf.
            Signed URL queries are preserved.
        chunks: Chunk sizes, as in `open_raster`; {} uses backend-preferred
            Dask chunks and None uses backend indexing without Dask.
        mask_and_scale: Decode missing values, scale_factor and add_offset.
        storage_options: Filesystem configuration, as in `open_raster`.
        **options: Format-specific options, as in `open_zarr_tree` or
            `open_netcdf_tree`.

    Returns:
        DataTree preserving the groups, including a root-only tree for a
        store without children. File handles close with the result;
        `close()` releases them early, such as before overwriting the file.
        Opening does not reproject.

    Raises:
        ValueError: Unsupported format or incompatible backend option.
        TypeError: A sequence is supplied instead of one location.

    Examples:
        >>> tree = open_raster_tree("scene.zarr")
        >>> raster = tree["s2/r10m"].to_dataset()
    """
    if isinstance(source, (list, tuple)):
        raise TypeError("open_raster_tree reads one file or store")
    opener = find_opener(source, RASTER_TREE_OPENERS)
    return opener(
        source,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        storage_options=storage_options,
        **options,
    )


def open_vector(
    source: Location,
    *,
    columns: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    storage_options: StorageOptions | None = None,
    **options: Any,
) -> GeoDataFrame:
    """Open one vector file as a native GeoDataFrame.

    Args:
        source: Path, URL, or UPath ending in .geojson, .json, .gpkg, .parquet,
            or .geoparquet. Parquet must contain GeoParquet geometry metadata.
        columns: Attribute names to read; None (default) reads all. Geometry is
            retained even if omitted; [] reads only geometry.
        bbox: Optional (minx, miny, maxx, maxy) filter in the file's CRS.
            Selects intersecting features without clipping. GeoParquet requires
            spatial covering metadata; an unsupported filter raises instead of
            silently loading the entire table. Cannot combine with mask.
        storage_options: Filesystem configuration listed below. None uses the
            UPath's options and backend credential discovery.
        **options: Format-specific options listed below. Omitted keys use the
            installed backend's defaults unless explicitly stated.

    Returns:
        GeoDataFrame retaining the file's CRS. STAC GeoParquet footer metadata
        is restored in attrs. Catalog asset hrefs are absolute; one a file
        stores relative is resolved against that file.

    Raises:
        ValueError: Unsupported format, ambiguous GeoPackage layer, or invalid filter.
        TypeError: Unsupported or duplicated backend option.

    GeoJSON options:
        columns (list[str] | None): Property columns to read; an empty list keeps only geometry.
            None reads all properties.
        bbox (tuple[float, float, float, float] | None): Feature filter `(minx, miny, maxx, maxy)` in the file's CRS.
            Mutually exclusive with `mask`.
        mask (Any): Geometry, GeoSeries, GeoDataFrame, or GeoJSON geometry selecting
            intersecting features. GeoPandas reprojects objects carrying a CRS;
            bare geometries must use the file's CRS. Cannot combine with `bbox`.
        rows (int | slice | None): Read the first N features with an integer, or a feature-position
            slice. None reads all matching features.
        engine_options (dict[str, Any]): Extra pyogrio read keywords, such as `use_arrow`. Do
            not repeat direct options or set `engine` or `ignore_geometry` here.

    GeoPackage options:
        layer: Layer name; None requires exactly one layer.
        columns (list[str] | None): Property columns to read; an empty list keeps only geometry.
            None reads all properties.
        bbox (tuple[float, float, float, float] | None): Feature filter `(minx, miny, maxx, maxy)` in the file's CRS.
            Mutually exclusive with `mask`.
        mask (Any): Geometry, GeoSeries, GeoDataFrame, or GeoJSON geometry selecting
            intersecting features. GeoPandas reprojects objects carrying a CRS;
            bare geometries must use the file's CRS. Cannot combine with `bbox`.
        rows (int | slice | None): Read the first N features with an integer, or a feature-position
            slice. None reads all matching features.
        engine_options (dict[str, Any]): Extra pyogrio read keywords, such as `use_arrow`. Do
            not repeat direct options or set `engine` or `ignore_geometry` here.

    GeoParquet / STAC GeoParquet options:
        columns (Sequence[str] | None): Attribute columns to read; the primary geometry column is
            retained. None reads all columns; [] reads only geometry.
        bbox (tuple[float, float, float, float] | None): Feature filter `(minx, miny, maxx, maxy)` in the stored CRS;
            requires covering-bbox metadata written with `write_covering_bbox`.
        to_pandas_kwargs (Mapping[str, object] | None): Options for converting non-geometry Arrow columns to
            pandas, such as dtype-conversion settings.
        parquet_options (dict[str, object]): Extra `pyarrow.parquet.read_table` keywords. Do not
            repeat direct options or the filesystem GeoSave supplies.

    Storage options:
        Options are merged with a source UPath's options. Omitted values use the
        installed filesystem backend's defaults and credential discovery. Install
        the corresponding fsspec backend for cloud protocols. These are synchronous
        openers: leave asynchronous=False. protocol and chain_parser are forbidden.
        Every format reads through the UPath's filesystem: NetCDF, GeoJSON and
        GeoPackage from an open file handle, Zarr and GeoParquet by filesystem
        reads. A GDAL raster given storage_options is read through the filesystem
        as Rasterio's opener; without options GDAL reads the URL itself, with the
        environment configure_gdal sets. HF bucket URLs without options go through
        the S3 gateway that environment names.

        Common filesystem keys (where supported by the selected backend):
            use_listings_cache (bool): Cache directory listings.
            listings_expiry_time (float | None): Listing lifetime in seconds.
            max_paths (int | None): Maximum cached directory listings.
            skip_instance_cache (bool): Construct a fresh filesystem instance.
            asynchronous (bool): Must remain False for these synchronous readers.
            loop (AbstractEventLoop | None): Event loop for asynchronous backends.
            batch_size (int | None): Maximum concurrent bulk filesystem operations.

        S3 / S3-compatible (s3://):
            anon (bool): Use anonymous requests rather than discovered credentials.
            key, secret, token (str | None): Access key, secret, session token.
            username, password (str | None): Aliases for key and secret.
            endpoint_url (str | None): Custom S3 service endpoint.
            use_ssl (bool): Use HTTPS unless overridden by the endpoint URL.
            client_kwargs (dict | None): Arguments to the botocore client, including
                region_name (str), endpoint_url (str), verify (bool | str CA path),
                aws_access_key_id, aws_secret_access_key, aws_session_token (str).
            config_kwargs (dict | None): Botocore Config arguments, including
                connect_timeout, read_timeout (float seconds), max_pool_connections
                (int), signature_version (str), retries (dict: mode (str),
                max_attempts or total_max_attempts (int)), s3 (dict:
                addressing_style ("auto", "virtual", "path"), payload_signing_enabled
                (bool), use_accelerate_endpoint (bool)).
            s3_additional_kwargs (dict | None): S3 operation arguments such as
                RequestPayer ("requester") and ExpectedBucketOwner (str).
            session (AioSession | None): Existing aiobotocore session.
            requester_pays (bool): Acknowledge requester-pays access.
            default_block_size (int | None): Read-ahead block size in bytes.
            default_fill_cache (bool): Fill intervening cache blocks on seeking.
            default_cache_type (str): fsspec cache, for example "readahead" or "none".
            max_concurrency (int): Concurrent multipart transfers.
            fixed_upload_size (bool): Use fixed multipart upload sizes; write-only.
            version_aware (bool): Enable object-version access.
            cache_regions (bool): Cache bucket regions.

        Google Cloud Storage (gs://, gcs://):
            project (str): Billing project.
            access (str): "read_only", "read_write", or "full_control" OAuth scope.
            token (str | dict | None): Credential path/dictionary, or "anon",
                "google_default", "cache", "browser", "cloud"; None discovers it.
            block_size (int | None): Read block size in bytes.
            consistency (str): "none", "size", or "md5" transfer checking.
            cache_timeout (float | None): Listing cache lifetime in seconds.
            requests_timeout, timeout (float | None): Backend request timeouts.
            requester_pays (bool | str): Enable requester pays, optionally a project.
            session_kwargs (dict | None): aiohttp ClientSession options, including
                headers (dict[str, str]), auth (BasicAuth), timeout (ClientTimeout),
                trust_env (bool), connector (BaseConnector).
            endpoint_url (str | None): Custom storage endpoint.
            check_connection (bool | None): Backend initial connectivity check.
            default_location (str | None): New bucket location; irrelevant to reads.
            version_aware (bool): Enable object generation access.

        Azure (az://, abfs://):
            account_name, account_key (str | None): Storage account credentials.
            connection_string (str | None): Complete Azure connection string.
            credential (str | AsyncTokenCredential | None): SAS or token credential.
            sas_token (str | None): Shared access signature.
            anon (bool | None): Anonymous access or backend credential discovery.
            client_id, client_secret, tenant_id (str | None): Service principal.
            account_host (str | None): Custom account host.
            location_mode (str): "primary" or "secondary" service endpoint.
            blocksize (int): Transfer block size in bytes.
            default_fill_cache (bool): Fill cache blocks between seeks.
            default_cache_type (str): fsspec read cache implementation.
            max_concurrency (int | None): Concurrent transfers.
            timeout (int | None): Server-side timeout in seconds.
            connection_timeout, read_timeout (int | None): Client timeouts in seconds.
            version_aware (bool): Enable blob version access.
            assume_container_exists (bool | None): Skip container existence checks.

        HTTP / HTTPS:
            block_size (int | None): Read block size; zero streams without seeking.
                Random-access raster/Parquet readers need a seekable source.
            simple_links (bool): Include bare URL-like links in directory listings.
            same_scheme (bool): Restrict listed links to the source URL scheme.
            cache_type (str): fsspec read cache, such as "bytes" or "readahead".
            cache_options (dict | None): Selected cache's constructor options;
                the bytes cache supports trim (bool). Other keys
                depend on cache_type and are backend extensions.
            client_kwargs (dict | None): aiohttp ClientSession options, including
                headers (dict[str, str]) for authorization, auth (BasicAuth),
                timeout (ClientTimeout), trust_env (bool), connector (BaseConnector).
            get_client (callable | None): Async client factory.
            encoded (bool): Preserve an already encoded URL, including signed URLs.

        Hugging Face (hf://):
            token (str | None): Hub access token; None uses configured credentials.
            endpoint (str | None): Hub endpoint.
            block_size (int | None): Read block size in bytes; zero streams.

        Local (plain paths, file://, local://):
            auto_mkdir (bool): Backend option for creating write parent directories;
                irrelevant to reads. Local paths use native filesystem handling.

        Memory (memory://):
            No protocol-specific keys; storage lives in the current process.

        Nested dictionaries remain backend extension points, not closed GeoSave
        schemas. The named nested keys above cover common connection settings;
        additional SDK keys depend on the installed backend version. A filesystem
        may reject options unsupported by its implementation.
    """
    opener = find_opener(source, VECTOR_OPENERS)
    frame = opener(
        source, columns=columns, bbox=bbox, storage_options=storage_options, **options
    )
    return cast("GeoDataFrame", frame)


def find_opener[T](
    source: Location, openers: Mapping[str, Callable[..., T]]
) -> Callable[..., T]:
    """Pick the reader of the format a location's suffix names.

    Args:
        source: Path or URL; a URL's query and fragment are not its suffix.
        openers: Readers by lowercase suffix.

    Returns:
        The format's `open_*` function.

    Raises:
        ValueError: No reader takes the suffix.

    Examples:
        >>> find_opener("s3://bucket/scene.TIF?versionId=3", RASTER_OPENERS)
        <function open_geotiff at ...>
    """
    text = str(source)
    path = urlsplit(text).path if "://" in text else text
    suffix = PurePath(path).suffix.lower()
    if suffix not in openers:
        raise ValueError(
            f"{source} ends in {suffix!r}, which this opener does not read; "
            f"it reads {sorted(openers)}"
        )
    return openers[suffix]
