"""Locations, filesystems, and the one way a single file is written."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from upath import UPath
from upath.types.storage_options import (
    AzureStorageOptions,
    FileStorageOptions,
    GCSStorageOptions,
    HfStorageOptions,
    HTTPStorageOptions,
    MemoryStorageOptions,
    S3StorageOptions,
)

# The filesystems a catalog or raster commonly lives on, typed as upath states them.
type StorageOptions = (
    S3StorageOptions
    | GCSStorageOptions
    | AzureStorageOptions
    | HTTPStorageOptions
    | HfStorageOptions
    | FileStorageOptions
    | MemoryStorageOptions
)
type Location = str | PathLike[str] | UPath

# Protocols UPath reports for a path on this machine: none for a plain path.
_LOCAL_PROTOCOLS = ("", "file", "local")


def location(path: Location, storage_options: StorageOptions | None = None) -> UPath:
    """Read a path or URL as a UPath carrying its filesystem's options.

    Args:
        path: Local path, fsspec URL, or UPath.
        storage_options: Options for the filesystem a URL names, merged into
            any a UPath already carries.

    Returns:
        UPath whose `fs` and `path` address the location. A local one, a
        `file://` URL included, is a `PosixUPath`, which is a `pathlib.Path`.

    Examples:
        >>> location("s3://bucket/scene.tif", {"anon": True}).fs
        <s3fs.core.S3FileSystem ...>
        >>> location("file:///data/scene.tif")
        PosixUPath('/data/scene.tif')
    """
    # upath's option types are open TypedDicts, so a checker cannot rule out a
    # key UPath reserves; one would silently choose another filesystem.
    options: dict[str, Any] = dict(storage_options or {})
    if reserved := sorted({"protocol", "chain_parser"} & set(options)):
        raise ValueError(
            f"storage_options cannot set {reserved}; the path's URL names its "
            f"filesystem"
        )
    found = UPath(path, **options)
    # A `file://` URL reads as a FilePath, which pathlib and GDAL do not take.
    if found.protocol in ("file", "local"):
        return UPath(found.path)
    return found


def is_local(path: Location) -> bool:
    """Return whether a location names a path on this machine."""
    return location(path).protocol in _LOCAL_PROTOCOLS


def local_path(path: Location) -> Path:
    """Read a local location as a path, without the `file://` it may carry."""
    return Path(location(path).path)


def absolute_location(path: Location) -> str:
    """Spell a location as the one string xarray, zarr, GDAL and STAC take.

    A remote UPath is not `os.PathLike`, so libraries beneath take its URL,
    with its `storage_options` passed beside it where they read through
    fsspec.

    Returns:
        The resolved absolute path of a local location, else its URL.

    Examples:
        >>> absolute_location("scene.tif")
        '/home/me/scene.tif'
        >>> absolute_location(UPath("s3://bucket/scene.tif", anon=True))
        's3://bucket/scene.tif'
    """
    if is_local(path):
        return str(local_path(path).resolve())
    return str(path)


def write(
    path: Location,
    save: Callable[[Path], None],
    *,
    suffixes: Sequence[str],
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> UPath:
    """Write one file through `save`, leaving its destination whole or untouched.

    Every single-file format writes this way: `save` only ever sees a local
    path, and the finished file is moved or uploaded to where it belongs.

    Args:
        path: Output path, fsspec URL, or UPath.
        save: Writes the file at the local path it is handed.
        suffixes: Lowercase suffixes the destination may end in.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        The written file: a `PosixUPath` for a local write, else the remote
        UPath carrying its `storage_options`.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong.

    Examples:
        >>> write("plots.parquet", plots.to_parquet, suffixes=(".parquet",))
        PosixUPath('plots.parquet')
    """
    destination = location(path, storage_options)
    if destination.suffix.lower() not in suffixes:
        raise ValueError(
            f"destination {destination.name!r} must end in one of {list(suffixes)}"
        )
    if not overwrite and destination.exists():
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    # A remote file is written in a temporary folder, then uploaded.
    if not is_local(destination):
        with TemporaryDirectory() as folder:
            staging = Path(folder) / destination.name
            save(staging)
            destination.fs.put(str(staging), destination.path)
        return destination

    # A local file is written beside its destination, then moved over it.
    target = local_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.stem}.staging{target.suffix}")
    try:
        save(staging)
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return destination
