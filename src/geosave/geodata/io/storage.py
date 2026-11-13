"""Locations: every path or URL becomes one UPath carrying its filesystem's options.

Each format reaches storage through the UPath: pyarrow through `path.fs`,
Zarr through `str(path)` and `path.storage_options`, readers through
`path.open()`, GDAL through `io.rasterio`. Writers that need random access to
their file (HDF5, GDAL, SQLite) write a local file that `stage_file` puts in
place. No code asks whether a path is local; a local path is a UPath on the
local filesystem.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
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


def create_upath(
    path: Location, storage_options: StorageOptions | None = None
) -> UPath:
    """Create the UPath of a path or URL, carrying its filesystem's options.

    Args:
        path: Local path, fsspec URL, or UPath.
        storage_options: Options for the filesystem a URL names, merged into
            any a UPath already carries.

    Returns:
        UPath whose `fs` and `path` address the location. A local one, a
        `file://` URL included, is a `PosixUPath`, which is a `pathlib.Path`.

    Raises:
        ValueError: `storage_options` sets a key UPath reserves.

    Examples:
        >>> create_upath("s3://bucket/scene.tif", {"anon": True}).fs
        <s3fs.core.S3FileSystem ...>
        >>> create_upath("file:///data/scene.tif")
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
    created = UPath(path, **options)
    # A `file://` URL reads as a FilePath, which pathlib and GDAL do not take.
    if created.protocol in ("file", "local"):
        return UPath(created.path)
    return created


def check_destination(path: UPath, suffixes: Sequence[str], *, overwrite: bool) -> None:
    """Refuse a destination of another format, or one that exists unless replaced.

    Args:
        path: Destination of one file or store.
        suffixes: Lowercase suffixes the format's destinations end in.
        overwrite: Allow an existing destination.

    Raises:
        ValueError: The suffix is not one of `suffixes`.
        FileExistsError: The destination exists and `overwrite` is false.
    """
    if path.suffix.lower() not in suffixes:
        raise ValueError(
            f"destination {path.name!r} must end in one of {list(suffixes)}"
        )
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")


@contextmanager
def stage_file(path: UPath) -> Generator[Path]:
    """Yield a local file to write, put at `path` when the block ends without error.

    A writer that seeks back into its file, as HDF5, GDAL and SQLite do,
    cannot write an object store's forward-only stream, so it writes here.

    Args:
        path: Destination of the finished file.

    Yields:
        Local path in a temporary folder, named as `path`.

    Examples:
        >>> with stage_file(create_upath("s3://bucket/scene.nc")) as local:
        ...     ds.to_netcdf(local)
    """
    with TemporaryDirectory() as folder:
        local = Path(folder) / path.name
        yield local
        create_parent(path)
        path.fs.put_file(str(local), path.path)


def create_parent(path: UPath) -> None:
    """Create the folder a file is written in, where its filesystem has folders.

    Object stores such as S3 or Hugging Face buckets have none; a key is
    written whole, and some UPaths refuse `mkdir`.

    Args:
        path: The file about to be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    # pathlib's UnsupportedOperation, which object-store UPaths raise, is one.
    except NotImplementedError:
        pass
