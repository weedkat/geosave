"""Locations: every path or URL becomes one UPath carrying its filesystem's options.

Each format reaches storage through the UPath: pyarrow through `path.fs`,
Zarr through `str(path)` and `path.storage_options`, readers through
`path.open()`, GDAL through Rasterio filesystem openers. Writers that need
random access unavailable on their destination write a local file through
`stage_file`, the one place that asks a filesystem whether it is local.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from dask.delayed import Delayed

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
    refuse_existing(path, overwrite=overwrite)


def refuse_existing(path: UPath, *, overwrite: bool) -> None:
    """Refuse a destination that exists, unless it may be replaced.

    A deferred write calls this again when computed, since the destination
    may have appeared after it was first checked.

    Args:
        path: Destination of one file or store.
        overwrite: Allow an existing destination.

    Raises:
        FileExistsError: The destination exists and `overwrite` is false.
    """
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")


@contextmanager
def stage_file(path: UPath) -> Generator[Path]:
    """Yield a local file to write, which ends up at `path`.

    A writer that seeks back into its file or reopens it by name, as HDF5,
    GDAL and SQLite do, cannot write an object store's forward-only stream.
    A local destination is written in place; any other is written in a
    temporary folder and put at `path` when the block ends without error.

    Args:
        path: Destination of the finished file.

    Yields:
        Local path named as `path`.

    Examples:
        >>> with stage_file(create_upath("s3://bucket/scene.nc")) as local:
        ...     ds.to_netcdf(local)
    """
    create_parent(path)
    if getattr(path.fs, "local_file", False):
        yield Path(path.path)
        return
    with TemporaryDirectory() as folder:
        local = Path(folder) / path.name
        yield local
        path.fs.put_file(str(local), path.path)


def defer[Written](write: Callable[[], Written], *, compute: bool) -> Written | Delayed:
    """Run a checked write now, or hand it back as one Dask task.

    The write is passed whole, so a raster's lazy arrays stay out of the
    graph's dependencies and are computed by the write itself.

    Args:
        write: The write, taking no arguments and returning what it wrote.
        compute: False returns one `dask.Delayed` running `write` when computed.

    Returns:
        What `write` returns, or the delayed call of it.

    Examples:
        >>> defer(partial(store, raster, path), compute=False).compute()
        UPath('/abs/scene.zarr')
    """
    if compute:
        return write()
    from dask.delayed import delayed

    return delayed(write)()


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
