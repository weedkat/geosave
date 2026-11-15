"""Open and write raster Datasets and trees as netCDF files.

xarray writes and reads the file through h5netcdf. The codecs spell time
cells as CF bounds and the grid as CF axes, and back.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import xarray as xr
from upath import UPath

from geosave.geodata.io import storage
from geosave.geodata.io.raster import codecs
from geosave.geodata.io.raster.codecs.cf import SPATIAL_AXES, TIME_BOUNDS
from geosave.geodata.io.raster.encoding import check_encoding, split_encoding
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from dask.delayed import Delayed
    from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder

    from geosave.geodata import Dataset, DataTree

SUFFIXES = (".nc", ".nc4", ".cdf")

# Cells are stored as CF bounds; coordinates carry no fill (CF §5, §7.1).
CODECS = (TIME_BOUNDS, SPATIAL_AXES)

type NetCDFChunkSpec = (
    int | tuple[int, ...] | Mapping[str, int] | Literal["auto"] | None
)


class NetCDFOpenOptions(TypedDict, total=False):
    """Optional `xarray.open_dataset` behavior when opening netCDF.

    Omitted keys use xarray or h5netcdf defaults. GeoSave passes
    `decode_coords="all"` to retain grid mappings and defaults `mask_and_scale`
    to False in the reader's explicit arguments.

    Attributes:
        cache: Cache accessed pixels as NumPy arrays; None delegates the
            choice to xarray based on whether chunking is enabled.
        decode_cf: Decode CF conventions. False can leave time and grid
            metadata encoded rather than usable as geospatial coordinates.
        decode_times: Decode CF dates, use a CFDatetimeCoder, or choose a
            policy per variable with a mapping. False retains numeric dates.
        decode_timedelta: Decode CF durations using a boolean, CFTimedeltaCoder,
            or per-variable mapping; None delegates the decoding policy.
        use_cftime: Use cftime dates when True or NumPy dates when False,
            optionally per variable. None lets xarray select a representation.
            Xarray prefers configuring this on `decode_times`' coder.
        concat_characters: Combine character-array axes into strings, as one
            boolean or a per-variable mapping.
        drop_variables: One variable name or a sequence to exclude while opening.
        create_default_indexes: Create pandas indexes for dimension coordinates,
            which reads those coordinate values into memory.
        inline_array: Place backend arrays directly in Dask tasks instead of
            separate graph nodes; applies when Dask chunking is used.
        chunked_array_type: Registered xarray chunk manager, such as `"dask"`.
        from_array_kwargs: Extra keywords for the chunk manager's `from_array`
            operation, such as options for `dask.array.from_array`.
        backend_kwargs: Extra keywords for h5netcdf's open function.
    """

    cache: bool | None
    decode_cf: bool | None
    decode_times: bool | CFDatetimeCoder | Mapping[str, bool | CFDatetimeCoder] | None
    decode_timedelta: (
        bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None
    )
    use_cftime: bool | Mapping[str, bool] | None
    concat_characters: bool | Mapping[str, bool] | None
    drop_variables: str | Sequence[str]
    create_default_indexes: bool
    inline_array: bool
    chunked_array_type: str | None
    from_array_kwargs: dict[str, Any] | None
    backend_kwargs: dict[str, Any] | None


class NetCDFWriteOptions(TypedDict, total=False):
    """Optional `Dataset.to_netcdf` behavior when writing netCDF.

    Omitted keys use xarray or h5netcdf defaults. GeoSave owns the
    destination, engine, write mode, and computation arguments. Files are
    netCDF4 (HDF5), written by h5netcdf through the destination's filesystem.

    Attributes:
        encoding: Variable names mapped to storage settings such as `dtype`,
            `_FillValue`, `scale_factor`, compression, and chunk sizes.
            If provided, xarray uses this mapping instead of existing encodings.
        unlimited_dims: Dimensions stored as unlimited dimensions. This
            declares file structure; GeoSave still writes a new complete file.
        invalid_netcdf: Allow HDF5 features outside the netCDF standard.
    """

    encoding: Mapping[Hashable, Mapping[str, Any]] | Mapping[str, Any]
    unlimited_dims: Sequence[Hashable]
    invalid_netcdf: bool


def write_netcdf(
    raster: xr.Dataset,
    destination: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[NetCDFWriteOptions],
) -> UPath | Delayed:
    """Write a raster as one netCDF file.

    Args:
        raster: Raster Dataset.
        destination: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in `.nc`, `.nc4`, or `.cdf`.
        compute: False returns one native `dask.Delayed`; computing it writes
            the file and returns its path.
        overwrite: Replace an existing destination when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported xarray netCDF write options.

    Returns:
        Path of the file written, or one delayed result.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        KeyError: An encoding names no variable of the raster.
        ValueError: The destination path is invalid.

    Examples:
        >>> write_netcdf(raster, "scene.nc")
        UPath('/abs/scene.nc')
    """
    path = storage.create_upath(destination, storage_options)
    storage.check_destination(path, SUFFIXES, overwrite=overwrite)
    check_encoding(raster.variables, write_options.get("encoding"))
    encoded = codecs.encode_dataset(raster, CODECS)
    write = partial(_store, encoded, path, overwrite=overwrite, **write_options)
    return storage.defer(write, compute=compute)


def write_netcdf_tree(
    tree: xr.DataTree,
    destination: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[NetCDFWriteOptions],
) -> UPath | Delayed:
    """Write a raster tree as one netCDF file with a group per node.

    Args:
        tree: Tree of raster Datasets.
        destination: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in `.nc`, `.nc4`, or `.cdf`.
        compute: False returns one native `dask.Delayed`; computing it writes
            the file and returns its path.
        overwrite: Replace an existing destination when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported xarray netCDF write options. `encoding` is
            keyed by variable name and reaches every group holding it.

    Returns:
        Path of the file written, or one delayed result.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        KeyError: An encoding names no variable of the tree.
        ValueError: The destination path is invalid.

    Examples:
        >>> write_netcdf_tree(tree, "scene.nc")
        UPath('/abs/scene.nc')
    """
    path = storage.create_upath(destination, storage_options)
    storage.check_destination(path, SUFFIXES, overwrite=overwrite)
    encoded = codecs.encode_tree(tree, CODECS)
    write_options["encoding"] = split_encoding(encoded, write_options.get("encoding"))
    write = partial(
        _store,
        encoded,
        path,
        overwrite=overwrite,
        # A group's grid mapping must sit in that group for its CF decoding.
        write_inherited_coords=True,
        **write_options,
    )
    return storage.defer(write, compute=compute)


def _store(
    encoded: xr.Dataset | xr.DataTree, path: UPath, *, overwrite: bool, **options: Any
) -> UPath:
    """Hand an encoded raster or tree to xarray's h5netcdf writer.

    HDF5 seeks back into its file, so the file is staged locally.
    """
    storage.refuse_existing(path, overwrite=overwrite)
    with storage.stage_file(path) as local:
        encoded.to_netcdf(local, mode="w", engine="h5netcdf", compute=True, **options)
    return path


def open_netcdf(
    source: Location,
    *,
    group: str | None = None,
    chunks: NetCDFChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[NetCDFOpenOptions],
) -> Dataset:
    """Open one NetCDF group as a Dataset.

    Args:
        source: NetCDF path, URL, or UPath, read lazily through its filesystem.
        group: Group path to open; None opens the root group.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported `xarray.open_dataset` options.

    Returns:
        Dataset for the selected group. Child groups are not opened;
        use `open_netcdf_tree` for the hierarchy. Closing releases the file.

    Examples:
        >>> open_netcdf("scene.nc", group="s2/r10m").gs.variables
        ('B04', 'B08')
    """
    path = storage.create_upath(source, storage_options)
    with ExitStack() as handles:
        # The file stays open while the lazy arrays read from it.
        handle = handles.enter_context(path.open("rb"))
        stored = xr.open_dataset(
            handle,
            engine="h5netcdf",
            group=group,
            # A grid mapping variable is a coordinate; CF's default loses it.
            decode_coords="all",
            chunks=chunks,
            mask_and_scale=mask_and_scale,
            **open_options,
        )
        handles.callback(stored.close)
        opened = codecs.decode_dataset(stored, CODECS)
        if opened is stored:
            opened = stored.copy(deep=False)
        opened.encoding["source"] = str(path.absolute())
        opened.set_close(handles.pop_all().close)
        return cast("Dataset", opened)


def open_netcdf_tree(
    source: Location,
    *,
    chunks: NetCDFChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[NetCDFOpenOptions],
) -> DataTree:
    """Open a whole NetCDF file as a DataTree, including root-only files.

    Args:
        source: NetCDF path, URL, or UPath, read lazily through its filesystem.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported `xarray.open_datatree` options.

    Returns:
        DataTree with stored groups. Closing releases the file.

    Examples:
        >>> list(open_netcdf_tree("scene.nc").gs.datasets)
        ['s2/r10m', 's2/r20m']
    """
    path = storage.create_upath(source, storage_options)
    with ExitStack() as handles:
        handle = handles.enter_context(path.open("rb"))
        stored = xr.open_datatree(
            handle,
            engine="h5netcdf",
            decode_coords="all",
            chunks=chunks,
            mask_and_scale=mask_and_scale,
            **open_options,
        )
        handles.callback(stored.close)
        opened = codecs.decode_tree(stored, CODECS)
        opened.set_close(handles.pop_all().close)
        return cast("DataTree", opened)
