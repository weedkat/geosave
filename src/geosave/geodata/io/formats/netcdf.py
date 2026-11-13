"""Open and write raster Datasets and trees as netCDF files."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import xarray as xr
from upath import UPath

from geosave.geodata.io import storage
from geosave.geodata.io.formats.utils.stores import check_store
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from dask.delayed import Delayed
    from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder

    from geosave.geodata import Dataset, DataTree

FILE_SUFFIXES = (".nc", ".nc4", ".cdf")

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
    raster: xr.Dataset | xr.DataTree,
    destination: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[NetCDFWriteOptions],
) -> UPath | Delayed:
    """Write a raster, or a raster tree with its groups, as one netCDF file.

    Args:
        raster: Raster Dataset, or a tree of them.
        destination: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in `.nc`, `.nc4`, or `.cdf`.
        compute: False returns one native `dask.Delayed`; computing it writes
            the file and returns its path.
        overwrite: Replace an existing destination when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported xarray netCDF write options. `encoding` is
            keyed by variable name and reaches every group holding it.

    Returns:
        Path of the artifact actually written, or one delayed result.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        KeyError: An encoding names no variable of the raster.
        TypeError: `raster` is neither a Dataset nor a DataTree.
        ValueError: The destination path is invalid.

    Examples:
        >>> write_netcdf(raster, "scene.nc")
        UPath('/abs/scene.nc')
    """
    if not isinstance(raster, (xr.Dataset, xr.DataTree)):
        raise TypeError(
            f"raster must be an xarray Dataset or DataTree, got {type(raster).__name__}"
        )
    store = storage.create_upath(destination, storage_options)
    check_store(
        raster,
        store,
        FILE_SUFFIXES,
        overwrite=overwrite,
        encoding=write_options.get("encoding"),
    )
    if compute:
        return _write(raster, store, dict(write_options))
    from dask.delayed import delayed

    # A partial keeps the raster's lazy arrays out of the graph's dependencies.
    return delayed(partial(_write, raster, store, dict(write_options)))()


def _write(
    raster: xr.Dataset | xr.DataTree, store: UPath, options: dict[str, Any]
) -> UPath:
    """Write a checked raster through the destination's filesystem."""
    from geosave.geodata.io.formats.cf import (
        encode_spatial_axes,
        encode_time_bounds,
    )

    if isinstance(raster, xr.DataTree):
        # Cells are stored as CF bounds; coordinates carry no fill (CF §5, §7.1).
        nodes = {}
        for node in raster.subtree:
            dataset = node.to_dataset(inherit=False)
            dataset = encode_time_bounds(dataset)
            nodes[node.path] = encode_spatial_axes(dataset)
        raster = xr.DataTree.from_dict(nodes)
        shared = dict(options.pop("encoding", None) or {})
        encoding: dict[str, dict[Hashable, Any]] = {}
        for node in raster.subtree:
            encoding[node.path] = {}
            for name, settings in shared.items():
                if name in node.variables:
                    encoding[node.path][name] = settings
        options["encoding"] = encoding
        # A group's grid mapping must sit in that group for its CF decoding.
        options["write_inherited_coords"] = True
    else:
        raster = encode_spatial_axes(encode_time_bounds(raster))

    with storage.stage_file(store) as local:
        raster.to_netcdf(local, mode="w", engine="h5netcdf", compute=True, **options)
    return store


def open_netcdf(
    source: Location,
    *,
    chunks: NetCDFChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[NetCDFOpenOptions],
) -> Dataset | DataTree:
    """Open a NetCDF file, preserving child groups when present.

    Args:
        source: NetCDF path, URL, or UPath, read lazily through its filesystem.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported `xarray.open_datatree` options.

    Returns:
        Dataset for a root-only file, otherwise DataTree with stored groups.
        Closing the result releases the file.

    Examples:
        >>> list(open_netcdf("scene.nc").gs.datasets)
        ['s2/r10m', 's2/r20m']
    """
    from geosave.geodata.io.formats.cf import decode_time_bounds

    path = storage.create_upath(source, storage_options)
    with ExitStack() as opened:
        # The file stays open while the lazy arrays read from it.
        handle = opened.enter_context(path.open("rb"))
        stored = xr.open_datatree(
            handle,
            engine="h5netcdf",
            # A grid mapping variable is a coordinate; CF's default loses it.
            decode_coords="all",
            chunks=chunks,
            mask_and_scale=mask_and_scale,
            **open_options,
        )
        opened.callback(stored.close)
        # Stored CF bounds come back as start/end coordinates.
        if stored.children:
            nodes = {}
            for node in stored.subtree:
                nodes[node.path] = decode_time_bounds(node.to_dataset(inherit=False))
            result = xr.DataTree.from_dict(nodes)
        else:
            result = decode_time_bounds(stored.to_dataset().copy(deep=False))
            result.encoding["source"] = str(path.absolute())
        result.set_close(opened.pop_all().close)
        return cast("Dataset | DataTree", result)
