"""Open and write raster Datasets and trees as Zarr stores."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import xarray as xr
import zarr
from upath import UPath
from xarray.core.types import T_Chunks

from geosave.geodata.io.formats.utils.stores import check_store
from geosave.geodata.io.storage import (
    Location,
    StorageOptions,
    absolute_location,
    location,
)

if TYPE_CHECKING:
    from dask.delayed import Delayed

    from geosave.geodata import Dataset, DataTree

type ZarrChunkSpec = T_Chunks

STORE_SUFFIX = ".zarr"
_STORE_ZARR_FORMAT = 3
_VARIABLE_ORDER_ATTR = "zarr_variable_order"
_GROUP_ORDER_ATTR = "zarr_group_order"


class ZarrOpenOptions(TypedDict, total=False):
    """Optional xarray behavior supported when opening Zarr.

    CF decoding options are absent because disabling them drops `spatial_ref`,
    the time axis, or the spatial index.

    Omitted keys use xarray or Zarr defaults, except `consolidated=False`,
    which GeoSave supplies. The reader enables grid-mapping coordinate decoding.

    Attributes:
        synchronizer: Zarr synchronization object where the backend supports
            it. The current Zarr 3 library warns that this is unimplemented.
        drop_variables: One variable name or a sequence to exclude while opening.
        consolidated: Require consolidated metadata when True; False reads
            ordinary metadata. None delegates metadata discovery to xarray.
        overwrite_encoded_chunks: Discard stored chunk sizes from each variable's
            encoding when opening with a different requested chunk layout.
        chunk_store: Separate store for chunk data where the backend supports
            it. The current Zarr 3 library warns that this is unimplemented.
        storage_options: Filesystem credentials and connection options merged
            into the source UPath. These are forwarded only for remote stores.
        decode_timedelta: Decode CF duration variables as timedeltas; False
            retains numbers, and None delegates the decoding policy.
        use_zarr_fill_value_as_mask: Use the Zarr fill value as missing-value
            metadata for CF decoding with `mask_and_scale=True`. False ignores
            it; None delegates to the store-format policy.
        chunked_array_type: Registered xarray chunk manager, such as `"dask"`.
        from_array_kwargs: Extra keywords for the chunk manager's `from_array`
            operation, such as options for `dask.array.from_array`.
    """

    synchronizer: Any
    drop_variables: str | Sequence[str]
    consolidated: bool | None
    overwrite_encoded_chunks: bool
    chunk_store: Any
    storage_options: Mapping[str, Any]
    decode_timedelta: bool | None
    use_zarr_fill_value_as_mask: bool | None
    chunked_array_type: str | None
    from_array_kwargs: dict[str, Any] | None


class ZarrWriteOptions(TypedDict, total=False):
    """Optional xarray behavior supported when writing Zarr.

    Layout options are absent on purpose: a store holds one raster at its
    root, in the Zarr format this writer fixes. `region` is absent because this writer fixes `mode`, which it forbids.

    Omitted keys use xarray or Zarr defaults, except `consolidated=False`.
    GeoSave writes Zarr format 3 and supplies per-variable fill encodings.

    Attributes:
        encoding: Variable names mapped to storage settings such as `dtype`,
            `chunks`, packing, and fill values. A supplied variable entry replaces
            GeoSave's generated fill encoding for that variable.
        consolidated: Consolidate metadata when True; False leaves it ordinary.
            None delegates to xarray and the installed Zarr implementation.
        safe_chunks: Require each Dask chunk to cover complete storage chunks.
            False disables this check; it does not align the chunks itself.
        align_chunks: Rechunk Dask arrays to align with the requested Zarr chunk
            boundaries before writing.
        write_empty_chunks: Store chunks consisting entirely of the fill value
            when True; False omits them. None uses encoding or Zarr defaults.
        chunkmanager_store_kwargs: Extra keywords for the chunk manager's
            `store` operation, such as options for `dask.array.store`.
        synchronizer: Zarr synchronization object where the backend supports
            it. The current Zarr 3 library warns that this is unimplemented.
    """

    encoding: Mapping[Hashable, Mapping[str, Any]] | Mapping[str, Any]
    consolidated: bool | None
    safe_chunks: bool
    align_chunks: bool
    write_empty_chunks: bool | None
    chunkmanager_store_kwargs: dict[str, Any] | None
    synchronizer: Any


def _recording_order(ds: xr.Dataset) -> xr.Dataset:
    """Write the Dataset's own variable order into its attrs.

    Args:
        ds: Raster about to be written.

    Returns:
        Raster with its variable order recorded in storage attrs, or `ds`
        itself where it holds no variables to order.
    """
    if not ds.data_vars:
        return ds
    names = [str(name) for name in ds.data_vars]
    return ds.assign_attrs({_VARIABLE_ORDER_ATTR: names})


def _in_written_order(ds: xr.Dataset) -> xr.Dataset:
    """Put the variables back in the order the store was written in.

    A Zarr group lists its members in no order, so the written one is read from
    storage attrs. Variables it does not name trail the ones it does, which is
    where a variable written by something else ends up.

    Args:
        ds: Dataset as the store listed it.

    Returns:
        Dataset holding the same variables, ordered as written and without
        the order attr, or `ds` itself where the store names no order.
    """
    order = ds.attrs.get(_VARIABLE_ORDER_ATTR)
    if order is None:
        return ds
    named = [name for name in order if name in ds.data_vars]
    trailing = [name for name in ds.data_vars if name not in named]
    ordered = ds[[*named, *trailing]].assign_coords(ds.coords)
    # The order is restored, so the attr would only leak into other formats.
    ordered.attrs = {
        key: value for key, value in ds.attrs.items() if key != _VARIABLE_ORDER_ATTR
    }
    return ordered


def write_zarr(
    raster: xr.Dataset | xr.DataTree,
    destination: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[ZarrWriteOptions],
) -> UPath | Delayed:
    """Write a raster, or a raster tree with its groups, as one Zarr store.

    A Zarr group records no member order, so each group's variable order
    travels in storage attrs, and a tree's group order too, for `open_raster`
    and `open_raster` to restore. Each variable's fill value is written as its
    array's own, which is what a GDAL reader masks on.

    Args:
        raster: Raster Dataset, or a tree of them.
        destination: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in `.zarr`.
        compute: False returns one native `dask.Delayed`; computing it writes
            the store and returns its path.
        overwrite: Replace an existing destination when true.
        storage_options: Filesystem options merged into the destination UPath.
        **write_options: Supported xarray Zarr write options. `encoding` is
            keyed by variable name and reaches every group holding it.

    Returns:
        Path of the store actually written, or one delayed result.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        KeyError: An encoding names no variable of the raster.
        TypeError: `raster` is neither a Dataset nor a DataTree.
        ValueError: The destination does not end in `.zarr`.

    Examples:
        >>> write_zarr(raster, "scene.zarr")
        UPath('/abs/scene.zarr')
    """
    if not isinstance(raster, (xr.Dataset, xr.DataTree)):
        raise TypeError(
            f"raster must be an xarray Dataset or DataTree, got {type(raster).__name__}"
        )
    store = location(destination, storage_options)
    check_store(
        raster,
        store,
        (STORE_SUFFIX,),
        overwrite=overwrite,
        encoding=write_options.get("encoding"),
    )
    if compute:
        return _write(raster, store, overwrite, dict(write_options))
    from dask.delayed import delayed

    # A partial keeps the raster's lazy arrays out of the graph's dependencies.
    return delayed(partial(_write, raster, store, overwrite, dict(write_options)))()


def _write(
    raster: xr.Dataset | xr.DataTree,
    store: UPath,
    overwrite: bool,
    options: dict[str, Any],
) -> UPath:
    """Write a checked raster, refusing a destination that appeared since the check."""
    if not overwrite and store.exists():
        raise FileExistsError(f"{store} exists; pass overwrite=True to replace it")
    # zarr refuses storage_options for a local path, so only a URL passes them.
    if store.storage_options:
        options["storage_options"] = dict(store.storage_options)
    options.setdefault("consolidated", False)
    if isinstance(raster, xr.DataTree):
        _write_tree(raster, store, overwrite, options)
    else:
        _write_dataset(raster, store, overwrite, options)
    return store.absolute()


def _write_dataset(
    raster: xr.Dataset, store: UPath, overwrite: bool, options: dict[str, Any]
) -> None:
    """Write one Dataset at the store's root."""
    from geosave.geodata.core.raster.utils.cf import (
        drop_coordinate_fill,
        encode_time_bounds,
    )

    # Cells are stored as CF bounds; coordinates carry no fill (CF §5, §7.1).
    raster = _recording_order(drop_coordinate_fill(encode_time_bounds(raster)))
    options["encoding"] = {
        **fill_encoding(raster),
        **dict(options.get("encoding") or {}),
    }
    raster.to_zarr(
        absolute_location(store),
        mode="w" if overwrite else "w-",
        zarr_format=_STORE_ZARR_FORMAT,
        compute=True,
        **options,
    )


def _write_tree(
    tree: xr.DataTree, store: UPath, overwrite: bool, options: dict[str, Any]
) -> None:
    """Write a tree with one group per node."""
    from geosave.geodata.core.raster.utils.cf import (
        drop_coordinate_fill,
        encode_time_bounds,
    )

    shared = dict(options.pop("encoding", None) or {})
    nodes = {}
    encoding = {}
    for node in tree.subtree:
        # Cells are stored as CF bounds; coordinates carry no fill (CF §5, §7.1).
        dataset = drop_coordinate_fill(
            encode_time_bounds(node.to_dataset(inherit=False))
        )
        recorded = _recording_order(dataset)
        if node.children:
            # A Zarr group lists its subgroups in no order either.
            recorded = recorded.assign_attrs({_GROUP_ORDER_ATTR: list(node.children)})
        nodes[node.path] = recorded
        selected = {}
        for name, settings in shared.items():
            if name in dataset.variables:
                selected[name] = settings
        encoding[node.path] = {**fill_encoding(dataset), **selected}
    xr.DataTree.from_dict(nodes).to_zarr(
        absolute_location(store),
        mode="w" if overwrite else "w-",
        encoding=encoding,
        zarr_format=_STORE_ZARR_FORMAT,
        # A group's grid mapping must sit in that group for its CF decoding.
        write_inherited_coords=True,
        compute=True,
        **options,
    )


def open_zarr(
    source: Location,
    *,
    chunks: ZarrChunkSpec = None,
    mask_and_scale: bool = False,
    **open_options: Unpack[ZarrOpenOptions],
) -> Dataset | DataTree:
    """Open a Zarr store, preserving child groups when present.

    Args:
        source: Local Zarr path, fsspec URL, or UPath.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values.
        **open_options: Supported `xarray.open_datatree` options.

    Returns:
        Dataset for a root-only store, otherwise DataTree with stored group
        and variable order. Closing the result releases the store.

    Examples:
        >>> list(open_zarr("scene.zarr").gs.datasets)
        ['s2/r10m', 's2/r20m']
    """
    options: dict[str, Any] = dict(open_options)
    store = location(source, options.pop("storage_options", None))
    remote_options = dict(store.storage_options) or None
    options.setdefault("consolidated", False)
    options.setdefault("decode_coords", "all")
    opened = xr.open_datatree(
        absolute_location(store),
        engine="zarr",
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        storage_options=remote_options,
        **options,
    )
    try:
        nodes: dict[str, xr.Dataset] = {}
        _read_in_written_order(opened, nodes)
        if opened.children:
            result = xr.DataTree.from_dict(nodes)
        else:
            result = nodes["/"]
            group = zarr.open_group(
                absolute_location(store), mode="r", storage_options=remote_options
            )
            result.encoding["zarr"] = {
                "zarr_format": group.metadata.zarr_format,
                "node_type": group.metadata.node_type,
                "consolidated": group.metadata.consolidated_metadata is not None,
            }
            result.encoding["source"] = absolute_location(store)
        result.set_close(opened.close)
        return cast("Dataset | DataTree", result)
    except BaseException:
        opened.close()
        raise


def _read_in_written_order(node: xr.DataTree, nodes: dict[str, xr.Dataset]) -> None:
    """Add `node` and its descendants to `nodes`, groups and variables as written."""
    from geosave.geodata.core.raster.utils.cf import decode_time_bounds

    dataset = decode_time_bounds(_in_written_order(node.to_dataset(inherit=False)))
    order = dataset.attrs.pop(_GROUP_ORDER_ATTR, [])
    nodes[node.path] = dataset
    names = [name for name in order if name in node.children]
    for name in node.children:
        if name not in names:
            names.append(name)
    for name in names:
        _read_in_written_order(node.children[name], nodes)


def fill_encoding(raster: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Say which fill value each variable's Zarr array should hold.

    A Zarr array carries a fill value of its own, defaulting to zero, and that
    is what a GDAL reader masks on — the CF `_FillValue` attr rides alongside
    for a CF reader but is not read as the array's own.

    Args:
        raster: Raster Dataset about to be written.

    Returns:
        {
            "<variable name>": {"fill_value": the value it calls absent},
        }
        Variables declaring none are absent, leaving Zarr its own default.
        Existing CF grid-mapping references accompany each generated fill
        encoding.

    Examples:
        >>> fill_encoding(raster)
        {'red': {'fill_value': 0, 'grid_mapping': 'spatial_ref'}, 'nir': {'fill_value': 0, 'grid_mapping': 'spatial_ref'}}
    """
    declared: dict[str, dict[str, Any]] = {}
    for name, variable in raster.data_vars.items():
        fill = variable.gs.nodata
        if fill is None:
            fill = variable.encoding.get("_FillValue")
        if fill is not None:
            # xarray consumes coordinates before backend options, but grid_mapping after.
            encoding = variable.encoding
            declared[str(name)] = {"fill_value": fill}
            for key in (
                "grid_mapping",
                "dtype",
                "scale_factor",
                "add_offset",
                "_FillValue",
                "missing_value",
            ):
                if key in encoding:
                    declared[str(name)][key] = encoding[key]
    return declared
