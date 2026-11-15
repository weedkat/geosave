"""Open and write raster Datasets and trees as Zarr stores.

xarray writes and reads the store. The codecs add what a Zarr group cannot
state on its own: CF axes, each array's fill value, and member order.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import xarray as xr
from upath import UPath
from xarray.core.types import T_Chunks

from geosave.geodata.io import storage
from geosave.geodata.io.raster import codecs
from geosave.geodata.io.raster.codecs import Codec
from geosave.geodata.io.raster.codecs.cf import SPATIAL_AXES
from geosave.geodata.io.raster.encoding import check_encoding, split_encoding
from geosave.geodata.io.storage import Location, StorageOptions

if TYPE_CHECKING:
    from dask.delayed import Delayed

    from geosave.geodata import Dataset, DataTree

type ZarrChunkSpec = T_Chunks

SUFFIXES = (".zarr",)
_ZARR_FORMAT = 3
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


def record_variable_order(raster: xr.Dataset) -> xr.Dataset:
    """Write the Dataset's own variable order into its attrs.

    Args:
        raster: Raster about to be written.

    Returns:
        Raster with its variable order recorded in storage attrs, or `raster`
        itself where it holds no variables to order.
    """
    if not raster.data_vars:
        return raster
    names = [str(name) for name in raster.data_vars]
    return raster.assign_attrs({_VARIABLE_ORDER_ATTR: names})


def restore_variable_order(raster: xr.Dataset) -> xr.Dataset:
    """Put the variables back in the order the store was written in.

    A Zarr group lists its members in no order, so the written one is read from
    storage attrs. Variables it does not name trail the ones it does, which is
    where a variable written by something else ends up.

    Args:
        raster: Dataset as the store listed it.

    Returns:
        Dataset holding the same variables, ordered as written and without
        the order attr, or `raster` itself where the store names no order.
    """
    order = raster.attrs.get(_VARIABLE_ORDER_ATTR)
    if order is None:
        return raster
    named = [name for name in order if name in raster.data_vars]
    trailing = [name for name in raster.data_vars if name not in named]
    ordered = raster[[*named, *trailing]].assign_coords(raster.coords)
    # The order is restored, so the attr would only leak into other formats.
    ordered.attrs = dict(raster.attrs)
    del ordered.attrs[_VARIABLE_ORDER_ATTR]
    return ordered


def encode_fill_value(raster: xr.Dataset) -> xr.Dataset:
    """State each variable's absent value as its Zarr array's own fill value.

    A Zarr array carries a fill value of its own, defaulting to zero, and that
    is what a GDAL reader masks on. The CF `_FillValue` attr rides alongside
    for a CF reader but is not read as the array's own.

    Args:
        raster: Raster about to be written.

    Returns:
        Shallow copy whose variables declaring an absent value carry it as
        `encoding["fill_value"]`; the others leave Zarr its default.

    Examples:
        >>> encode_fill_value(raster).red.encoding["fill_value"]
        0
    """
    encoded = raster.copy(deep=False)
    for variable in encoded.data_vars.values():
        fill = variable.gs.nodata
        if fill is None:
            fill = variable.encoding.get("_FillValue")
        if fill is not None:
            variable.encoding["fill_value"] = fill
    return encoded


FILL_VALUE = Codec(encode_fill_value)
VARIABLE_ORDER = Codec(record_variable_order, restore_variable_order)

# Time cells stay start/end coordinates; Zarr stores datetimes as they are.
CODECS = (SPATIAL_AXES, FILL_VALUE, VARIABLE_ORDER)


def record_group_order(tree: xr.DataTree) -> xr.DataTree:
    """Write each node's child order into its attrs.

    A Zarr group lists its subgroups in no order either.

    Args:
        tree: Tree about to be written.

    Returns:
        Copy whose nodes with children name them, in order, in storage attrs.
    """
    recorded = tree.copy()
    for node in recorded.subtree:
        if node.children:
            node.attrs[_GROUP_ORDER_ATTR] = list(node.children)
    return recorded


def restore_group_order(tree: xr.DataTree) -> xr.DataTree:
    """Put every node's children back in the order the store was written in.

    Children the stored order does not name trail the ones it does.

    Args:
        tree: Tree as the store listed it.

    Returns:
        Tree of the same groups, ordered as written and without the order attrs.
    """
    nodes: dict[str, xr.Dataset] = {}
    _restore_node(tree, nodes)
    return xr.DataTree.from_dict(nodes)


def _restore_node(node: xr.DataTree, nodes: dict[str, xr.Dataset]) -> None:
    """Add `node`, then its descendants in written order, to `nodes`."""
    dataset = node.to_dataset(inherit=False)
    order = dataset.attrs.pop(_GROUP_ORDER_ATTR, [])
    nodes[node.path] = dataset
    names = [name for name in order if name in node.children]
    for name in node.children:
        if name not in names:
            names.append(name)
    for name in names:
        _restore_node(node.children[name], nodes)


def write_zarr(
    raster: xr.Dataset,
    destination: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[ZarrWriteOptions],
) -> UPath | Delayed:
    """Write a raster as one Zarr store.

    A Zarr group records no member order, so the variable order travels in
    storage attrs for `open_zarr` to restore. Each variable's fill value is
    written as its array's own, which is what a GDAL reader masks on.

    Args:
        raster: Raster Dataset.
        destination: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in `.zarr`.
        compute: False returns one native `dask.Delayed`; computing it writes
            the store and returns its path.
        overwrite: Replace an existing destination when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported xarray Zarr write options.

    Returns:
        Path of the store actually written, or one delayed result.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        KeyError: An encoding names no variable of the raster.
        ValueError: The destination does not end in `.zarr`.

    Examples:
        >>> write_zarr(raster, "scene.zarr")
        UPath('/abs/scene.zarr')
    """
    store = storage.create_upath(destination, storage_options)
    storage.check_destination(store, SUFFIXES, overwrite=overwrite)
    check_encoding(raster.variables, write_options.get("encoding"))
    encoded = codecs.encode_dataset(raster, CODECS)
    write = partial(_store, encoded, store, overwrite=overwrite, **write_options)
    return storage.defer(write, compute=compute)


def write_zarr_tree(
    tree: xr.DataTree,
    destination: Location,
    *,
    compute: bool = True,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[ZarrWriteOptions],
) -> UPath | Delayed:
    """Write a raster tree as one Zarr store with a group per node.

    Each group is encoded as `write_zarr` encodes a raster, and the group
    order travels in storage attrs alongside each group's variable order.

    Args:
        tree: Tree of raster Datasets.
        destination: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in `.zarr`.
        compute: False returns one native `dask.Delayed`; computing it writes
            the store and returns its path.
        overwrite: Replace an existing destination when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported xarray Zarr write options. `encoding` is
            keyed by variable name and reaches every group holding it.

    Returns:
        Path of the store actually written, or one delayed result.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        KeyError: An encoding names no variable of the tree.
        ValueError: The destination does not end in `.zarr`.

    Examples:
        >>> write_zarr_tree(tree, "scene.zarr")
        UPath('/abs/scene.zarr')
    """
    store = storage.create_upath(destination, storage_options)
    storage.check_destination(store, SUFFIXES, overwrite=overwrite)
    encoded = record_group_order(codecs.encode_tree(tree, CODECS))
    write_options["encoding"] = split_encoding(encoded, write_options.get("encoding"))
    write = partial(
        _store,
        encoded,
        store,
        overwrite=overwrite,
        # A group's grid mapping must sit in that group for its CF decoding.
        write_inherited_coords=True,
        **write_options,
    )
    return storage.defer(write, compute=compute)


def _store(
    encoded: xr.Dataset | xr.DataTree, store: UPath, *, overwrite: bool, **options: Any
) -> UPath:
    """Hand an encoded raster or tree to xarray's Zarr writer.

    `mode="w-"` refuses a destination that appeared since it was checked.
    """
    options.setdefault("consolidated", False)
    encoded.to_zarr(
        str(store),
        mode="w" if overwrite else "w-",
        zarr_format=_ZARR_FORMAT,
        # zarr refuses empty storage_options for a local path.
        storage_options=dict(store.storage_options) or None,
        compute=True,
        **options,
    )
    return store.absolute()


def open_zarr(
    source: Location,
    *,
    group: str | None = None,
    chunks: ZarrChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[ZarrOpenOptions],
) -> Dataset:
    """Open one Zarr group as a Dataset.

    Args:
        source: Local Zarr path, fsspec URL, or UPath.
        group: Group path to open; None opens the root group.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported xarray Zarr opening options.

    Returns:
        Dataset with stored variable order. Child groups are not opened;
        use `open_zarr_tree` for the hierarchy. Closing releases the store.

    Examples:
        >>> open_zarr("scene.zarr", group="s2/r10m").gs.variables
        ('B04', 'B08')
    """
    options: dict[str, Any] = dict(open_options)
    options.setdefault("consolidated", False)
    options.setdefault("decode_coords", "all")
    store = storage.create_upath(source, storage_options)
    stored = xr.open_dataset(
        str(store),
        engine="zarr",
        group=group,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        # zarr refuses empty storage_options for a local path.
        storage_options=dict(store.storage_options) or None,
        **options,
    )
    try:
        opened = codecs.decode_dataset(stored, CODECS)
        opened.attrs.pop(_GROUP_ORDER_ATTR, None)
        opened.encoding["source"] = str(store.absolute())
        if opened is not stored:
            opened.set_close(stored.close)
        return cast("Dataset", opened)
    except BaseException:
        stored.close()
        raise


def open_zarr_tree(
    source: Location,
    *,
    chunks: ZarrChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[ZarrOpenOptions],
) -> DataTree:
    """Open a whole Zarr store as a DataTree, including root-only stores.

    Args:
        source: Local Zarr path, fsspec URL, or UPath.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values.
        storage_options: Options for the filesystem a URL names.
        **open_options: Supported `xarray.open_datatree` options.

    Returns:
        DataTree with stored group and variable order. Closing releases the store.

    Examples:
        >>> list(open_zarr_tree("scene.zarr").gs.datasets)
        ['s2/r10m', 's2/r20m']
    """
    options: dict[str, Any] = dict(open_options)
    options.setdefault("consolidated", False)
    options.setdefault("decode_coords", "all")
    store = storage.create_upath(source, storage_options)
    stored = xr.open_datatree(
        str(store),
        engine="zarr",
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        storage_options=dict(store.storage_options) or None,
        **options,
    )
    try:
        opened = restore_group_order(codecs.decode_tree(stored, CODECS))
        opened.set_close(stored.close)
        return cast("DataTree", opened)
    except BaseException:
        stored.close()
        raise
