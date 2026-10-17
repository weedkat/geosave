"""Open and write raster Datasets as Zarr stores."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import xarray as xr
import zarr
from xarray.core.types import T_Chunks

from geosave.geodata.conventions import to_yx
from ..storage import Location, absolute_location, location
from .file import RasterFile


if TYPE_CHECKING:
    from geosave.geodata import Dataset

type ZarrChunkSpec = T_Chunks

_STORE_SUFFIX = ".zarr"
_STORE_ZARR_FORMAT = 3
_VARIABLE_ORDER_ATTR = "zarr_variable_order"


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


def open_raster(
    source: Location,
    *,
    chunks: ZarrChunkSpec = None,
    mask_and_scale: bool = False,
    **open_options: Unpack[ZarrOpenOptions],
) -> Dataset:
    """Open a Zarr store as a raster Dataset.

    Args:
        source: Local Zarr path, fsspec URL, or UPath.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        **open_options: Supported `xarray.open_zarr` options.

    Returns:
        Raster Dataset carrying stored digital numbers, or physical values
        when `mask_and_scale` asked for them. Variables come back in the order
        they were written, recorded in the store's attrs.

    Raises:
        ValueError: The store cannot be read.

    Examples:
        >>> raster = open_raster("scene.zarr")
        >>> str(raster.red.dtype)
        'uint16'
    """
    options: dict[str, Any] = dict(open_options)
    store = location(source, options.pop("storage_options", None))
    # zarr refuses storage_options for a local path, so only a URL passes them.
    remote_options = dict(store.storage_options) or None
    options.setdefault("consolidated", False)
    # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
    options.setdefault("decode_coords", "all")
    opened = xr.open_dataset(
        absolute_location(store),
        engine="zarr",
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        storage_options=remote_options,
        **options,
    )
    try:
        cube = to_yx(_in_written_order(opened))
        group = zarr.open_group(
            absolute_location(store), mode="r", storage_options=remote_options
        )
        cube.encoding["zarr"] = {
            "zarr_format": group.metadata.zarr_format,
            "node_type": group.metadata.node_type,
            "consolidated": group.metadata.consolidated_metadata is not None,
        }
    except BaseException:
        opened.close()
        raise
    if cube is not opened:
        cube.set_close(opened.close)
    cube.encoding["source"] = absolute_location(source)
    return cast("Dataset", cube)


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
    ordered = ds[[*named, *trailing]]
    # The order is restored, so the attr would only leak into other formats.
    ordered.attrs = {
        key: value for key, value in ds.attrs.items() if key != _VARIABLE_ORDER_ATTR
    }
    return ordered


def write(
    raster: xr.Dataset,
    destination: Location,
    *,
    overwrite: bool = False,
    **write_options: Unpack[ZarrWriteOptions],
) -> RasterFile:
    """Write a raster to a Zarr store.

    A Zarr group records no member order, so a Dataset's variable order
    travels in storage attrs for `read` to restore.

    Args:
        raster: Raster Dataset.
        destination: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in `.zarr`.
        overwrite: Replace an existing destination when true.
        **write_options: Supported xarray Zarr write options.

    Returns:
        The written store with the header it states, as `open_raster` reads it.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        TypeError: `raster` is not a Dataset.
        ValueError: The destination does not end in `.zarr`.

    Examples:
        >>> write(raster, "scene.zarr")
        RasterFile(/abs/scene.zarr, ['red', 'nir'], 2025-06-01)
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            f"raster must be an xarray Dataset, got {type(raster).__name__}; "
            f"write a tree with its own gs.to_zarr"
        )

    options: dict[str, Any] = dict(write_options)
    store = location(destination)
    if store.suffix.lower() != _STORE_SUFFIX:
        raise ValueError(f"destination {store.name!r} must end in {_STORE_SUFFIX!r}")
    if not overwrite and store.exists():
        raise FileExistsError(
            f"{destination} exists; pass overwrite=True to replace it"
        )
    # zarr refuses storage_options for a local path, so only a URL passes them.
    if store.storage_options:
        options["storage_options"] = dict(store.storage_options)

    # A Zarr group records no member order, so the raster's own travels in attrs.
    raster = _recording_order(raster)

    # A Zarr array holds its own fill value, which is what a GDAL reader masks on.
    options["encoding"] = {
        **fill_encoding(raster),
        **dict(options.get("encoding") or {}),
    }
    options.setdefault("consolidated", False)
    raster.to_zarr(
        absolute_location(store),
        mode="w" if overwrite else "w-",
        zarr_format=_STORE_ZARR_FORMAT,
        compute=True,
        **options,
    )
    with open_raster(store) as written:
        return RasterFile.from_xarray(store, written)


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
        if fill is not None:
            # xarray consumes coordinates before backend options, but grid_mapping after.
            encoding = variable.encoding
            declared[str(name)] = {"fill_value": fill}
            if "grid_mapping" in encoding:
                declared[str(name)]["grid_mapping"] = encoding["grid_mapping"]
    return declared
