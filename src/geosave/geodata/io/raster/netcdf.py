"""Open and write raster Datasets as netCDF files."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import xarray as xr

from geosave.geodata.conventions import to_yx

from .. import storage
from ..storage import Location, StorageOptions, absolute_location
from .file import RasterFile

if TYPE_CHECKING:
    from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder

    from geosave.geodata import Dataset

_FILE_SUFFIXES = (".nc", ".nc4", ".cdf")

type NetCDFEngine = Literal["netcdf4", "h5netcdf"]
type NetCDFChunkSpec = (
    int | tuple[int, ...] | Mapping[str, int] | Literal["auto"] | None
)
type NetCDFFormat = Literal[
    "NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"
]


class NetCDFOpenOptions(TypedDict, total=False):
    """Optional `xarray.open_dataset` behavior when opening netCDF.

    Omitted keys use xarray or the selected engine's defaults. GeoSave passes
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
        backend_kwargs: Extra keywords for the selected NetCDF engine's open
            function. Supported keys depend on that engine.
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

    Omitted keys use xarray or engine defaults. GeoSave owns the destination,
    engine, write mode, and computation arguments.

    Attributes:
        format: NetCDF file variant from NetCDFFormat. None delegates to the
            engine; the selected engine must support the requested variant.
        encoding: Variable names mapped to storage settings such as `dtype`,
            `_FillValue`, `scale_factor`, compression, and chunk sizes.
            If provided, xarray uses this mapping instead of existing encodings.
        unlimited_dims: Dimensions stored as unlimited dimensions. This
            declares file structure; GeoSave still writes a new complete file.
        invalid_netcdf: Allow HDF5 features outside the NetCDF standard with
            `engine="h5netcdf"`; other engines do not support this option.
        auto_complex: Enable netCDF4-python's automatic complex-number storage
            conventions with `engine="netcdf4"`. None uses the backend default.
    """

    format: NetCDFFormat | None
    encoding: Mapping[Hashable, Mapping[str, Any]] | Mapping[str, Any]
    unlimited_dims: Sequence[Hashable]
    invalid_netcdf: bool
    auto_complex: bool | None


def open_raster(
    source: Location,
    *,
    engine: NetCDFEngine = "netcdf4",
    chunks: NetCDFChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[NetCDFOpenOptions],
) -> Dataset:
    """Open a netCDF file as a raster Dataset.

    Args:
        source: Local netCDF path, `file://` URL, or UPath.
        engine: Xarray netCDF backend.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        storage_options: Accepted for a uniform call; a URL is refused.
        **open_options: Supported `xarray.open_dataset` options.

    Returns:
        Raster Dataset as stored.

    Raises:
        ValueError: The file cannot be read, or `source` is a URL.
    """
    if not storage.is_local(source):
        raise ValueError(
            f"{source} is remote, and NetCDF reads only local files; download "
            f"it first, or store the raster as Zarr"
        )
    opened = xr.open_dataset(
        storage.local_path(source),
        engine=engine,
        # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
        decode_coords="all",
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **open_options,
    )
    cube = to_yx(opened)
    if cube is not opened:
        cube.set_close(opened.close)
    cube.encoding["source"] = absolute_location(source)
    return cast("Dataset", cube)


def write(
    raster: xr.Dataset,
    destination: Location,
    *,
    engine: NetCDFEngine = "netcdf4",
    overwrite: bool = False,
    **write_options: Unpack[NetCDFWriteOptions],
) -> RasterFile:
    """Write a raster to a netCDF file.

    Args:
        raster: Raster Dataset.
        destination: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in `.nc`, `.nc4`, or `.cdf`. A URL is
            uploaded once the file is written.
        engine: Xarray netCDF backend.
        overwrite: Replace an existing destination when true.
        **write_options: Supported xarray netCDF write options.

    Returns:
        The written file with the header it states, read back from the file
        before it leaves its staging folder.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        TypeError: `raster` is not a Dataset.
        ValueError: The destination path is invalid.

    Examples:
        >>> write(raster, "scene.nc")
        RasterFile(/abs/scene.nc, ['red', 'nir'], 2025-06-01)
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            f"raster must be an xarray Dataset, got {type(raster).__name__}; "
            f"write a tree with its own gs.to_netcdf"
        )

    stated: list[xr.Dataset] = []

    def save(target: Path) -> None:
        raster.to_netcdf(target, mode="w", engine=engine, compute=True, **write_options)
        # The header is read while the file is still local, coordinates and all.
        with open_raster(target, engine=engine) as opened:
            header = RasterFile.from_xarray(target, opened).header
            stated.append(
                header.assign_coords({k: v.load() for k, v in header.coords.items()})
            )

    written = storage.write(
        destination, save, suffixes=_FILE_SUFFIXES, overwrite=overwrite
    )
    return RasterFile.from_xarray(written, stated[0])
