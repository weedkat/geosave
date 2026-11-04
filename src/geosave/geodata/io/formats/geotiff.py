"""Write a Dataset as a COG or a plain GeoTIFF, one band per variable.

`write_header` puts each attr where GDAL reads it: a band property where GDAL
keeps one, a metadata tag otherwise. rioxarray is left the pixels and the grid.
"""

from __future__ import annotations

from geosave.geodata.io import rasterio as raster_reader
from geosave.geodata.io.rasterio import RasterioOpenOptions

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import orjson
import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.enums import ColorInterp

# rasterio ships this as a compiled module, which no type checker can resolve.
from rasterio.shutil import (
    copy,  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]
)
from upath import UPath
from xarray.conventions import encode_cf_variable
from xarray.core.utils import equivalent

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import TIME_COORDINATE
from geosave.geodata.io import storage
from geosave.geodata.io.storage import Location, StorageOptions
from geosave.geodata.utils.color import parse_color
from geosave.geodata.core.raster.time import read_bounds

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from dask.delayed import Delayed
    from rasterio.io import DatasetWriter

    from geosave.geodata import Dataset

type CreationOptionValue = str | int | float | bool

FILE_SUFFIXES = (".tif", ".tiff")


def _repack(ds: Dataset) -> Dataset:
    """Apply each variable's CF packing to its values, moving it into attrs.

    rioxarray's Dataset writer declares a band's scale and offset without
    applying them, so decoded pixels would be stored physical and then read
    back scaled a second time. Packing them here keeps values and tags agreed.

    Args:
        ds: Cube whose variables may carry CF packing in their encoding, as
            every raster read with ``mask_and_scale=True`` does.

    Returns:
        The same cube with stored values, each variable's packing now in its
        attrs where the header reads it. A variable carrying none is untouched.
    """
    packed = ds.copy()
    for name, variable in ds.data_vars.items():
        if not attrs.Packing.is_stated_in(variable.encoding):
            continue
        encoded = encode_cf_variable(variable.variable)
        # The CF encoder names the grid mapping, which rioxarray writes itself.
        packed_attrs = {
            key: value for key, value in encoded.attrs.items() if key != "grid_mapping"
        }
        stored = xr.DataArray(
            encoded.data, dims=variable.dims, coords=variable.coords, name=name
        )
        stored.attrs = {**variable.attrs, **packed_attrs}
        stored.encoding = {
            key: value
            for key, value in variable.encoding.items()
            if key not in (*attrs.Packing.KEYS, *attrs.Nodata.KEYS)
        }
        packed[name] = stored
    return packed


class GeoTIFFWriteOptions(TypedDict, total=False):
    """Declared creation options passed to GDAL's TIFF drivers.

    Omitted keys use the selected driver's defaults. Driver and GDAL build
    determine which codecs and settings are available.

    Attributes:
        compress: Compression codec name, such as `"DEFLATE"` or `"LZW"`.
        predictor: Compression predictor: 1 disables prediction, 2 differences
            adjacent samples, and 3 predicts floating-point samples.
        zlevel: GTiff's DEFLATE compression effort. The COG driver calls its
            corresponding setting `LEVEL`; GeoSave does not translate this key.
        num_threads: GDAL worker-thread count or `"ALL_CPUS"`.
        bigtiff: `"YES"` forces BigTIFF, `"NO"` forces classic TIFF;
            `"IF_NEEDED"` and `"IF_SAFER"` delegate size heuristics to GDAL.
        interleave: `"PIXEL"` stores bands together per pixel;
            `"BAND"` stores band planes separately.
        sparse_ok: Omit all-zero or all-nodata blocks from the TIFF allocation.
        creation_options: Mapping of additional driver settings. Currently
            forwarded as one keyword rather than unpacked; its entries do not
            configure the driver. Use the explicit fields for supported controls.
    """

    compress: str
    predictor: int
    zlevel: int
    num_threads: int | Literal["ALL_CPUS"]
    bigtiff: Literal["YES", "NO", "IF_NEEDED", "IF_SAFER"]
    interleave: Literal["PIXEL", "BAND"]
    sparse_ok: bool
    creation_options: dict[str, CreationOptionValue]


class COGWriteOptions(GeoTIFFWriteOptions, total=False):
    """Creation options passed to GDAL's COG driver.

    Omitted keys use GDAL defaults. This includes the inherited TIFF options;
    the writer does not change pixel resolution when creating overviews.

    Attributes:
        compress: Compression codec name, such as `"DEFLATE"` or `"LZW"`.
        predictor: Compression predictor: 1 disables prediction, 2 differences
            adjacent samples, and 3 predicts floating-point samples.
        zlevel: Inherited GTiff effort option, passed unchanged. COG names its
            compression-effort setting `LEVEL` rather than `ZLEVEL`.
        num_threads: GDAL worker-thread count or `"ALL_CPUS"`.
        bigtiff: Force or prohibit BigTIFF with `"YES"` or `"NO"`, or ask
            GDAL's `"IF_NEEDED"`/`"IF_SAFER"` size heuristics.
        interleave: `"PIXEL"` stores bands together per pixel;
            `"BAND"` stores band planes separately.
        sparse_ok: Omit all-zero or all-nodata blocks from the TIFF allocation.
        creation_options: Mapping currently forwarded without unpacking its
            entries; use explicit option fields instead.
        blocksize: Square tile side in pixels; GDAL requires a multiple of 16.
        overview_resampling: GDAL kernel used to build reduced-resolution
            overviews, such as `"NEAREST"`, `"AVERAGE"`, or `"BILINEAR"`.
        overview_count: Number of overview levels to include; 0 omits them.
    """

    blocksize: int
    overview_resampling: str
    overview_count: int


class GTiffWriteOptions(GeoTIFFWriteOptions, total=False):
    """Creation options passed to GDAL's GTiff driver.

    Omitted keys use GDAL defaults. These include the inherited TIFF options
    and GTiff-specific strip, tile, and sample-layout settings.

    Attributes:
        compress: Compression codec name, such as `"DEFLATE"` or `"LZW"`.
        predictor: Compression predictor: 1 disables prediction, 2 differences
            adjacent samples, and 3 predicts floating-point samples.
        zlevel: DEFLATE compression effort, validated against GDAL's supported
            level range for its compression library.
        num_threads: GDAL worker-thread count or `"ALL_CPUS"`.
        bigtiff: Force or prohibit BigTIFF with `"YES"` or `"NO"`, or ask
            GDAL's `"IF_NEEDED"`/`"IF_SAFER"` size heuristics.
        interleave: `"PIXEL"` stores bands together per pixel;
            `"BAND"` stores band planes separately.
        sparse_ok: Omit all-zero or all-nodata blocks from the TIFF allocation.
        creation_options: Mapping currently forwarded without unpacking its
            entries; use explicit option fields instead.
        tiled: Write tiled blocks when True; False uses strips.
        blockxsize: Tile width in pixels when tiled; must be a multiple of 16.
        blockysize: Block height in pixels; tiled blocks need a multiple of 16.
        copy_src_overviews: Ask GDAL to copy existing source overviews. GeoSave
            creates this file from xarray pixels and supplies no source overviews.
        photometric: GDAL color interpretation, such as `"RGB"` or `"MINISBLACK"`.
        nbits: Stored bits per sample, subject to the pixel dtype and GDAL's
            supported bit depths; values must fit that storage representation.
    """

    tiled: bool
    blockxsize: int
    blockysize: int
    copy_src_overviews: bool
    photometric: str
    nbits: int


def validate(ds: xr.Dataset, path: Location) -> None:
    """Check that one raster selection can be written as a GeoTIFF.

    Args:
        ds: Raster whose pixel dimensions, time and fill values are checked.
        path: Destination ending in `.tif` or `.tiff`.

    Raises:
        ValueError: The suffix, grid, dimensions, dates or band fill values
            cannot be represented by one GeoTIFF. No pixels are computed.
    """
    if storage.location(path).suffix.lower() not in FILE_SUFFIXES:
        raise ValueError(f"{path} is not a GeoTIFF destination; use .tif or .tiff")
    spatial = ds.odc.spatial_dims
    if spatial is None:
        raise ValueError("a GeoTIFF needs a locatable grid; assign a CRS first")
    grid = ds.odc.geobox
    if grid is not None and not isinstance(grid, GeoBox):
        raise ValueError(
            f"this raster is placed by {type(grid).__name__}, which GeoSave "
            "does not write as GeoTIFF ground control points; reproject first"
        )
    dimensions = set()
    for variable in ds.data_vars.values():
        dimensions.update(variable.dims)
    extra = dimensions - {*spatial, TIME_COORDINATE}
    if extra:
        raise ValueError(
            f"a COG holds one instant of one grid, but the raster also spans {sorted(extra)}"
        )
    count = ds.sizes.get(TIME_COORDINATE, 1)
    if count != 1:
        raise ValueError(
            f"one GeoTIFF holds one instant; {count} dates would share {path}; "
            "add {time} to path_format or select one observation"
        )
    cube = ds.squeeze(TIME_COORDINATE) if TIME_COORDINATE in ds.dims else ds
    time = cube.coords.get(TIME_COORDINATE)
    if time is not None and np.issubdtype(time.dtype, np.datetime64):
        if np.isnat(time.values).any():
            raise ValueError("a GeoTIFF observation time cannot be NaT")
    _time_coverage(cube)
    fill_values = {}
    for name, variable in ds.data_vars.items():
        value = variable.gs.nodata
        if attrs.Packing.is_stated_in(variable.encoding):
            value = variable.encoding.get("_FillValue", value)
        fill_values[name] = value
    if not fill_values:
        raise ValueError("a GeoTIFF needs at least one data variable")
    first = next(iter(fill_values.values()))
    if not all(equivalent(value, first) for value in fill_values.values()):
        raise ValueError(
            f"one GeoTIFF holds one fill value, but these bands declare {fill_values}; "
            "set a shared nodata value or split bands with {band} in path_format"
        )


def write_files(
    files: Mapping[UPath, xr.Dataset],
    writer: Callable[..., UPath] | None = None,
    *,
    compute: bool = True,
    overwrite: bool = False,
    **options: Any,
) -> list[UPath] | Delayed:
    """Write several GeoTIFFs, each destination and raster checked before any is written.

    Args:
        files: Destinations, each ending in `.tif` or `.tiff`, mapped to the
            raster one GeoTIFF holds there.
        writer: `write_cog`, the default, or `write_geotiff`.
        compute: False returns one native `dask.Delayed`; computing it checks
            the destinations again and writes every file in turn.
        overwrite: Replace existing destinations.
        **options: Creation options for the writer.

    Returns:
        Written paths in destination order, or one delayed result.

    Raises:
        ValueError: Two destinations resolve to one file, or a raster cannot be
            one GeoTIFF: more than one instant, band fill values that differ,
            no grid.
        FileExistsError: A destination exists and overwrite is false.

    Examples:
        >>> write_files({root / "red.tif": scene[["red"]], root / "nir.tif": scene[["nir"]]})
        [UPath('/abs/red.tif'), UPath('/abs/nir.tif')]
    """
    writer = write_cog if writer is None else writer
    _check_files(files, overwrite)
    if compute:
        return [
            writer(raster, path, overwrite=overwrite, **options)
            for path, raster in files.items()
        ]
    from dask.delayed import delayed

    # One task writes every file in turn, checked again when it runs.
    return delayed(partial(_write_checked, files, writer, overwrite, options))()


def _check_files(files: Mapping[UPath, xr.Dataset], overwrite: bool) -> None:
    """Refuse a batch where two destinations collide, one exists, or a raster cannot be a GeoTIFF."""
    addresses = set()
    for path, raster in files.items():
        address = storage.absolute_location(path)
        if address in addresses:
            raise ValueError(f"path selections collide at {path}")
        addresses.add(address)
        if not overwrite and path.exists():
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        validate(raster, path)


def _write_checked(
    files: Mapping[UPath, xr.Dataset],
    writer: Callable[..., UPath],
    overwrite: bool,
    options: dict[str, Any],
) -> list[UPath]:
    """Check a batch again, since a destination may have appeared since it was planned."""
    _check_files(files, overwrite)
    return [
        writer(raster, path, overwrite=overwrite, **options)
        for path, raster in files.items()
    ]


def write_cog(
    ds: Dataset,
    path: Location,
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[COGWriteOptions],
) -> UPath:
    """Write one Dataset as a Cloud Optimized GeoTIFF, a band per variable.

    Each variable names itself in its band's metadata. A scalar `time`
    coordinate becomes ACDD ``time_coverage_start`` and ``time_coverage_end``
    attrs at full precision; a composite's bounds fill them instead.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid, optionally with one
            singleton time axis or a scalar time coordinate.
        path: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in ``.tif`` or ``.tiff``.
        overwrite: Replace an existing file when true.
        storage_options: Filesystem credentials and connection options, merged
            with any options already carried by the destination UPath.
        **options: COG creation options.

    Returns:
        Path of the artifact actually written.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube holds several time steps,
            or the variables declare more than one fill value.

    Examples:
        >>> write_cog(ds, "scene.tif", compress="DEFLATE")
        UPath('/abs/scene.tif')
    """
    return _write(
        ds,
        storage.location(path, storage_options),
        "COG",
        overwrite=overwrite,
        options=options,
    )


def write_geotiff(
    ds: Dataset,
    path: Location,
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[GTiffWriteOptions],
) -> UPath:
    """Write one Dataset as a plain GeoTIFF, a band per variable.

    Reach for `write_cog` unless a consumer needs a striped or otherwise
    non-COG file. Coordinates map to tags exactly as `write_cog` maps them.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid, optionally with one
            singleton time axis or a scalar time coordinate.
        path: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in ``.tif`` or ``.tiff``.
        overwrite: Replace an existing file when true.
        storage_options: Filesystem credentials and connection options, merged
            with any options already carried by the destination UPath.
        **options: GTiff creation options.

    Returns:
        Path of the artifact actually written.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube holds several time steps,
            or the variables declare more than one fill value.

    Examples:
        >>> write_geotiff(ds, "scene.tif", tiled=True, blockxsize=512)
        UPath('/abs/scene.tif')
    """
    return _write(
        ds,
        storage.location(path, storage_options),
        "GTiff",
        overwrite=overwrite,
        options=options,
    )


def _write(
    ds: Dataset,
    path: Location,
    driver: Literal["COG", "GTiff"],
    *,
    overwrite: bool,
    options: Mapping[str, Any],
) -> UPath:
    """Encode one cube through the GDAL driver `driver` names.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid.
        path: Output path, fsspec URL, or UPath ending in ``.tif`` or ``.tiff``.
        driver: GDAL driver creating the file.
        overwrite: Replace an existing file when true.
        options: Creation options the driver supports.

    Returns:
        Path of the artifact actually written.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix or the `time` axis is invalid, or the variables
            declare more than one fill value.
    """
    validate(ds, path)

    ds = _repack(ds)

    cube = ds
    # One file holds one instant, however the axis spells it.
    if TIME_COORDINATE in cube.dims:
        cube = cube.squeeze(TIME_COORDINATE)

    coverage = _time_coverage(cube)
    if TIME_COORDINATE in cube.coords:
        cube = cube.drop_vars(TIME_COORDINATE)

    names = [str(name) for name in cube.data_vars]

    root = {**cube.attrs, **coverage}
    # A band names itself in its own metadata; GDAL owns the interpretation.
    bands = [_named(cube[name], name) for name in names]

    # rioxarray turns any attr it is handed into a tag, so it is handed none.
    cube = cube.copy(deep=False)
    cube.attrs = {}
    for name in names:
        cube[name].attrs = {}

    def save(target: Path) -> None:
        with TemporaryDirectory(dir=target.parent) as workspace:
            # A COG is copied from a source raster, which its own options create.
            if driver == "COG":
                source = Path(workspace) / target.name
                source_options: Mapping[str, Any] = {"tiled": True}
            else:
                source = target
                source_options = options

            cube.rio.to_raster(
                source,
                driver="GTiff",
                # rioxarray computes a chunked cube whole unless told to walk its windows.
                windowed=bool(cube.chunks),
                **source_options,
            )
            # GDAL honours metadata on an open dataset, not as creation options.
            with rasterio.open(source, "r+") as dst:
                write_header(dst, root, bands)

            if driver == "COG":
                copy(source, target, driver="COG", **options)

    return storage.write_file(
        path, save, suffixes=FILE_SUFFIXES, overwrite=overwrite
    ).absolute()


def write_header(
    dst: DatasetWriter, root: Mapping[str, Any], bands: Sequence[xr.DataArray]
) -> None:
    """Write every attr onto an open GDAL raster, in the slot GDAL reads it from.

    GDAL keeps the description, unit, scale, offset, fill value, interpretation
    and palette as band properties of its own, where `gdalinfo` and QGIS look;
    every other attr is a plain metadata item, and none is stated twice.

    Args:
        dst: Raster open for update.
        root: Attrs of the whole file.
        bands: One variable per band of `dst`, in band order.

    Raises:
        ValueError: `bands` does not hold one variable per band of `dst`.

    Examples:
        >>> with rasterio.open(path, "r+") as dst:
        ...     write_header(dst, ds.attrs, list(ds.data_vars.values()))
    """
    if len(bands) != dst.count:
        raise ValueError(
            f"{dst.name} holds {dst.count} bands but {len(bands)} variables "
            f"name them; pass one variable per band, in band order"
        )

    dst.update_tags(**_as_text(root))

    owned = {*attrs.Packing.KEYS, *attrs.Nodata.KEYS, "color_map", "colorinterp"}
    scales: list[float] = []
    offsets: list[float] = []
    interpretations: list[ColorInterp] = []
    for index, band in enumerate(bands, start=1):
        packing = band.gs.packing
        legend = band.gs.legend
        identity = attrs.GDALBand.from_attrs(band.attrs)
        tags = {key: value for key, value in band.attrs.items() if key not in owned}

        # Each of these leaves the tags for a band property GDAL keeps it in.
        description = tags.pop("long_name", None)
        unit = tags.pop("units", None)
        dst.update_tags(index, **_as_text(tags))

        if description is not None:
            dst.set_band_description(index, description)
        if unit is not None:
            dst.set_band_unit(index, unit)

        scales.append(1.0 if packing is None else packing.scale_factor)
        offsets.append(0.0 if packing is None else packing.add_offset)
        palette = None if legend is None else legend.color_map
        interpretation = None if identity is None else identity.colorinterp
        # A colour table makes a band a palette, whatever it was read as.
        interpretations.append(
            ColorInterp.palette
            if palette is not None
            else ColorInterp[interpretation or ColorInterp.undefined.name]
        )

        if palette is not None:
            dst.write_colormap(
                index,
                {
                    int(value): (*parse_color(colour), 255)
                    for value, colour in palette.items()
                },
            )

    if any(scale != 1.0 for scale in scales):
        dst.scales = scales
    if any(offset != 0.0 for offset in offsets):
        dst.offsets = offsets
    # Assigning none of them would clear what GDAL inferred for itself.
    if any(band is not ColorInterp.undefined for band in interpretations):
        dst.colorinterp = interpretations

    shared = bands[0].gs.nodata if bands else None
    if shared is not None:
        dst.nodata = shared


def _time_coverage(cube: xr.Dataset) -> dict[str, str]:
    """State the period a file covers as ACDD `time_coverage` attrs.

    A TIFF has no coordinates, so the scalar `time` coordinate and a composite's
    `start_datetime` and `end_datetime` are carried as attrs and restored on
    read. An instant covers itself: start equals end.

    Raises:
        ValueError: `time` is not a datetime.
    """
    time = cube.coords.get(TIME_COORDINATE)
    if time is None or time.ndim != 0:
        return {}
    instant = time.values
    if not isinstance(instant, np.datetime64):
        raise ValueError(
            f"the time coordinate must be a datetime to be stored in a GeoTIFF, "
            f"got {type(instant).__name__}; convert it with pd.to_datetime first"
        )
    if np.isnat(instant):
        return {}

    bounds = read_bounds(cube.coords)
    if bounds is None:
        start = end = instant
    else:
        starts, ends = bounds
        start, end = starts.min(), ends.max()
    coverage = attrs.ACDD(
        time_coverage_start=np.datetime_as_string(start, unit="auto"),
        time_coverage_end=np.datetime_as_string(end, unit="auto"),
    )
    return coverage.to_attrs()


def _named(band: xr.DataArray, name: str) -> xr.DataArray:
    """Return the band stating its variable name, its interpretation kept."""
    stated = attrs.GDALBand.from_attrs(band.attrs) or attrs.GDALBand()
    return attrs.stamp_attrs(band, stated.model_copy(update={"variable_name": name}))


def _as_text(attrs: Mapping[str, Any]) -> dict[str, str]:
    """Spell attrs as the text a GDAL metadata tag holds.

    Args:
        attrs: Flat attrs to write as tags.

    Returns:
        {
            "<attr key>": its value as text,
        }
        A collection is spelled as JSON, which reads back as itself.
    """
    return {
        key: value if isinstance(value, str) else orjson.dumps(value).decode()
        for key, value in attrs.items()
    }


def open_geotiff(
    source: Location,
    *,
    chunks: Any = {},
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[RasterioOpenOptions],
) -> xr.Dataset:
    """Open GeoTIFF bands with their grid and metadata, retaining lazy pixels.

    Args:
        source: GeoTIFF path or URL.
        chunks: Dask chunks; {} uses native tiles, None uses backend indexing.
        mask_and_scale: Decode packing and missing values when True.
        storage_options: Filesystem credentials and connection settings.
        **options: Rasterio read options.

    Returns:
        Dataset with one variable per band.
    """
    return raster_reader.open_raster(
        source,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        storage_options=storage_options,
        **options,
    )
