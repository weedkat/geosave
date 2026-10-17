"""Write a Dataset as a COG or a plain GeoTIFF, one band per variable.

`write_header` puts each attr where GDAL reads it: a band property where GDAL
keeps one, a metadata tag otherwise. rioxarray is left the pixels and the grid.
"""

from __future__ import annotations

from pathlib import Path

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
from odc.geo.geobox import GeoBox
from rasterio.enums import ColorInterp

# rasterio ships this as a compiled module, which no type checker can resolve.
from rasterio.shutil import copy  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]

import orjson
import xarray as xr
from xarray.conventions import encode_cf_variable
from xarray.core.utils import equivalent

import geosave.geodata.attrs as attrs

from geosave.geodata.conventions import TIME_COORDINATE
from geosave.geodata.utils.color import parse_color

from .. import storage
from ..storage import Location
from . import gdal
from .file import RasterFile
from geosave.geodata.attrs.groups.tiff import GeoTIFFTags

if TYPE_CHECKING:
    from rasterio.io import DatasetWriter
    from collections.abc import Mapping, Sequence

    from geosave.geodata import Dataset

type CreationOptionValue = str | int | float | bool

_FILE_SUFFIXES = (".tif", ".tiff")


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


def write_cog(
    ds: Dataset,
    path: Location,
    *,
    map_scale: float | None = None,
    overwrite: bool = False,
    **options: Unpack[COGWriteOptions],
) -> RasterFile:
    """Write one Dataset as a Cloud Optimized GeoTIFF, a band per variable.

    Each variable names itself in its band's metadata. A scalar `time`
    coordinate becomes whole-second ``TIFFTAG_DATETIME`` plus precise
    ``GEOSAVE_DATETIME`` metadata.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid, optionally with one
            singleton time axis or a scalar time coordinate.
        path: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in ``.tif`` or ``.tiff``.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace an existing file when true.
        **options: COG creation options.

    Returns:
        The written file with the header it states, read back from the file
        before it leaves its staging folder.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube holds several time steps,
            or the variables declare more than one fill value.

    Examples:
        >>> write_cog(ds, "scene.tif", compress="DEFLATE")
        RasterFile(/abs/scene.tif, ['red', 'nir'], 2025-06-01)
    """
    return _write(
        ds,
        path,
        "COG",
        map_scale=map_scale,
        overwrite=overwrite,
        options=options,
    )


def write_gtiff(
    ds: Dataset,
    path: Location,
    *,
    map_scale: float | None = None,
    overwrite: bool = False,
    **options: Unpack[GTiffWriteOptions],
) -> RasterFile:
    """Write one Dataset as a plain GeoTIFF, a band per variable.

    Reach for `write_cog` unless a consumer needs a striped or otherwise
    non-COG file. Coordinates map to tags exactly as `write_cog` maps them.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid, optionally with one
            singleton time axis or a scalar time coordinate.
        path: Output path, fsspec URL, or UPath carrying its
            `storage_options`, ending in ``.tif`` or ``.tiff``.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace an existing file when true.
        **options: GTiff creation options.

    Returns:
        The written file with the header it states, read back from the file
        before it leaves its staging folder.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube holds several time steps,
            or the variables declare more than one fill value.

    Examples:
        >>> write_gtiff(ds, "scene.tif", tiled=True, blockxsize=512)
        RasterFile(/abs/scene.tif, ['red', 'nir'], 2025-06-01)
    """
    return _write(
        ds,
        path,
        "GTiff",
        map_scale=map_scale,
        overwrite=overwrite,
        options=options,
    )


def _write(
    ds: Dataset,
    path: Location,
    driver: Literal["COG", "GTiff"],
    *,
    map_scale: float | None,
    overwrite: bool,
    options: Mapping[str, Any],
) -> RasterFile:
    """Encode one cube through the GDAL driver `driver` names.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid.
        path: Output path, fsspec URL, or UPath ending in ``.tif`` or ``.tiff``.
        driver: GDAL driver creating the file.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace an existing file when true.
        options: Creation options the driver supports.

    Returns:
        The written file with the header it states.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix or the `time` axis is invalid, or the variables
            declare more than one fill value.
    """
    # A GCP grid writes back as no geotransform at all, leaving the file unplaced.
    grid = ds.odc.geobox
    if grid is not None and not isinstance(grid, GeoBox):
        raise ValueError(
            f"this raster is placed by {type(grid).__name__}, which a GeoTIFF "
            f"records as ground control points GeoSave does not write; reproject "
            f"it onto a regular grid with warp.reproject first"
        )

    ds = _repack(ds)

    cube = ds
    # One file holds one instant, however the axis spells it.
    if TIME_COORDINATE in cube.dims:
        if cube.sizes[TIME_COORDINATE] > 1:
            raise ValueError(
                f"one GeoTIFF holds one instant, but this cube spans "
                f"{cube.sizes[TIME_COORDINATE]}; select one with .isel(time=0) "
                f"or resample the time axis away first"
            )
        cube = cube.squeeze(TIME_COORDINATE)

    tags = GeoTIFFTags.from_xarray(cube, map_scale=map_scale)
    if TIME_COORDINATE in cube.coords:
        cube = cube.drop_vars(TIME_COORDINATE)

    names = [str(name) for name in cube.data_vars]

    # GDAL writes one fill value for the whole file, so the bands must share one.
    fill_values = [cube[name].gs.nodata for name in names]
    if not all(equivalent(fill_value, fill_values[0]) for fill_value in fill_values):
        raise ValueError(
            f"one GeoTIFF holds one fill value, but these bands declare "
            f"{dict(zip(names, fill_values, strict=True))}; write one across "
            f"them with raster.gs.write_nodata(...) first"
        )

    root = {**cube.attrs, **tags.model_dump(mode="json", exclude_none=True)}
    # A band names itself in its own metadata; GDAL owns the interpretation.
    bands = [_named(cube[name], name) for name in names]

    # rioxarray turns any attr it is handed into a tag, so it is handed none.
    cube = cube.copy(deep=False)
    cube.attrs = {}
    for name in names:
        cube[name].attrs = {}

    stated: list[xr.Dataset] = []

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

        # GDAL infers some of what a file states, so the header is read back
        # from the file itself while it is still local.
        stated.append(gdal.open_raster(target))

    written = storage.write(path, save, suffixes=_FILE_SUFFIXES, overwrite=overwrite)
    return RasterFile.from_xarray(written, stated[0])


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
            f"describe them; pass one variable per band, in band order"
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


def _named(band: xr.DataArray, name: str) -> xr.DataArray:
    """Return the band stating its variable name, its interpretation kept."""
    return attrs.stamp_attrs(
        band, attrs.GDALBand.replace(band.attrs, variable_name=name)
    )


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
