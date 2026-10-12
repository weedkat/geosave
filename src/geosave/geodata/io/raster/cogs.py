"""How a raster is arranged as Cloud Optimized GeoTIFF files.

A COG holds one `(band, y, x)` array: one instant of one grid, its bands the
data variables. The path a raster is written to is a `str.format` template,
and the fields it names decide the files:

    "out/scene.tif"                  one file, every band
    "out/s2_{band}.tif"              one file per band
    "out/{time:%Y%m%d}.tif"          one file per date
    "out/{time:%Y%m%d}/{band}.tif"   one file per date and band

Three fields are built in, the ones that can split a raster: `time`, each
file's instant, a datetime taking `strftime` directives; `band`, each file's
variable; and `collection`, a stack's layer. Any other field is the caller's, a
value or a callable given each date's raster:

    write(ds, "out/{lat:.3f}.tif", fields={"lat": lambda r: r.gs.anchor.lonlat[1]})
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from string import Formatter
from typing import TYPE_CHECKING, Any, NamedTuple, Unpack, cast

import numpy as np
import pandas as pd
import xarray as xr

from odc.geo.geobox import GeoBox

from geosave import __path__ as _package_paths
from geosave.geodata.conventions import TIME_COORDINATE
from geosave.geodata.warnings import AssumedFillWarning

from .geotiff import COGWriteOptions, write_cog
from ..storage import StorageOptions, absolute_location

if TYPE_CHECKING:
    from os import PathLike

    from geosave.geodata import Dataset
    from geosave.geodata.transform.warp import Resampling


class _Planned(NamedTuple):
    """One COG a raster is about to be written as.

    Args:
        path: Where the file goes, as text.
        raster: Pixels the file holds, without a time dimension.
        time: Instant the file holds, or None where the raster has no time
            dimension.
        collection: Stack layer the raster is, or None for a lone raster.
    """

    path: str
    raster: xr.Dataset
    time: pd.Timestamp | None
    collection: str | None


@dataclass(frozen=True)
class CogFile:
    """One Cloud Optimized GeoTIFF a write produced.

    Args:
        href: Absolute local path or URL of the file.
        bands: Variables the file holds, in band order.
        datetime: Instant the file holds, or None when undated.
        collection: Stack layer the file was written from, or None for a lone
            raster.

    Examples:
        >>> scene.gs.to_cog("out/s_{time:%Y%m%d}.tif")[0]
        CogFile(/abs/out/s_20250601.tif, bands=['red', 'nir'], 2025-06-01, -)
    """

    href: str
    bands: tuple[str, ...]
    datetime: pd.Timestamp | None
    collection: str | None = None

    def __repr__(self) -> str:
        when = "undated" if self.datetime is None else f"{self.datetime:%Y-%m-%d}"
        return (
            f"CogFile({self.href}, bands={list(self.bands)}, {when}, "
            f"{self.collection or '-'})"
        )


BUILT_IN = ("time", "band", "collection")


def fields_named(template: str) -> dict[str, str | None]:
    """Map each field a template names to the format spec it carries."""
    return {name: spec for _, name, spec, _ in Formatter().parse(template) if name}


def split_keywords(
    keywords: Mapping[str, Any],
) -> tuple[COGWriteOptions, dict[str, Any]]:
    """Split an accessor's keywords into COG creation options and fields."""
    known = COGWriteOptions.__annotations__
    options = {key: value for key, value in keywords.items() if key in known}
    fields = {key: value for key, value in keywords.items() if key not in known}
    return cast("COGWriteOptions", options), fields


def layout(
    ds: xr.Dataset,
    template: str,
    *,
    collection: str | None = None,
    fields: Mapping[str, Any] | None = None,
) -> list[_Planned]:
    """Say which COG files a raster is written as, without writing them.

    The template is filled in once per date and band. Bands whose paths come
    out the same share one file, so a template without `{band}` keeps them
    together; dates never share one.

    Args:
        ds: Raster to arrange.
        template: Path ending in `.tif` or `.tiff`, naming any of the fields
            in this module's description.
        collection: The stack layer `ds` is, filling `{collection}`. None
            leaves it unfilled.
        fields: The caller's fields. A callable is given the raster of each
            date and its result filled in.

    Returns:
        One entry per file, in time then variable order.

    Raises:
        ValueError: The path is not a TIFF; several dates would share a file;
            `{time}` is named on a raster without dates; `ds` carries no
            locatable grid or spans a non-spatial axis other than time; or its
            instants are empty, repeated or NaT.
        TypeError: `fields` repeats a built-in field.
        KeyError: The template names a field nothing fills.

    Examples:
        >>> [file.path for file in layout(ds, "out/{time:%Y%m%d}/{band}.tif")]
        ['out/20250601/red.tif', 'out/20250601/nir.tif', ...]
    """
    fields = dict(fields or {})
    for name in BUILT_IN:
        if name in fields:
            raise TypeError(
                f"{name}= repeats the field {{{name}}}, which the writer fills; "
                f"name it something else"
            )
    if PurePosixPath(template).suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("to_cog writes GeoTIFFs; end the path in .tif")

    spatial = ds.odc.spatial_dims
    if spatial is None:
        raise ValueError(
            "the raster carries no locatable grid, so its files cannot be "
            "georeferenced; assign a CRS with odc.geo.xr.assign_crs first"
        )
    # Only pixels have to fit a file; an axis only a coordinate spans writes nothing.
    pixel_dims = {str(dim) for array in ds.data_vars.values() for dim in array.dims}
    extra_dims = sorted(pixel_dims - {*spatial, TIME_COORDINATE})
    if extra_dims:
        raise ValueError(
            f"a COG holds one instant of one grid, but the raster also spans "
            f"{extra_dims}; select those axes away before writing"
        )

    # A scalar time coordinate dates the raster as one step of a time axis does.
    dated = TIME_COORDINATE in ds.coords and ds[TIME_COORDINATE].ndim <= 1
    # Field name to the format spec it carries, for messages naming the caller's own.
    named = fields_named(template)
    if not dated and TIME_COORDINATE in named:
        raise ValueError(
            "this raster has no dates, so {time} cannot be filled; drop it from "
            "the path"
        )

    scenes: list[tuple[pd.Timestamp | None, xr.Dataset]] = [(None, ds)]
    if dated:
        times = pd.DatetimeIndex(np.atleast_1d(ds[TIME_COORDINATE].values))
        if times.hasnans or not times.is_unique or len(times) == 0:
            raise ValueError("scene times must be nonempty, unique and not NaT")
        # Free of NaT, as just checked.
        instants = cast("list[pd.Timestamp]", list(times))
        scenes = (
            [
                (instant, ds.isel({TIME_COORDINATE: index}))
                for index, instant in enumerate(instants)
            ]
            if TIME_COORDINATE in ds.dims
            else [(instants[0], ds)]
        )

    bands: dict[str, list[str]] = {}
    times_of: dict[str, pd.Timestamp | None] = {}
    rasters: dict[str, xr.Dataset] = {}
    for instant, scene in scenes:
        values = {
            name: value(scene) if callable(value) else value
            for name, value in fields.items()
        }
        for band in scene.data_vars:
            built_in: dict[str, Any] = {"band": band}
            if dated:
                built_in["time"] = instant
            if collection is not None:
                built_in["collection"] = collection
            path = template.format(**built_in, **values)
            if path in times_of and times_of[path] != instant:
                raise _collision(path, times_of[path], instant, named, len(scenes))
            times_of[path] = instant
            rasters[path] = scene
            bands.setdefault(path, []).append(str(band))

    return [
        _Planned(path, rasters[path][names], times_of[path], collection)
        for path, names in bands.items()
    ]


def _collision(
    path: str,
    first: pd.Timestamp | None,
    second: pd.Timestamp | None,
    named: Mapping[str, str | None],
    dates: int,
) -> ValueError:
    """Say how to keep two dates from sharing one file."""
    if TIME_COORDINATE in named:
        return ValueError(
            f"{first:%Y-%m-%d} and {second:%Y-%m-%d} would share {path}; make "
            f"{{time:{named[TIME_COORDINATE]}}} more precise, as {{time:%Y%m%d}}"
        )
    return ValueError(
        f"{dates} dates would share {path}; add {{time}} to the path, as "
        f"{{time:%Y%m%d}}, or save with to_zarr or to_netcdf, which hold dates "
        f"in one store"
    )


def read(
    files: Sequence[CogFile],
    *,
    geobox: GeoBox | None = None,
    resampling: Resampling = "nearest",
    **options: Any,
) -> Dataset:
    """Open the records one write produced, or one Item lists, as one raster.

    Files of one instant are merged band by band, after each is placed on
    `geobox` when one is given; instants join along `time` in order.

    Args:
        files: Records of one raster: one layer, any instants and bands.
        geobox: Grid every file is reprojected onto. None keeps the files'
            own grid, which they must share.
        resampling: Kernel `reproject` applies where a file is reprojected.
        **options: Read options forwarded to the GeoTIFF reader. Chunks
            default to an empty mapping, so reading stays lazy.

    Returns:
        Lazy Dataset, variables in band order; a dated file brings its
        instant along `time`, as reading it alone does.

    Raises:
        ValueError: No file is given, or files of one instant repeat a band.

    Examples:
        >>> read(scene.gs.to_cog("out/s_{time:%Y%m%d}_{band}.tif")).gs.variables
        ('red', 'nir')
    """
    # Deferred: transform imports core, which imports io.
    from geosave.geodata.transform.combine import concat_time, merge_bands

    from .gdal import read as read_tiff

    if not files:
        raise ValueError("read needs at least one CogFile")
    instants: dict[pd.Timestamp | None, list[CogFile]] = {}
    for file in files:
        instants.setdefault(file.datetime, []).append(file)

    scenes = []
    unfilled: dict[str, None] = {}
    for when in sorted(instants, key=lambda instant: (instant is None, instant)):
        bands = []
        for file in instants[when]:
            raster = read_tiff(file.href, **{"chunks": {}, **options})
            raster = raster.rename(_names(file, raster))
            if geobox is not None:
                window = _window(raster.gs.geobox, geobox)
                if window is not None:
                    raster = raster.isel(y=window[0], x=window[1])
                else:
                    raster = _reproject(raster, geobox, resampling, unfilled)
            bands.append(raster)
        scenes.append(merge_bands(bands) if len(bands) > 1 else bands[0])
    if unfilled:
        # One warning per read, naming every band, so a read repeated for each
        # chip shows once under Python's default filter.
        warnings.warn(
            f"{list(unfilled)} carry no fill value, so the pixels their warp does "
            f"not cover take NaN, or their dtype's far end where integer; choose "
            f"one with gs.write_nodata(...) before warping",
            AssumedFillWarning,
            skip_file_prefixes=tuple(_package_paths),
        )
    return cast("Dataset", concat_time(scenes) if len(scenes) > 1 else scenes[0])


def _names(file: CogFile, raster: xr.Dataset) -> dict[Any, str]:
    """Map a file's variables to the names its record gives them.

    A record names each band, or names the whole file once; a file holding
    more bands than that one name then takes `name.1` to `name.n`, so no band
    is dropped.
    """
    found = list(raster.data_vars)
    if len(file.bands) == len(found):
        return dict(zip(found, file.bands))
    if len(file.bands) == 1:
        (name,) = file.bands
        return {band: f"{name}.{index}" for index, band in enumerate(found, start=1)}
    raise ValueError(
        f"{file.href} holds {len(found)} bands but its record names "
        f"{len(file.bands)}: {list(file.bands)}"
    )


def _reproject(
    raster: xr.Dataset,
    geobox: GeoBox,
    resampling: Resampling,
    unfilled: dict[str, None],
) -> xr.Dataset:
    """Warp one file's raster, adding the bands it warped without a fill to `unfilled`."""
    # Deferred: transform imports core, which imports io.
    from geosave.geodata.transform.warp import reproject

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warped = reproject(raster, geobox, resampling=resampling)
    for warning in caught:
        if issubclass(warning.category, AssumedFillWarning):
            unfilled.update(
                dict.fromkeys(
                    str(name)
                    for name, band in raster.data_vars.items()
                    if band.gs.nodata is None
                )
            )
        else:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    return warped


def _window(source: object, target: GeoBox) -> tuple[slice, slice] | None:
    """Return the pixels of `source` that are `target`, when no warp is needed.

    A target on the same CRS and resolution, offset by whole pixels and lying
    inside the source, is a slice of it; anything else is warped, a grid placed
    by control points included.
    """
    if not isinstance(source, GeoBox) or source.crs != target.crs:
        return None
    if source.resolution != target.resolution:
        return None
    column, row = ~source.transform * (target.transform.c, target.transform.f)
    if abs(column - round(column)) > 1e-6 or abs(row - round(row)) > 1e-6:
        return None
    top, left = round(row), round(column)
    height, width = target.shape
    if (
        top < 0
        or left < 0
        or top + height > source.shape[0]
        or left + width > source.shape[1]
    ):
        return None
    return slice(top, top + height), slice(left, left + width)


def write(
    ds: xr.Dataset,
    template: str | PathLike[str],
    *,
    fields: Mapping[str, Any] | None = None,
    map_scale: float | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[COGWriteOptions],
) -> tuple[CogFile, ...]:
    """Write a raster as the COG files its path template names.

    Args:
        ds: Raster to write.
        template: Local path or fsspec URL ending in `.tif`, naming any of
            the fields in this module's description; see `layout`.
        fields: Further fields. A callable is given the raster of each date.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace files that already exist.
        storage_options: Options for the filesystem a URL names.
        **options: COG creation options passed to every file.

    Returns:
        One record per file written, in time then variable order.

    Raises:
        FileExistsError: A file exists and `overwrite` is false.
        ValueError: As `layout` raises.
        TypeError: As `layout` raises.

    Examples:
        >>> write(ds, "samples/forest_{time:%Y%m%d}.tif")[0].bands
        ('red', 'nir')
    """
    return write_files(
        layout(ds, str(template), fields=fields),
        map_scale=map_scale,
        overwrite=overwrite,
        storage_options=storage_options,
        **options,
    )


def write_files(
    files: Sequence[_Planned],
    *,
    map_scale: float | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[COGWriteOptions],
) -> tuple[CogFile, ...]:
    """Write the files a `layout` planned.

    Args:
        files: What `layout` returned.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace files that already exist.
        storage_options: Options for the filesystem a URL names.
        **options: COG creation options passed to every file.

    Returns:
        One record per file written, in the order given.

    Raises:
        FileExistsError: A file exists and `overwrite` is false.
    """
    written = []
    for file in files:
        written_path = write_cog(
            cast("Dataset", file.raster),
            file.path,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )
        written.append(
            CogFile(
                href=absolute_location(written_path),
                bands=tuple(str(name) for name in file.raster.data_vars),
                datetime=file.time,
                collection=file.collection,
            )
        )
    return tuple(written)
