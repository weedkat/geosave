"""Read the files a writer produced, or a catalog lists, back as one raster."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave import __path__ as _package_paths
from geosave.geodata.warnings import AssumedFillWarning

from .file import RasterFile, Scene
from ..storage import Location

if TYPE_CHECKING:
    from affine import Affine

    from geosave.geodata import Dataset
    from geosave.geodata.transform.warp import Resampling


def open_raster(
    files: Iterable[Scene | RasterFile | Location],
    *,
    geobox: GeoBox | None = None,
    resampling: Resampling | Mapping[str, Resampling] = "nearest",
    **options: Any,
) -> Dataset:
    """Open the scenes one write produced, or one catalog lists, as one raster.

    Files of one instant are merged band by band, after each is placed on
    `geobox` when one is given; instants join along `time` in order.

    Args:
        files: Scenes of one collection, any instants and bands, as `to_cog`
            returns them or a catalog's Items read; their files; or paths,
            each read as its own variables and date. A file's header names its
            variables, dates it where the file states no time, and fills in
            the attrs the file does not state.
        geobox: Grid every file is reprojected onto. None keeps the files'
            own grid, which they must share.
        resampling: Kernel or per-variable mapping passed to `reproject`;
            `"*"` supplies the mapping's fallback.
        **options: Read options forwarded to the GeoTIFF reader. Chunks
            default to an empty mapping, so reading stays lazy.

    Returns:
        Lazy Dataset, variables in band order. Files of one instant keep it
        as a scalar `time`, as reading one alone does; several instants
        span a `time` axis.

    Raises:
        ValueError: No file is given, or files of one instant repeat a band.

    Examples:
        >>> open_raster(scene.gs.to_cog("out/s_{time:%Y%m%d}_{band}.tif")).gs.variables
        ('red', 'nir')
    """
    # Deferred: openers imports this module to read several files as one.
    from ..openers import open_raster as open_file

    flat = [
        entry
        for each in files
        for entry in (each.written.values() if isinstance(each, Scene) else [each])
    ]
    if not flat:
        raise ValueError("read needs at least one file")
    rasters = []
    for entry in flat:
        # A saved file is read by the format its suffix names.
        raster = open_file(
            entry.path if isinstance(entry, RasterFile) else entry,
            **{"chunks": {}, **options},
        )
        if isinstance(entry, RasterFile):
            raster = _apply_header(raster, entry)
        rasters.append(raster)
    return from_rasters(rasters, geobox=geobox, resampling=resampling)


def from_rasters(
    rasters: Iterable[xr.Dataset],
    *,
    geobox: GeoBox | None = None,
    resampling: Resampling | Mapping[str, Resampling] = "nearest",
) -> Dataset:
    """Assemble opened rasters using lazy windows or explicit reprojection."""
    # Deferred: transform imports core, which imports io.
    from geosave.geodata.transform.combine import merge
    from geosave.geodata.transform.temporal import concat

    instants: dict[pd.Timestamp | None, list[xr.Dataset]] = {}
    for raster in rasters:
        times = raster.gs.times
        when = None if times is None else times[0]
        instants.setdefault(when, []).append(raster)

    if not instants:
        raise ValueError("read needs at least one raster")
    scenes = []
    unfilled: dict[str, None] = {}
    for when in sorted(instants, key=lambda instant: (instant is None, instant)):
        bands = []
        for raster in instants[when]:
            if geobox is not None:
                window = _window(raster.gs.geobox, geobox)
                if window is not None:
                    raster = raster.isel(y=window[0], x=window[1])
                else:
                    raster = _reproject(raster, geobox, resampling, unfilled)
                # Native and warped CRS coordinates can use different WKT
                # spellings. All files now share this explicit target CRS.
                raster = raster.odc.assign_crs(geobox.crs).assign_coords(
                    spatial_ref=xr_coords(geobox)["spatial_ref"]
                )
            bands.append(raster)
        scenes.append(merge(bands) if len(bands) > 1 else bands[0])
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
    return cast("Dataset", concat(scenes) if len(scenes) > 1 else scenes[0])


def _apply_header(raster: xr.Dataset, file: RasterFile) -> xr.Dataset:
    """Name, date and describe a file's raster as its header states.

    A header names each band, or names the whole file once; a file holding
    more bands than that one name then takes `name.1` to `name.n`, so no band
    is dropped. A catalog's header fills in what the file does not state.
    """
    header = file.header
    raster = rename_bands(raster, list(header.data_vars), source=str(file.path))

    raster = raster.assign(
        {
            name: raster[name].assign_attrs(
                {**header[name].attrs, **raster[name].attrs}
            )
            for name in raster.data_vars
            if name in header
        }
    )
    if raster.gs.times is None and "time" in header.coords:
        raster = raster.assign_coords(time=header.time)
    return raster


def _reproject(
    raster: xr.Dataset,
    geobox: GeoBox,
    resampling: Resampling | Mapping[str, Resampling],
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
    offset = cast("Affine", ~source.transform * target.transform)
    column, row = offset.c, offset.f
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


def rename_bands(
    raster: xr.Dataset,
    names: Sequence[Hashable],
    *,
    source: str,
    fallback: str | None = None,
) -> xr.Dataset:
    """Name file bands by position, expanding a lone name for a multiband file.

    Args:
        raster: Opened file with variables in stored band order.
        names: Names stated by the file header or catalog.
        source: File location for a band-count error.
        fallback: Prefix for unnamed bands; defaults to the lone stated name.

    Returns:
        Dataset with named variables and unchanged pixels.

    Raises:
        ValueError: The file and stated band counts disagree.
    """
    found = list(raster.data_vars)
    stated = list(names)
    if len(stated) == 1 and len(found) > 1:
        prefix = stated[0] if fallback is None else fallback
        stated = [f"{prefix}.{index}" for index in range(1, len(found) + 1)]
    if len(found) != len(stated):
        raise ValueError(
            f"{source} holds {len(found)} bands but its metadata names {len(stated)}: {stated}"
        )

    return raster.rename(dict(zip(found, stated, strict=True)))
