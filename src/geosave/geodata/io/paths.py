"""Format safe relative destinations and grid node names."""

import pandas as pd
import xarray as xr
from pathlib import PurePosixPath, PureWindowsPath

from odc.geo.geobox import GeoBox

from geosave.geodata.utils.template import fields_named


def _relative_name(name: str) -> str:
    """Keep path format output below its folder on local and remote filesystems."""
    windows = PureWindowsPath(name)
    if (
        not name
        or name.startswith(("/", "\\"))
        or windows.drive
        or "://" in name
        or ".." in windows.parts
        or "\\" in name
        or PurePosixPath(name) == PurePosixPath(".")
    ):
        raise ValueError("a path format must name a relative file below its folder")
    return PurePosixPath(name).as_posix()


def format_path(template: str, raster: xr.Dataset, **fields: object) -> str:
    """Render one safe relative destination for a raster from its own labels.

    Args:
        template: Path template, such as ``"{time:%Y%m%d}/{asset}.tif"``.
        raster: Raster the path names. Its scalar coordinates fill the
            template, and ``{variable}`` names its sole data variable.
        **fields: Names only the writer knows, such as ``asset`` or ``id``.

    Returns:
        POSIX path relative to the destination folder.

    Raises:
        ValueError: The template names an unavailable field, or renders a
            path outside its folder.

    Examples:
        >>> format_path("{time:%Y%m%d}/{asset}.tif", chip, asset="red")
        '20250115/red.tif'
    """
    values = read_scalar_coords(raster)
    if len(raster.gs.variables) == 1:
        values["variable"] = str(raster.gs.variables[0])
    values.update(fields)
    missing = fields_named(template).keys() - values.keys()
    if missing:
        raise ValueError(f"format_paths needs unavailable labels: {sorted(missing)}")
    return _relative_name(template.format_map(values))


def read_scalar_coords(raster: xr.Dataset) -> dict[str, object]:
    """Read a raster's scalar coordinates as plain values, dates as Timestamps.

    The CRS coordinate is no label, so it is left out.

    Examples:
        >>> read_scalar_coords(chip)
        {'time': Timestamp('2025-01-15 00:00:00')}
    """
    crs_coord = raster.odc.crs_coord
    carriers = () if crs_coord is None else (crs_coord.name,)
    values: dict[str, object] = {}
    for name, coord in raster.coords.items():
        if coord.ndim != 0 or name in carriers:
            continue
        if coord.dtype.kind == "M":
            values[str(name)] = pd.Timestamp(coord.values[()])
        else:
            values[str(name)] = coord.item()
    return values


# Short spellings of projected CRS units in grid node names, as EOPF writes `r10m`.
UNIT_ABBREVIATIONS = {"metre": "m", "meter": "m"}


def format_resolution(geobox: GeoBox) -> str:
    """Format a grid's pixel size as a node name, as EOPF writes `r10m`.

    The name is a label, not grid identity: CRS, affine and shape can differ
    at the same pixel size.

    Raises:
        ValueError: The grid carries no CRS, so its pixels have no ground size.

    Examples:
        >>> format_resolution(sentinel_10m)
        'r10m'
        >>> format_resolution(geographic)
        'r0.0001deg'
    """
    if geobox.crs is None:
        raise ValueError(
            "a grid without a CRS has no pixel size on the ground to name it by"
        )
    # odc spells a geographic CRS's units per axis, as degrees_north and degrees_east.
    unit = "deg" if geobox.crs.geographic else geobox.crs.units[0]
    return f"r{abs(geobox.resolution.x):g}{UNIT_ABBREVIATIONS.get(unit, unit)}"
