"""CF codecs: the structure GeoSave stores in netCDF and Zarr files.

start/end coordinates  <-> CF time bounds (netCDF only; Zarr keeps them)
spatial dims and CRS   <-> `axis`, `standard_name`, `grid_mapping`
"""

from __future__ import annotations

from typing import Literal

import cf_xarray  # noqa: F401 registers .cf
import odc.geo.xr  # noqa: F401 registers .odc
import xarray as xr

from geosave.geodata.io.raster.codecs import Codec

# A raster's spatial dimensions, in the order `gs.spatial_dims` names them.
_AXES: tuple[Literal["Y"], Literal["X"]] = ("Y", "X")
# A boundary variable's vertex dimension, named as cf-xarray's add_bounds does.
_BOUNDS_DIMENSION = "bounds"
# Encoding for written time when it states none: exact for nanosecond labels.
_TIME_UNITS = "nanoseconds since 1970-01-01"
_TIME_CALENDAR = "proleptic_gregorian"


def encode_time_bounds(raster: xr.Dataset) -> xr.Dataset:
    """Replace start/end coordinates with a CF boundary variable before writing.

    The boundary variable is `{time}_bounds(time, bounds)`, named by `time`'s
    `bounds` attr. Both share `time`'s units, calendar and dtype (CF §7.1),
    `time`'s own encoding when it has one.

    Args:
        raster: Dataset about to be written.

    Returns:
        Dataset with `{time}_bounds` in place of `start_datetime` and
        `end_datetime`, or `raster` unchanged when it states no cells.

    Raises:
        ValueError: `time` already names another boundary variable, so the
            raster states its cells twice.
    """
    from geosave.geodata.core.conventions import END_DATETIME, START_DATETIME

    time = raster.gs.time_coordinate
    if time is None or START_DATETIME not in raster.coords:
        return raster
    stated = time.attrs.get("bounds", time.encoding.get("bounds"))
    if stated is not None and stated in raster.variables:
        raise ValueError(
            f"time names CF bounds {stated!r} and the raster also states "
            "start_datetime/end_datetime; drop one of them before writing"
        )
    starts = raster[START_DATETIME].variable.set_dims(time.dims)
    ends = raster[END_DATETIME].variable.set_dims(time.dims)
    cells = xr.Variable.concat([starts, ends], _BOUNDS_DIMENSION)
    cells = cells.transpose(*time.dims, _BOUNDS_DIMENSION)
    name = f"{time.name}_bounds"
    # A shallow copy owns its variables' attrs and encoding, so the caller's stay as they were.
    encoded = raster.copy().drop_vars([START_DATETIME, END_DATETIME])
    encoded = encoded.assign_coords({name: cells})
    encoding = {
        "units": time.encoding.get("units", _TIME_UNITS),
        "calendar": time.encoding.get("calendar", _TIME_CALENDAR),
        # Lazy datetimes encode only with both units and dtype prescribed.
        "dtype": time.encoding.get("dtype", "int64"),
    }
    encoded[time.name].attrs["bounds"] = name
    for variable in (time.name, name):
        encoded[variable].encoding = {**encoded[variable].encoding, **encoding}
    return encoded


def decode_time_bounds(raster: xr.Dataset) -> xr.Dataset:
    """Replace a CF time boundary variable with start/end coordinates after reading.

    The boundary variable is the one `time`'s `bounds` reference names, in
    attrs or, after `decode_coords="all"`, in encoding; any name and vertex
    dimension are accepted.

    Args:
        raster: Dataset as a reader opened it.

    Returns:
        Dataset with `start_datetime` and `end_datetime` in place of the
        boundary variable, or `raster` unchanged when time states none.
    """
    from geosave.geodata.core.conventions import END_DATETIME, START_DATETIME

    time = raster.gs.time_coordinate
    if time is None:
        return raster
    name = time.attrs.get("bounds", time.encoding.get("bounds"))
    if name is None or name not in raster.variables:
        return raster
    cells = raster[name]
    vertex = [dim for dim in cells.dims if dim not in time.dims][0]
    decoded = (
        raster.copy()
        .drop_vars(name)
        .assign_coords(
            {
                START_DATETIME: cells.isel({vertex: 0}, drop=True).variable,
                END_DATETIME: cells.isel({vertex: 1}, drop=True).variable,
            }
        )
    )
    decoded[time.name].attrs.pop("bounds", None)
    decoded[time.name].encoding.pop("bounds", None)
    return decoded


def name_axis(raster: xr.Dataset, axis: Literal["Y", "X"]) -> str | None:
    """Return the CF `standard_name` of a spatial axis on the raster's CRS.

    Args:
        raster: Dataset whose CRS decides the name.
        axis: The axis to name.

    Returns:
        `latitude`/`longitude` on a geographic CRS, the projection coordinate
        names on a projected one, or None without a CRS.
    """
    crs = raster.gs.crs
    if crs is None:
        return None
    if crs.geographic:
        return "latitude" if axis == "Y" else "longitude"
    return f"projection_{axis.lower()}_coordinate"


def encode_spatial_axes(raster: xr.Dataset) -> xr.Dataset:
    """Write CF axis metadata and grid mapping links before writing.

    In memory ODC finds spatial axes by name and the CRS coordinate by its
    attrs; files carry `axis` and `standard_name` (CF §4), each variable's
    `grid_mapping` (CF §5.6), and no `_FillValue` on coordinates (CF §5).

    Args:
        raster: Dataset about to be written.

    Returns:
        Shallow copy whose spatial coordinates carry CF attrs, or `raster`
        unchanged without spatial dimensions.
    """
    dims = raster.gs.spatial_dims
    if dims is None:
        return raster
    encoded = raster.copy(deep=False)
    for dim, axis in zip(dims, _AXES, strict=True):
        if dim not in encoded.coords:
            continue
        coord = encoded[dim].variable
        coord.attrs = {**coord.attrs, "axis": axis}
        standard_name = name_axis(raster, axis)
        if standard_name is not None:
            coord.attrs["standard_name"] = standard_name
        coord.encoding = {**coord.encoding, "_FillValue": None}
    crs_coord = raster.odc.crs_coord
    if crs_coord is None:
        return encoded
    # As an attr the link survives an `encoding=` the caller passes to the writer.
    for variable in encoded.data_vars.values():
        if set(dims) <= set(variable.dims):
            variable.encoding.pop("grid_mapping", None)
            variable.attrs["grid_mapping"] = str(crs_coord.name)
    return encoded


def decode_spatial_axes(raster: xr.Dataset) -> xr.Dataset:
    """Drop the CF axis attrs `encode_spatial_axes` writes, after reading.

    They restate what the dimension names and the CRS already say, so a file
    reads back as the raster that was written. An attr stating anything else
    is the file's own and stays.

    Args:
        raster: Dataset as a reader opened it.

    Returns:
        Shallow copy without the derived axis attrs, or `raster` unchanged
        without spatial dimensions.
    """
    dims = raster.gs.spatial_dims
    if dims is None:
        return raster
    decoded = raster.copy(deep=False)
    for dim, axis in zip(dims, _AXES, strict=True):
        if dim not in decoded.coords:
            continue
        coord = decoded[dim].variable
        derived = {"axis": axis, "standard_name": name_axis(raster, axis)}
        kept = {}
        for key, value in coord.attrs.items():
            if key not in derived or derived[key] != value:
                kept[key] = value
        coord.attrs = kept
    return decoded


TIME_BOUNDS = Codec(encode_time_bounds, decode_time_bounds)
SPATIAL_AXES = Codec(encode_spatial_axes, decode_spatial_axes)
