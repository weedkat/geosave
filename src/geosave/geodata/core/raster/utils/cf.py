"""Read CF structural relationships without interpreting them as raster pixels."""

from __future__ import annotations

from collections.abc import Hashable

import cf_xarray  # noqa: F401 registers .cf
import xarray as xr


def read_support_variables(dataset: xr.Dataset) -> tuple[Hashable, ...]:
    """Name the scalar CRS carriers in Dataset order, including coordinates.

    Other CF structure GeoSave does not model, such as spatial bounds or cell
    measures, stays as ordinary variables for `.cf` to read.
    """
    names = set()
    for references in dataset.cf.grid_mapping_names.values():
        names.update(references)
    # GDAL/ODC can carry WKT without a CF projection name.
    for name, variable in dataset.variables.items():
        if not variable.dims and {"crs_wkt", "spatial_ref"}.intersection(
            variable.attrs
        ):
            names.add(name)
    return tuple(name for name in dataset.variables if name in names)


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
    from geosave.geodata.conventions import END_DATETIME, START_DATETIME

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
    from geosave.geodata.conventions import END_DATETIME, START_DATETIME

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


def drop_coordinate_fill(raster: xr.Dataset) -> xr.Dataset:
    """Keep `_FillValue` off spatial coordinate variables before writing (CF §5).

    Args:
        raster: Dataset about to be written.

    Returns:
        Shallow copy whose spatial coordinates encode no fill value.
    """
    dims = raster.gs.spatial_dims
    if dims is None:
        return raster
    encoded = raster.copy()
    for dim in dims:
        if dim in encoded.coords:
            encoded[dim].encoding = {**encoded[dim].encoding, "_FillValue": None}
    return encoded
