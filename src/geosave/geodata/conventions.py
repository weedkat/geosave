"""Xarray raster structure and coordinate names used by GeoSave.

A Dataset: measurements as data variables, each time label's cell as two
coordinates, the CRS as a scalar coordinate::

    >>> scene
    <xarray.Dataset> Size: 164B
    Dimensions:         (time: 2, y: 2, x: 3)
    Coordinates:
      * time            (time) datetime64[ns] 16B 2025-06-01 2025-07-01
        start_datetime  (time) datetime64[ns] 16B 2025-06-01 2025-07-01
        end_datetime    (time) datetime64[ns] 16B 2025-07-01 2025-08-01
      * y               (y) float64 16B 5e+06 5e+06
      * x               (x) float64 24B 5e+05 5e+05 5e+05
        spatial_ref     int32 4B 32633
    Data variables:
        red             (time, y, x) uint16 24B dask.array<chunksize=(1, 2, 3), meta=np.ndarray>
        nir             (time, y, x) uint16 24B dask.array<chunksize=(1, 2, 3), meta=np.ndarray>
        dem             (y, x) float32 24B dask.array<chunksize=(2, 3), meta=np.ndarray>
    Attributes:
        title:    Monthly reflectance and elevation

A DataArray keeps its cells, because they share the time dimension::

    >>> scene["red"]
    <xarray.DataArray 'red' (time: 2, y: 2, x: 3)> Size: 24B
    dask.array<xarray-red, shape=(2, 2, 3), dtype=uint16, chunksize=(1, 2, 3), chunktype=numpy.ndarray>
    Coordinates:
      * time            (time) datetime64[ns] 16B 2025-06-01 2025-07-01
        start_datetime  (time) datetime64[ns] 16B 2025-06-01 2025-07-01
        end_datetime    (time) datetime64[ns] 16B 2025-07-01 2025-08-01
      * y               (y) float64 16B 5e+06 5e+06
      * x               (x) float64 24B 5e+05 5e+05 5e+05
        spatial_ref     int32 4B 32633
    Attributes:
        long_name:  Red reflectance
        units:      1

Where each part lives, in memory and in a netCDF or Zarr file::

    in memory                              file
    time(time)                             time(time)  bounds="time_bounds"
    start_datetime(time) ┐   encode  ──►   time_bounds(time, bounds)
    end_datetime(time)   ┘   ◄── decode      units, calendar = time's (CF §7.1)
    x(x), y(y)                             x(x), y(y)  no _FillValue (CF §5)
    spatial_ref()  grid mapping (ODC)      spatial_ref()
    red.encoding["grid_mapping"]           red:grid_mapping = "spatial_ref"

Who writes the cells::

    one STAC Item per label            its start_datetime / end_datetime
    groupby="solar_day" mosaic         the local solar day it covers
    gs.composite(freq)                 the resample bin
    gs.write_time_bounds("MS")         one freq step per label
    a cell ends where the next starts (CF convention)

``.cf`` reads the CF structure; it stores no separate metadata::

    scene.cf.axes            {"X": ["x"], "Y": ["y"], "T": ["time"]}
    scene.cf.grid_mappings   spatial_ref, with its CRS

STAC metadata keeps its own shape beside it::

    scene.coords["stac_properties"] (time)  JSON Item properties per label
    scene.attrs["stac_collection"]          JSON Collection
    scene.attrs["stac_extensions"]          JSON extension schema URIs

A DataTree node inherits its parents' coordinates, so asset nodes share the
collection node's time and cells::

    /s2              time, start_datetime, end_datetime, stac_properties
    ├── /B04         B04 (time, y, x)   its own grid
    └── /B11         B11 (time, y, x)   its own grid

GeoSave creators use ``y``/``x``, ``time``, and ``band`` when channels are
stacked in one DataArray. Native inputs keep other names; CF attrs identify
their axes.

See https://cf-xarray.readthedocs.io/en/latest/quickstart.html.
"""

from __future__ import annotations

from typing import cast

import cf_xarray  # noqa: F401 registers .cf
import xarray as xr

# Canonical spatial names for rasters GeoSave creates.
SPATIAL_DIMENSIONS = ("y", "x")

# Scalar coordinate holding the CRS, which CF reaches through `grid_mapping`.
GRID_MAPPING = "spatial_ref"

# Axis the bands occupy: a file's bands, stacked variables, or display channels.
BAND_DIMENSION = "band"

# Axis a raster spans when it carries observations over time.
TIME_COORDINATE = "time"

# The period a time label covers, one value per label along `time`, named as
# STAC Items name it. These observations are distinct from CF cell bounds.
START_DATETIME = "start_datetime"
END_DATETIME = "end_datetime"

# Coordinate along `time` holding each date's STAC Item properties as JSON
# text: one mapping per date, which NetCDF and Zarr store as a string.
STAC_PROPERTIES = "stac_properties"


def to_yx[T: (xr.DataArray, xr.Dataset)](data: T) -> T:
    """Rename a foreign spatial pair to `y` and `x`.

    The pair is found by name (`latitude`/`longitude`, `lat`/`lon`) or, under
    any other name, by the CF `axis` or `standard_name` its coordinates carry.

    Args:
        data: Band or raster as a loader named its grid.

    Returns:
        Renamed view sharing its pixels, lazily. A band or raster already on
        `y` and `x`, or without identifiable spatial dimensions, comes back
        as itself. A tree normalizes node by node with
        ``tree.map_over_datasets(to_yx)``.

    Examples:
        >>> to_yx(xr.open_dataset("era5.nc")).t2m.dims
        ('time', 'y', 'x')
    """
    dims = data.gs.spatial_dims
    if dims in (None, SPATIAL_DIMENSIONS):
        return data
    return cast("T", data.rename(dict(zip(dims, SPATIAL_DIMENSIONS, strict=True))))
