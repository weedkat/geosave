"""Xarray raster structure and coordinate names used by GeoSave.

`.gs` works on 2-D georeferenced rasters, as odc-geo does: one grid, with
optional `time` observations and a `band` axis that only stacking adds. Over
ODC it adds time cells, STAC metadata kept on the object, and band attr
groups. Other layouts (depth, ensembles, rotated poles) are out of scope.

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

The structure is the one ODC produces; the accessor finds every part by
name, as ODC does. Spatial dimensions are named by the grid's CRS::

    projected CRS    (..., y, x)
    geographic CRS   (..., latitude, longitude)
    time             always "time", as odc-stac names it

Where each part lives, in memory and in a file::

    in memory                    Zarr                     netCDF
    time(time)                   time(time)               time(time) bounds="time_bounds"
    start_datetime(time) ┐       start_datetime(time)     time_bounds(time, bounds)  (CF §7.1)
    end_datetime(time)   ┘       end_datetime(time)
    x(x), y(y)  ODC attrs        + axis, standard_name (CF §4), no coordinate _FillValue
    spatial_ref()                spatial_ref(); red:grid_mapping = "spatial_ref"

    GeoTIFF/COG: TIFFTAG_DATETIME and ACDD time_coverage_start/end

Who writes the cells::

    one STAC Item per label            its start_datetime / end_datetime
    groupby="solar_day" mosaic         the local solar day it covers
    gs.composite(freq)                 the resample bin
    gs.write_time_bounds("MS")         one freq step per label
    a cell ends where the next starts (CF convention)

STAC metadata keeps its own shape beside it::

    scene.coords["stac_properties"] (time)  JSON Item properties per label
    scene.attrs["stac_collection"]          JSON Collection
    scene.attrs["stac_extensions"]          JSON extension schema URIs

A DataTree node inherits its parents' coordinates, so asset nodes share the
collection node's time and cells::

    /s2              time, start_datetime, end_datetime, stac_properties
    ├── /B04         B04 (time, y, x)   its own grid
    └── /B11         B11 (time, y, x)   its own grid

Channels stacked in one DataArray span ``band``. Layouts ODC does not
recognize, such as a weather model's rotated pole, are renamed by the caller.
"""

from __future__ import annotations

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

# The collection node of Items that name no collection, in a STAC-shaped tree.
DEFAULT_COLLECTION = "default"


# Coordinate along `time` holding each date's STAC Item properties as JSON
# text: one mapping per date, which NetCDF and Zarr store as a string.
STAC_PROPERTIES = "stac_properties"
