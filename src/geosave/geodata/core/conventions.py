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

    stac.load_dataset                  its Items' earliest start to latest end;
                                       an Item stating only `datetime`, that instant
    gs.composite(freq)                 the resample bin
    gs.write_time_bounds("MS")         one freq step per label
    a cell ends where the next starts (CF convention)

STAC Item properties are coordinates, marked by a `stac_property` attr that
holds the property's STAC key. The mark, not the name, makes a coordinate a
property: ODC names `eo:cloud_cover` as `eo_cloud_cover`, or as the request's
`name` says, and a rename keeps the mark::

    >>> scene = stac.load_dataset(
    ...     items, geobox=grid, groupby="id", chunks={},
    ...     with_properties=[{"key": "eo:cloud_cover", "dtype": "float64"},
    ...                      {"key": "platform", "dtype": "str"}],
    ... )
    >>> scene
    <xarray.Dataset> Size: 188B
    Dimensions:         (y: 2, x: 3, time: 2)
    Coordinates:
      * y               (y) float64 16B 5e+06 5e+06
      * x               (x) float64 24B 5e+05 5e+05 5e+05
      * time            (time) datetime64[ns] 16B 2025-06-01 2025-07-01
        eo_cloud_cover  (time) float64 16B 12.5 40.0
        platform        (time) <U11 88B 'sentinel-2b' 'sentinel-2b'
        spatial_ref     int32 4B 32633
    Data variables:
        red             (time, y, x) uint16 24B dask.array<chunksize=(1, 2, 3), meta=np.ndarray>
    Attributes:
        stac_extensions:  ["https://stac-extensions.github.io/projection/v2.0.0/s...
    >>> scene["eo_cloud_cover"].attrs
    {'units': '1', 'stac_property': 'eo:cloud_cover'}
    >>> scene.gs.stac_properties
    {'eo:cloud_cover': 'eo_cloud_cover', 'platform': 'platform'}

Who writes each property::

    with_properties=[...]       ODC loads it as a (time,) data variable, one
                                value per step (fused Items: numbers averaged,
                                text joined with ","); the "properties" stamp
                                moves it to a coordinate and marks it
    gs.write_stac_property      a (time,) or scalar coordinate, by hand

The Collection and the schemas the properties need are Dataset attrs::

    scene.attrs["stac_collection"]          JSON Collection, without links
    scene.attrs["stac_extensions"]          JSON extension schema URIs

DataTree keeps its native, caller-defined hierarchy. General raster helpers
operate on raster-bearing nodes, including parents. The STAC-specific tree
layout is documented by ``stac.raster.RasterStac``; it is not a general
DataTree convention. The tree stages publication only; Items load back as Datasets.

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

# How RasterStac.from_dataset names an Item: its start, its MGRS square and its centroid, so
# Items of one collection over many regions and dates stay unique and sort by
# time. Fields are those of `GeoRegion.format`.
ITEM_ID_FORMAT = "{start:%Y%m%dT%H%M%S}_{mgrs}_{lat:.4f}_{lon:.4f}"
