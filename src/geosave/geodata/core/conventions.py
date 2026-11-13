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

    scene.coords["eo:cloud_cover"]          one coordinate per Item property, scalar
                                            when shared, attr stac_property = its key
    scene.attrs["stac_collection"]          JSON Collection
    scene.attrs["stac_extensions"]          JSON extension schema URIs

A STAC tree is a DataTree shaped as STAC: catalog, collection, Item,
asset. Each level holds the STAC metadata of its object, and a node
inherits its ancestors' coordinates; attrs are not inherited::

    /                          catalog      nothing of its own
    └── /sentinel-2-l2a        collection   attrs   stac_collection, stac_extensions
        ├── /20250601T...      Item         attrs   the Dataset's other attrs (title, ...)
        │   │                               coords  time, start_datetime, end_datetime,
        │   │                                       one scalar coord per property
        │   ├── /B04           asset        vars    B04 (y, x)  attrs: CF packing, flags
        │   │                               coords  y, x, spatial_ref: its own grid
        │   └── /SCL           asset        vars    SCL (y, x) on another grid
        └── /20250606T...      Item
    ├── /labels                collection   another layer, on its own timeline

Rules:

    data              only on assets, the leaves, each at level 3
    collections       side by side at level 1, one layer each
    node names        collection id ("default" when none is stated), the
                      Item id, the asset key; each node path is a file path
    one asset         one variable, with its own attrs and grid
    grid              on each asset, as STAC's proj:* is; Items may cover
                      different extents of one grid
    Item properties   scalar coords on the Item node, so combining Items lays
                      them out as (time,) coords, as on a Dataset

``ds.gs.expand(item_dim)`` builds the tree from one collection's Dataset;
``tree["<collection>"].gs.combine(assets)`` returns it. The tree rules
``stac_catalog``, ``stac_collection``, ``same_grid``, ``same_geobox`` and
``same_pixel_meaning`` live in ``core.raster.datatree``.

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

# How `expand` names an Item: its start, its MGRS square and its centroid, so
# Items of one collection over many regions and dates stay unique and sort by
# time. Fields are those of `GeoRegion.format`.
ITEM_ID_FORMAT = "{start:%Y%m%dT%H%M%S}_{mgrs}_{lat:.4f}_{lon:.4f}"
