"""Xarray raster structure and coordinate names used by GeoSave.

A Dataset holds named measurements and their CF support variables. Here
reflectance varies with time, elevation does not, and ``time_bounds``
delimits each time label's cell rather than being another image band::

    >>> scene
    <xarray.Dataset> Size: 164B
    Dimensions:      (time: 2, y: 2, x: 3, bounds: 2)
    Coordinates:
      * time         (time) datetime64[ns] 16B 2025-06-01 2025-06-11
      * y            (y) float64 16B 5e+06 5e+06
      * x            (x) float64 24B 5e+05 5e+05 5e+05
        time_bounds  (time, bounds) datetime64[ns] 32B 2025-05-31 ... 2025-06-12
        spatial_ref  int32 4B 32633
    Dimensions without coordinates: bounds
    Data variables:
        red          (time, y, x) uint16 24B dask.array<chunksize=(1, 2, 3), meta=np.ndarray>
        nir          (time, y, x) uint16 24B dask.array<chunksize=(1, 2, 3), meta=np.ndarray>
        dem          (y, x) float32 24B dask.array<chunksize=(2, 3), meta=np.ndarray>
    Attributes:
        title:    Reflectance and elevation

A DataArray holds one variable, its coordinates, and its own attrs. It cannot
hold ``time_bounds``, whose ``bounds`` dimension no pixel variable has::

    >>> scene["red"]
    <xarray.DataArray 'red' (time: 2, y: 2, x: 3)> Size: 24B
    dask.array<zeros_like, shape=(2, 2, 3), dtype=uint16, chunksize=(1, 2, 3), chunktype=numpy.ndarray>
    Coordinates:
      * time         (time) datetime64[ns] 16B 2025-06-01 2025-06-11
      * y            (y) float64 16B 5e+06 5e+06
      * x            (x) float64 24B 5e+05 5e+05 5e+05
        spatial_ref  int32 4B 32633
    Attributes:
        long_name:     Red reflectance
        units:         1
        scale_factor:  0.0001
        _FillValue:    0

The CF storage structure behind those printouts (selected attrs shown). Every
entry is a variable; ``coords`` and ``data_vars`` are xarray's grouping::

    scene: xr.Dataset
    ├── attrs
    │   └── title: "Reflectance and elevation"
    ├── coords
    │   ├── x(x), y(y)
    │   │   └── attrs: axis, standard_name ("projection_x_coordinate"), units
    │   ├── time(time)
    │   │   ├── data: datetime64[ns] [2025-06-01, 2025-06-11]
    │   │   └── attrs: axis "T", standard_name "time",
    │   │              bounds "time_bounds" ──► coords["time_bounds"]
    │   ├── time_bounds(time, bounds)            CF boundary variable
    │   │   └── data: [[2025-05-31, 2025-06-02], [2025-06-10, 2025-06-12]]
    │   └── spatial_ref()                        CF grid mapping, written by ODC
    │       └── attrs: grid_mapping_name, crs_wkt, projection parameters
    └── data_vars
        ├── red(time, y, x)
        │   ├── data: lazy uint16 pixels
        │   ├── attrs: long_name, units, scale_factor, _FillValue
        │   └── encoding: grid_mapping "spatial_ref" ──► coords["spatial_ref"]
        ├── nir(time, y, x)
        └── dem(y, x)

``.cf`` reads this structure; it stores no separate metadata::

    scene.cf.axes                    {"X": ["x"], "Y": ["y"], "T": ["time"]}
    scene.cf.get_bounds("time")      scene["time_bounds"], never estimated
    scene.cf.grid_mappings           spatial_ref, with its CRS

A reference such as ``bounds`` or ``grid_mapping`` lives in attrs or, once a
file is decoded, in encoding; ``.cf`` reads either. Decoding likewise moves
consumed ``scale_factor`` and ``_FillValue`` from attrs into encoding. GeoSave
adds structure through ``gs.write_*`` methods, which state it as these attrs,
encodings and variables.

Selection keeps a coordinate only when the selected variables have all of its
dimensions, so ``time_bounds`` follows ``drop_vars`` and ``.cf`` selection::

    scene[["red"]]               red, time, x, y, spatial_ref
    scene.cf[["red"]]            + time_bounds   (unreferenced coordinates dropped)
    scene.drop_vars(["nir", "dem"])  + time_bounds   (every other variable kept)

Time cells come from what GeoSave knows: a loaded Item's period, where one
label is one Item, or a composite's resample bin. STAC metadata stays in its
own shape beside the CF structure::

    scene.coords["stac_properties"] (time)  JSON Item properties per observation
    scene.attrs["stac_collection"]          JSON Collection
    scene.attrs["stac_extensions"]          JSON extension schema URIs
    scene.gs.stac_common_properties         intersection over observations

STAC's ``start_datetime`` and ``end_datetime`` appear only on Items, where
loading reads and publication writes them.

GeoSave creators use ``y``/``x``, ``time``, and ``band`` when channels are
stacked in one DataArray. Native inputs can retain other names: CF attrs
identify their axes; ``to_yx`` explicitly renames the spatial pair.
DataTree nodes hold Datasets and inherit parent coordinates;
``to_dataset(inherit="all_coords")`` includes a parent's ``time_bounds``.
Independent grids can occupy sibling nodes.

See https://cf-xarray.readthedocs.io/en/latest/quickstart.html.
"""

from __future__ import annotations

from typing import cast

import cf_xarray  # noqa: F401 registers .cf
import xarray as xr

# Canonical spatial names for rasters GeoSave creates.
SPATIAL_DIMENSIONS = ("y", "x")

# Scalar coordinate holding the CRS, which CF reaches through `grid_mapping`.
CRS_COORDINATE = "spatial_ref"

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

