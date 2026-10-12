# Prepare data from existing files

```python
import tempfile
from pathlib import Path

import numpy as np
import geosave as gs

# Make an input file so this example runs without downloads.
root = Path(tempfile.mkdtemp())
anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=4, resolution=10, crs="EPSG:32633"
)
red = np.arange(1, 17, dtype="uint16").reshape(4, 4)
original = gs.raster(
    {"red": (("y", "x"), red), "nir": (("y", "x"), red + 4)},
    anchor.geobox, nodata=0,
)
(source,) = original.gs.to_cog(root / "scene.tif")

scene = gs.open_raster(source, chunks={"y": 2, "x": 2})
scene
```

Output:

```text
<xarray.Dataset> Size: 136B
Dimensions:      (x: 4, y: 4)
Coordinates:
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
    spatial_ref  int64 8B 0
Data variables:
    red          (y, x) uint16 32B dask.array<chunksize=(2, 2), meta=np.ndarray>
    nir          (y, x) uint16 32B dask.array<chunksize=(2, 2), meta=np.ndarray>
Attributes:
    AREA_OR_POINT:  Area
```

For your own file, set `source = Path("data/scene.tif")` and start at
`gs.open_raster`. The file supplies its grid; you need an anchor only when
creating pixels or choosing a target grid. Opening reads metadata and builds
lazy arrays. It leaves stored values and their dtype unchanged unless you
request `mask_and_scale=True`.

Use this activity when you already know the file locations. To discover images
by location, date, or cloud cover, use [STAC acquisition](stac.md). To run the
acquisition recipes in a model spec as a Prefect job, use
[optional flows](workflows.md).

## Choose the reader

| Input | Public call | Result |
| --- | --- | --- |
| One TIFF, JPEG 2000, PNG, NetCDF file, or Zarr store | `gs.open_raster(path, **options)` | xarray `Dataset`, one variable per band or stored variable. |
| Names mapped to raster locations, or a folder with one file/store per name | `gs.open_stack(source, **options)` | xarray `DataTree`, one raster group per name. |
| GeoParquet, GeoJSON, or GeoPackage | `gs.open_vector(path, **options)` | GeoPandas `GeoDataFrame`, with the file's geometry and columns. |

Readers select a backend by suffix. Plain PNG files often lack a CRS and grid;
opening them does not invent a location. See [I/O](../basics/io.md) for supported
suffixes, remote access, and backend options.

The example file stores its variable names. Other GDAL files without that
metadata use `band_1`, `band_2`, and so on. Check the opened Dataset, then rename
with normal xarray operations when you know what each band represents.
Importing GeoSave registers `.gs` on the native objects.

## Select and calculate

```python
patch = scene.isel(y=slice(0, 2), x=slice(0, 2))
ndvi = gs.features.ndvi(patch, nir="nir", red="red").rename("ndvi")
ndvi.compute()
```

Output:

```text
<xarray.DataArray 'ndvi' (y: 2, x: 2)> Size: 16B
array([[0.6666667, 0.5      ],
       [0.2857143, 0.25     ]], dtype=float32)
Coordinates:
  * y            (y) float64 16B 4.987e+06 4.987e+06
  * x            (x) float64 16B 2.635e+05 2.635e+05
    spatial_ref  int64 8B 0
```

`isel` selects pixel positions and keeps their coordinates. The feature
function decodes the selected bands' nodata and packing, then calculates
`(nir - red) / (nir + red)` lazily. These synthetic values demonstrate the
calculation; real NDVI needs red and near-infrared reflectance.
The explicit `.compute()` reads the selected pixels for this displayed result.
It returns a computed array; `ndvi` itself remains lazy.

Use `.gs.crop(geometry)` for an area in ground coordinates, or
`.gs.reproject(target, resampling=...)` to change the grid. For your own
arithmetic, call `.gs.mask_and_scale()` first when the data is packed or marks
nodata with an integer. [Large rasters](large-data.md) shows these execution
boundaries in more detail.

## Save the result

```python
prepared = gs.raster({"ndvi": ndvi}, patch).gs.write_nodata(np.nan)
(destination,) = prepared.gs.to_cog(root / "prepared/ndvi.tif")
destination.name
```

Output:

```text
'ndvi.tif'
```

Writing computes the pixels and returns the destination `UPath`. Here NaN is
the declared fill value for the floating-point output. `gs.raster` uses the
patch as its grid reference; the result has one variable with new meaning.
Reopen with `gs.open_raster(destination)` in another process. Choose
`.gs.to_zarr(path)` or `.gs.to_netcdf(path)` for multidimensional output.
Writers refuse existing destinations unless you pass `overwrite=True`.

## Describe saved files in a catalog

```python
from datetime import UTC, datetime

items = gs.stac.create_item(
    [destination], collection="ndvi", datetime=datetime(2025, 1, 1, tzinfo=UTC)
)
catalog = items.gs.to_geoparquet(root / "prepared/catalog.parquet")
gs.open_vector(catalog)[["collection", "datetime"]]
```

Output:

```text
  collection                  datetime
0       ndvi 2025-01-01 00:00:00+00:00
```

`create_item` reads file headers and returns a GeoDataFrame of STAC Items;
it does not read their pixels. This undated synthetic file needs an explicit
date. For your own data, use its actual acquisition date or interval; files
with a `time` coordinate already supply that information.
The catalog stores asset addresses and metadata. The TIFF still holds the
pixels. Continue with [catalogs and windows](catalogs-and-cuts.md) to prepare
model inputs from many saved files.

## Use a vector file as labels

```python
import geopandas as gpd
from shapely.geometry import box

bounds = scene.gs.bounds
fields = gpd.GeoDataFrame(
    {"class": [1]},
    geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top)],
    crs=bounds.crs,
)
path = fields.gs.to_geoparquet(root / "fields.parquet")
features = gs.open_vector(path)
labels = features.gs.rasterize(patch, column="class", dtype="uint8", fill=0)
labels
```

Output:

```text
<xarray.DataArray 'class' (y: 2, x: 2)> Size: 4B
array([[1, 1],
       [1, 1]], dtype=uint8)
Coordinates:
  * y            (y) float64 16B 4.987e+06 4.987e+06
  * x            (x) float64 16B 2.635e+05 2.635e+05
    spatial_ref  int64 8B 0
```

Replace `path` with your vector file and start at `gs.open_vector`.
Rasterization uses the patch's exact grid, transforms geometries into its CRS,
and immediately burns the selected column into pixels. `fill=0` supplies the
value outside features; reserve it separately from valid class codes.
Reading a vector file loads a table, without the Dask pixel graph used by
rasters. See [vectors](../basics/vectors.md) for masks and spatial selection.

```python
scene.close()
```

Close the source after computing or saving derived results that still depend
on its file-backed arrays.
