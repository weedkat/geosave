# Save imagery for GIS or Python

Export selected bands as a Cloud Optimized GeoTIFF, or keep them in a chunked
Zarr store. Reopen the output with its band names and geographic grid intact.

**You need:** the base package and the [sample scene](first-raster.md#open-the-sample).

## Save a GeoTIFF for a GIS

```python
from pathlib import Path

import geosave as gs
from geosave.geodata import io

Path("output").mkdir(exist_ok=True)
scene = gs.open_raster("sample-scene.tif")
image = scene[["red", "nir"]]
path = io.write_cog(image, "output/scene.tif")
print(path)
```

```text
output/scene.tif
```

```python
reopened = gs.open_raster(path)
bands = list(reopened.data_vars)
bands
```

```text
['red', 'nir']
```

The file keeps the band names, stored values, metadata and grid. Writing reads
pixels. Existing destinations are refused unless you pass `overwrite=True`.

A COG is a GeoTIFF organized for reading spatial windows efficiently. It is
useful for sharing imagery with GIS tools or reading portions remotely.
Its bands share a grid and storage dtype; keep categorical labels separate
from imagery when their stored representations differ.

## Save a Zarr store for Python workflows

```python
store = image.gs.to_zarr("output/scene.zarr")
reopened = gs.open_raster(store)
bands = list(reopened.data_vars)
bands
```

```text
['red', 'nir']
```

Zarr supports chunked arrays and time series. NetCDF is another option when
you need multidimensional data in one file:

```python
path = image.gs.to_netcdf("output/scene.nc")
print(path)
```

```text
output/scene.nc
```

These formats keep named variables and dimensions, so a time series can stay
one Dataset instead of becoming a folder of individual image files. Zarr's
chunks can be read separately; NetCDF provides a single-file exchange format.

## Choose for your project

| Goal | Format |
| --- | --- |
| Share one image or map with GIS tools | Cloud Optimized GeoTIFF |
| Work with chunked arrays or time series | Zarr |
| Exchange scientific arrays in one file | NetCDF |
| Save vector features | [GeoParquet](vectors.md) |
| Organize dated raster files by footprint | [STAC catalog](../advanced/raster-catalog.md) |

`gs.open_raster` reads one Dataset. Use `gs.open_raster_tree` when a Zarr or
NetCDF file contains a hierarchy of groups. See [I/O reference](../reference/io.md)
for format options and [storage concepts](../concepts/storage.md) for remote paths.
