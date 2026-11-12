# Read and write data

Save a raster as a Cloud Optimized GeoTIFF (COG), then open it:

```python
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from odc.geo.xr import xr_coords

import geosave as gs

root = Path(tempfile.mkdtemp())
from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500040, 5000040),
    crs='EPSG:32633', resolution=10,
)
pixels = np.arange(1, 17, dtype="uint8").reshape(4, 4)
scene = xr.Dataset({"red": (("y", "x"), pixels)}).assign_coords(xr_coords(grid)).gs.write_nodata(0)
files = scene.gs.to_cog(root, split_asset=(), format_paths="scene.tif")
reopened = gs.open_raster(files[0])
reopened
```

Output:

```text
<xarray.Dataset> Size: 88B
Dimensions:      (x: 4, y: 4)
Coordinates:
  * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
  * y            (y) float64 32B 5e+06 5e+06 5e+06 5e+06
    spatial_ref  int64 8B 0
Data variables:
    red          (y, x) uint8 16B dask.array<chunksize=(4, 4), meta=np.ndarray>
Attributes:
    AREA_OR_POINT:  Area
```

The reopened object is an xarray `Dataset`, with one variable per band.
The file preserves this raster's grid, `uint8` values, variable name, and nodata.
`to_cog` returns a list of written UPaths, in time then variable order.
`gs.open_raster(files[0])` reads the single file in this example. For files split across bands or dates, write a catalog and use `rows.gs.to_datatree()`. Only GeoTIFF is split
into files, by `split_item` and `split_asset`, and a template names them
(`{collection}`, `{time}`, `{asset}`, `{variable}`), since a
GeoTIFF holds one grid at one date. `to_zarr` and `to_netcdf` take no template:
they write one store and return its UPath, and a tree's store keeps its
collections and grid nodes as groups, which `gs.open_raster(path)` reads back.
With `compute=False`, each writer returns one native Dask Delayed whose
computed result has the same shape.

Readers pick the format from the path suffix:

| Object | Reader | Writers |
| --- | --- | --- |
| Raster: xarray `Dataset` | `gs.open_raster(path)` | `.gs.to_cog`, `.gs.to_zarr`, `.gs.to_netcdf` |
| Tree: xarray `DataTree` | `gs.open_raster(path)` | `.gs.to_cog`, `.gs.to_zarr`, `.gs.to_netcdf` |
| Vector: GeoPandas `GeoDataFrame` | `gs.open_vector(path)` | `.gs.to_geoparquet`, `.gs.to_geojson`, `.gs.to_geopackage` |

A location can be a string, `Path`, or `UPath`. For a full workflow, see
[existing files](../guides/ingest-files.md).

## Choose a format

Use **COG** for geospatial image files and remote reads of selected areas.
Use **Zarr** for multidimensional arrays stored in chunks.
Use **NetCDF** for multidimensional scientific data in one file.
Use **GeoParquet** for vector tables and raster item catalogs.

```python
store = scene.gs.to_zarr(root / "scene.zarr")
zarr_scene = gs.open_raster(store, chunks={})
zarr_scene
```

Output:

```text
<xarray.Dataset> Size: 84B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 5e+06 5e+06 5e+06 5e+06
  * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
    spatial_ref  int32 4B ...
Data variables:
    red          (y, x) uint8 16B dask.array<chunksize=(4, 4), meta=np.ndarray>
```

```python
path = scene.gs.to_netcdf(root / "scene.nc")
netcdf_scene = gs.open_raster(path)
netcdf_scene
```

Output:

```text
<xarray.Dataset> Size: 84B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 5e+06 5e+06 5e+06 5e+06
  * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
    spatial_ref  int32 4B ...
Data variables:
    red          (y, x) uint8 16B dask.array<chunksize=(4, 4), meta=np.ndarray>
```

Zarr and NetCDF writers return a destination `UPath`, which readers accept.
`chunks={}` is the convenience opener default: Dask arrays keep arithmetic
lazy. Pass `chunks=None` for backend indexing without Dask. Writing computes
the pixels. See [lazy data](../concepts/lazy-data.md).

Close file-backed objects when you finish:

```python
reopened.close()
zarr_scene.close()
netcdf_scene.close()
```

Time-series file templates, tree layouts, suffix rules and remote storage are in
[Storage and file layouts](../concepts/storage.md).
