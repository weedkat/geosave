# Read and write data

Save a raster as a Cloud Optimized GeoTIFF (COG), then open it:

```python
import tempfile
from pathlib import Path

import numpy as np

import geosave as gs

root = Path(tempfile.mkdtemp())
anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=4, resolution=10, crs="EPSG:32633"
)
pixels = np.arange(1, 17, dtype="uint8").reshape(4, 4)
scene = gs.raster({"red": (("y", "x"), pixels)}, anchor.geobox, nodata=0)
files = scene.gs.to_cog(root / "scene.tif")
reopened = gs.open_raster(files)
reopened
```

Output:

```text
<xarray.Dataset> Size: 88B
Dimensions:      (x: 4, y: 4)
Coordinates:
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
    spatial_ref  int64 8B 0
Data variables:
    red          (y, x) uint8 16B dask.array<chunksize=(4, 4), meta=np.ndarray>
Attributes:
    AREA_OR_POINT:  Area
```

The reopened object is an xarray `Dataset`, with one variable per band.
The file preserves this raster's grid, `uint8` values, variable name, and nodata.
`to_cog` returns a list of `Scene`s, one per date. Each scene maps an asset key
to a `RasterFile`: the file's `path` and the `header` it states, its bands and
date, without pixels. A file holding one band is keyed by that band, and a file
holding several is keyed `"image"`. `gs.open_raster(scenes)` reads them back as
one raster.

Readers pick the format from the path suffix:

| Object | Reader | Writers |
| --- | --- | --- |
| Raster: xarray `Dataset` | `gs.open_raster(path)` | `.gs.to_cog`, `.gs.to_zarr`, `.gs.to_netcdf` |
| Tree: xarray `DataTree` | `gs.open_tree(paths)` | `.gs.to_cog`, `.gs.to_zarr`, `.gs.to_netcdf` |
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
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
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
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B ...
Data variables:
    red          (y, x) uint8 16B ...
```

Zarr and NetCDF writers return a destination `UPath`, which readers accept.
`chunks={}` opens Dask arrays, so arithmetic stays lazy; writing computes the
pixels. See [lazy data](../concepts/lazy-data.md).

Close file-backed objects when you finish:

```python
reopened.close()
zarr_scene.close()
netcdf_scene.close()
```

Time-series file templates, tree layouts, suffix rules and remote storage are in
[Storage and file layouts](../concepts/storage.md).
