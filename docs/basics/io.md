# Read and save data

Save a named band as a Cloud Optimized GeoTIFF, then check its grid and values.
This example creates a small raster locally and needs only the base package.

```python
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

import geosave as gs
from geosave.geodata import io

root = Path(tempfile.mkdtemp())
grid = GeoBox.from_bbox(
    (500000, 5000000, 500040, 5000040), crs="EPSG:32633", resolution=10,
)
pixels = np.arange(1, 17, dtype="uint8").reshape(4, 4)
scene = xr.Dataset({"red": (("y", "x"), pixels)}).assign_coords(xr_coords(grid))
scene = scene.gs.write_nodata(0)
path = root / "scene.tif"
io.write_cog(scene, path)
reopened = gs.open_raster(path)
list(reopened.data_vars), reopened.gs.geobox == scene.gs.geobox
```

```text
(['red'], True)
```

```python
np.array_equal(reopened["red"].values, pixels), reopened["red"].gs.nodata
```

```text
(True, 0)
```

`io.write_cog` returns the destination UPath and preserves variable names as
band metadata. It refuses an existing file unless `overwrite=True` is passed.
Opening a TIFF returns a Dataset with one variable per band; external TIFFs
without GeoSave identity tags use `band_1`, `band_2`, etc.

## Several bands in one TIFF

For bands on one grid, use `io.write_cog(scene[["red", "nir"]], "image.tif")`.
Variables keep their names and band metadata; the TIFF shares one stored dtype,
nodata value and pixel grid. Select one observation before writing a time series.

For dated observations or assets on different grids, use
[catalog publication](../guides/catalogs-and-cuts.md). Dataset `.gs.to_cog()`
remains unavailable; the explicit format writer handles one file.

## Choose a format

| Data and goal | Write | Return value |
| --- | --- | --- |
| Named bands on one grid at one instant | `io.write_cog(dataset, "image.tif")` | One file `UPath` |
| Dataset or DataTree, including multiple dates | `raster.gs.to_zarr("scene.zarr")` | One store `UPath` |
| Dataset or DataTree in one scientific file | `raster.gs.to_netcdf("scene.nc")` | One file `UPath` |
| One COG per staged asset | `staged.to_cog("assets")` on `RasterStac` | Native Items |
| COG assets plus an Item table | `dataset.gs.to_catalog("assets")` | GeoDataFrame |
| Vector features or STAC Item table | `rows.gs.to_geoparquet("rows.parquet")` | One file `UPath` |

The [I/O reference](../reference/io.md#saved-raster-files) lists deferred-write
returns and format options. A catalog table describes raster files; save the
table separately with `.gs.to_geoparquet(...)`.

## Save multidimensional data

```python
store = scene.gs.to_zarr(root / "scene.zarr")
zarr_scene = gs.open_raster(store)
path = scene.gs.to_netcdf(root / "scene.nc")
netcdf_scene = gs.open_raster(path)
zarr_scene.gs.geobox == grid, netcdf_scene.gs.geobox == grid
```

```text
(True, True)
```

Zarr stores chunked arrays; NetCDF stores scientific arrays in one file.
DataTree writers preserve groups. `gs.open_raster` always returns a Dataset:
the root group by default, or a selected group with `group="scene"`.
Use `gs.open_raster_tree(store)` to read the whole hierarchy as a DataTree,
including a root-only tree when there are no child groups.

Raster readers select their format from the suffix. The default `chunks={}`
uses Dask arrays; `chunks=None` uses backend indexing without Dask. Opening
reads metadata and defers pixels where the backend supports it.
[Lazy execution](../concepts/lazy-data.md) explains when work runs.

```python
reopened.close()
zarr_scene.close()
netcdf_scene.close()
```

## Vector files and remote paths

`gs.open_vector("fields.parquet")` returns a GeoDataFrame. Write it with
`.gs.to_geoparquet(...)`, `.gs.to_geojson(...)`, or `.gs.to_geopackage(...)`.
See [vector data](vectors.md) for a complete round trip.

Locations accept strings, `Path`, or `UPath`. See
[storage and file layouts](../concepts/storage.md) for suffixes, tree paths,
remote credentials, and the difference between lazy pixels and windowed downloads.
