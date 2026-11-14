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

root = Path(tempfile.mkdtemp())
grid = GeoBox.from_bbox(
    (500000, 5000000, 500040, 5000040), crs="EPSG:32633", resolution=10,
)
pixels = np.arange(1, 17, dtype="uint8").reshape(4, 4)
scene = xr.Dataset({"red": (("y", "x"), pixels)}).assign_coords(xr_coords(grid))
scene = scene.gs.write_nodata(0)
path = root / "scene.tif"
scene["red"].gs.to_cog(path)
reopened = gs.open_raster(path)
list(reopened.data_vars), reopened.gs.geobox == scene.gs.geobox
```

```text
(['band_1'], True)
```

```python
np.array_equal(reopened["band_1"].values, pixels), reopened["band_1"].gs.nodata
```

```text
(True, 0)
```

Array `to_cog` delegates to `.rio.to_raster(driver="COG")` and normally
returns `None`. Keep the destination path yourself. Native writes replace an
existing file. Opening a TIFF returns a Dataset with one variable per band;
without GeoSave publication metadata those variables use `band_1`, `band_2`, etc.

## Several bands in one TIFF

For bands on one grid with compatible units, class metadata and packing, use
`scene[["red", "nir"]].gs.to_array().gs.to_cog("image.tif")`. Select one
observation before writing a time series. Stacking retains shared band
metadata; it cannot preserve different per-band declarations in one array.

To keep bands with different metadata in separate files, use
[tree publication](../guides/catalogs-and-cuts.md). Dataset `.gs.to_cog()` is
unavailable; explicitly choose a band array, a tree of assets, or another format.

## Choose a format

| Data and goal | Write | Return value |
| --- | --- | --- |
| One band or compatible band array at one instant | `array.gs.to_cog("image.tif")` | Native result (normally `None`) |
| Dataset or DataTree, including multiple dates | `raster.gs.to_zarr("scene.zarr")` | One store `UPath` |
| Dataset or DataTree in one scientific file | `raster.gs.to_netcdf("scene.nc")` | One file `UPath` |
| One TIFF per asset of a STAC tree | `tree.gs.to_cog("assets")` | List of file UPaths |
| TIFF assets plus an Item table | `tree.gs.to_catalog("assets")` | GeoDataFrame |
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
DataTree writers preserve groups. `gs.open_raster` returns a DataTree for a
store with child groups and a Dataset for a root-only store.

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
