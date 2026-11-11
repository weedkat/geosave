# Prepare data from existing files

Open a raster, select a region, calculate a feature, and save a georeferenced
result. Start here when you already have files; use
[download imagery](download-imagery.md) when you need to find them first.

**You need:** the [base installation](../install.md). The example uses the
[sample scene](../first-raster.md#open-the-sample), but you can substitute your
own image with red and near-infrared bands.

## Open and inspect

```python
import geosave as gs

scene = gs.open_raster("sample-scene.tif", chunks={"y": 128, "x": 128})
list(scene.data_vars)
```

```text
['red', 'green', 'blue', 'nir']
```

The file supplies the grid and band metadata. Opening builds lazy arrays and
leaves stored values unchanged. A file without band names may open with names
such as `band_1`; confirm their meaning before renaming them with xarray.

## Select before calculating

```python
patch = scene.isel(y=slice(0, 128), x=slice(0, 128))
ndvi = gs.features.ndvi(patch, nir="nir", red="red").rename("ndvi")
ndvi.gs.plot(cmap="YlGn", clim=(-0.2, 0.9), xlabel="")
```

`isel` selects pixel positions and preserves their coordinates. It does not
resample. Use `.gs.crop(geometry)` for a region described in ground coordinates,
or `.gs.reproject(target, resampling=...)` when you need a different grid.

The NDVI function decodes declared nodata and packing before calculation. For
your own arithmetic, call `.gs.mask_and_scale()` when the source has packed
values. Preserve categorical label codes instead of treating them as imagery.

## Save and check the result

```python
files = ndvi.gs.to_cog("prepared", format_paths="ndvi.tif")
result = gs.open_raster(files[0])
result.gs.geobox == patch.gs.geobox
```

```text
True
```

Writing computes the selected pixels. `prepared/ndvi.tif` can be reopened in
Python or a GIS; its extent is the cropped region. Existing destinations are
protected unless you pass `overwrite=True`.

```python
result.close()
scene.close()
```

## Choose another format

| Data | Read | Write |
| --- | --- | --- |
| GeoTIFF | `gs.open_raster(path)` | `raster.gs.to_cog(folder, format_paths="scene.tif")` |
| Multidimensional raster | `gs.open_raster(path)` | `raster.gs.to_zarr(path)` or `.gs.to_netcdf(path)` |
| Vector features | `gs.open_vector(path)` | `features.gs.to_geoparquet(path)` |

Raster readers return native xarray objects; a store with child groups returns
a DataTree. Vector readers return a GeoDataFrame. See [I/O options](../basics/io.md)
for formats and [storage](../concepts/storage.md) for remote paths and credentials.

## Prepare labels and larger collections

- **Polygons to labels:** [rasterize vector features](../basics/vectors.md) on
  the image's grid, choosing a class column and an explicit background code.
- **Images with different grids:** [align before combining](../concepts/raster-grids.md).
  Use nearest-neighbor resampling for class labels.
- **Many scenes:** [publish a catalog](catalogs-and-cuts.md) when you need to
  select observations by footprint, date, or collection.
- **More data than memory:** [process large rasters](large-data.md) by selecting
  regions and bands before computing.

**Next:** [make training chips](../quickstart.md) or
[explore your prepared data](explore.md).
