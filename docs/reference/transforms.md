# Crop, reproject, mask, and combine

Use xarray for ordinary array operations and `.gs` for geospatial operations.
Accessors return native objects. Choose the concrete reference for your
[Dataset](dataset-accessor.md), [DataArray](dataarray-accessor.md), or
[DataTree](datatree-accessor.md).

## Grid, masks and values

```python title="Adaptation example — use your scene, region and target grid"
cropped = scene.gs.crop(region)
chip = scene.gs.crop_to_grid(grid, constant_value=0)
warped = scene.gs.reproject(grid, resampling="bilinear")
clear = scene.gs.mask(scene["scl"].isin([4, 5, 6, 7]))
decoded = clear.gs.mask_and_scale()
```

| Call | Changes | Pixel execution |
| --- | --- | --- |
| `.isel(y=..., x=...)` | Selects positions without a geometry mask. | Can stay lazy. |
| `.gs.crop(region)` | Selects bounding pixels and masks outside the geometry. | Can stay lazy. |
| `.gs.crop_to_grid(grid)` | Selects or pads aligned pixels onto an exact overlapping grid. | Can stay lazy. |
| `.gs.reproject(grid, resampling=...)` | Resamples onto the chosen grid. | Can stay lazy for Dask inputs. |
| `.gs.mask(valid)` | Fills invalid pixels with nodata or explicit `fill`. | Can stay lazy. |
| `.gs.mask_and_scale()` | Decodes stored missing values and scale/offset. | Can stay lazy. |

Crop and mask need a declared nodata value or explicit fill. Class metadata
prevents reprojection with blending kernels. For aligned imagery and labels,
use the [two-file alignment recipe](../advanced/align-rasters.md).

An exact GeoBox fixes CRS, resolution, shape and placement. Passing only a CRS
lets each source derive its own output grid. Native `.pad()` does not rebuild
affine coordinates; use `crop_to_grid` for geospatial padding.

## Combining and interpolating

```python title="Adaptation example — use prepared compatible rasters"
import xarray as xr

series = xr.concat(decoded_rasters, dim="time", join="exact", combine_attrs="drop_conflicts")
merged = xr.merge(datasets, join="exact", combine_attrs="drop_conflicts")
monthly = series.gs.composite("MS", "mean", interpolation="linear")
```

Native `join="exact"` checks indexes, not CRS, packing, units or class meanings.
Align and decode inputs before combining. `drop_conflicts` drops conflicting
attrs; it does not establish that pixel values are scientifically compatible.

`composite` reduces decoded observations into calendar bins. Each output cell
states that bin, including empty bins, whose pixels remain NaN without
interpolation. Native all-missing sums return zero. Class IDs permit only
`first` and `last`, without interpolation.

Interpolation accepts `linear` or `nearest`, fills interior pixel gaps after
reduction, and does not extrapolate. It keeps observation properties and cells;
it does not invent observations. Lazy interpolation rechunks the selected
dimension into one chunk, increasing per-task memory. Use native
`.interpolate_na()` for separate interpolation with explicit gap limits.
See [temporal reference](rasters.md#composites-and-time-selection).

## Geometry and class colors

Raster `.gs.vectorize(value_name="class")` converts a selected two-dimensional
band to polygons and computes its pixels. `.gs.colorize()` uses class metadata
to create colors and can preserve lazy arrays. Vector `.gs.rasterize()` is
currently unavailable while catalog operations are rebuilt.

Use native xarray operations for selection, concatenation, merging and
interpolation. The older standalone GeoSave transform functions are no longer
part of the public API.
