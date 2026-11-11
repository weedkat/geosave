# Native operations and geospatial accessors

```python
import xarray as xr
from geosave.geodata import attrs

# Decode before combining stored representations or reducing observations.
decoded = observations.gs.mask_and_scale()
monthly = decoded.gs.composite("MS", "mean")
april = monthly.sel(time="2024-04")

series = xr.concat(
    decoded_rasters,
    dim="time",
    join="exact",
    combine_attrs=attrs.combine_attrs,
)
```

Ordinary array operations use xarray. Geospatial operations live on `.gs`;
there is no public `geodata.transform` package. Accessors return native xarray
or GeoPandas values. See [raster accessors](rasters.md) for signatures and
[metadata behavior](../concepts/metadata.md) for attrs and encoding contracts.

## Combining and interpolating

```python
merged = xr.merge(datasets, join="exact", combine_attrs=attrs.combine_attrs)
filled = decoded.gs.composite(
    "MS", "mean", dim="time", interpolation="linear",
)
```

Native concat and merge default to `combine_attrs="override"`. The explicit
callback above uses `drop_conflicts` and warns about omitted keys. Native
`join="exact"` checks indexes, not CRS, packing, units or class meanings.
Align grids and decode inputs explicitly; native variable alignment and dtype
promotion replace the removed wrappers' restrictions.

Composite accepts an explicit `dim` (default `"time"`) and optional
`interpolation="linear"` or `"nearest"`. It reduces first, then fills interior
pixel gaps along the resulting coordinate without extrapolation. Lazy output
is rechunked along that dimension only. Observation bounds and properties stay
unchanged by interpolation; reconstructed pixels do not create observations.
Class codes reject interpolation. Use native `.interpolate_na()` separately
when gap limits or extrapolation are needed.

## Grid, masks and values

```python
cropped = scene.gs.crop(region)
chip = scene.gs.crop_to_grid(grid, constant_value=0)
warped = scene.gs.reproject(grid, resampling="bilinear")
clear = scene.gs.mask(scene.scl.isin([4, 5, 6, 7]))
decoded = clear.gs.mask_and_scale()
stored = decoded.gs.scale_and_fill()
```

`crop_to_grid` keeps exact aligned pixel placement and can pad beyond the source
edges; its target must overlap the source. Standalone `extent.pad` is withdrawn.
Use `crop_to_grid` with an explicit target grid when extending a geospatial
raster. Native `.pad()` remains available for ordinary arrays, but it does not
reconstruct affine spatial coordinates for you.

Masking uses each variable's fill declaration unless a fill is supplied.
Reprojection is explicit; categorical bands reject blending kernels.
`scale_and_fill` checks integer conversion lazily and restores CF attrs without
writing a file. Native NetCDF/Zarr writers retain their own encoding behavior.

## Geometry and class colors

```python
burned = features.gs.rasterize(grid, column="class", dtype="uint8", fill=255)
polygons = labels.gs.vectorize(value_name="class")
colors = labels.gs.colorize()
```

Rasterization uses an exact target grid. Polygonization computes pixels because
geometry depends on their values. Colorization uses the band's legend and
preserves lazy raster arrays.

## Migrating removed functions

| Removed interface | Replacement |
| --- | --- |
| `temporal.concat` | `xr.concat(..., dim="time", ...)` with explicit alignment and attrs policy |
| `combine.merge` | `xr.merge(...)`; grid-sensitive assembly stays with its accessor or reader |
| `temporal.interpolate` | `.gs.composite(..., interpolation="linear")` or native `.interpolate_na(...)` |
| `temporal.select` | Native `.sel` on labels, or on `start_datetime`/`end_datetime` for overlap |
| `temporal.composite` | Decode, then `.gs.composite(...)`; `skip_nodata` is removed |
| `packing.mask_and_scale`, `packing.scale_and_fill` | Existing `.gs.mask_and_scale()`, `.gs.scale_and_fill()` |
| `nodata.mask` | `.gs.mask(...)` |
| `nodata.read_mask` | Pass the boolean mask directly to `.gs.mask(...)` |
| `extent.crop`, `extent.crop_to_grid` | `.gs.crop(...)`, `.gs.crop_to_grid(...)` |
| `extent.pad` | Explicit aligned grid with `.gs.crop_to_grid(...)`; no standalone geospatial pad |
| `warp.reproject`, `warp.read_pixels` | `.gs.reproject(...)`, Dataset `.gs.read_pixels(...)` |
| `vector.vectorize`, `vector.rasterize` | Raster `.gs.vectorize(...)`, vector `.gs.rasterize(...)` |
| `color.colorize` | `.gs.colorize()` |

Composite requires decoded inputs, preserves observed periods and appends CF
cell methods. Empty bins have NaT periods and, without interpolation, NaN pixels. All-missing sums
follow native behavior and return zero. Variance drops `units` for the caller
to restate; variance and standard deviation drop `standard_name`. Reduction
does not restore the input's old integer storage encoding.
