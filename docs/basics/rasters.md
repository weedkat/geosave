# Work with rasters

Open the [sample scene](../first-raster.md#open-the-sample) and select two bands:

```python
import geosave as gs

scene = gs.open_raster("sample-scene.tif")
image = scene[["red", "nir"]]
list(image.data_vars), (image.sizes["y"], image.sizes["x"])
```

```text
(['red', 'nir'], (337, 267))
```

A raster stores measurements in pixels with geographic locations. `scene` is
an xarray **Dataset**; each named variable holds a band. Its `x` and `y`
coordinates locate pixel centres, and `.gs.geobox` describes its CRS, pixel
size, shape and placement. GeoSave adds geospatial operations through `.gs`.

**You need:** the base installation and the sample TIFF. These examples run
offline after downloading it.

## Select one band or several

```python
red = scene["red"]
image = scene[["red", "nir"]]
type(red).__name__, type(image).__name__
```

```text
('DataArray', 'Dataset')
```

A **DataArray** holds one array with named axes. A Dataset holds named arrays
on shared coordinates. Use a list when you want to keep a Dataset, even for
one band: `scene[["red"]]`.

## Select pixels

```python
patch = scene.isel(y=slice(0, 128), x=slice(0, 128))
patch.sizes["y"], patch.sizes["x"]
```

```text
(128, 128)
```

`isel` selects pixel positions and keeps their coordinates. It changes the
covered area without resampling. For a polygon in ground coordinates, use
[`.gs.crop(...)`](../reference/transforms.md#grid-masks-and-values).

## Calculate and plot an index

```python
ndvi = gs.features.ndvi(patch, nir="nir", red="red").rename("ndvi")
ndvi.gs.plot(cmap="YlGn", clim=(-0.2, 0.9), xlabel="")
```

![Natural color and vegetation index from the sample scene](../assets/vegetation.jpg)

NDVI compares near-infrared and red reflectance. The feature function decodes
declared missing values and scaling before calculation. See
[describe raster values](attrs.md) before writing your own arithmetic.
`xlabel=""` avoids a place-name lookup, keeping this plot offline.

Plotting reads pixels. Selection and the index calculation can remain lazy.
For interactive inspection, use [the map explorer](../guides/explore.md).

## Save the result

```python
from pathlib import Path

path = Path("output/ndvi.tif")
path.parent.mkdir(parents=True, exist_ok=True)
ndvi.gs.to_cog(path)
result = gs.open_raster(path)
list(result.data_vars), result.gs.geobox == patch.gs.geobox
```

```text
(['band_1'], True)
```

A Cloud Optimized GeoTIFF preserves this result's grid. Native array export
does not add GeoSave variable identity tags, so the reader uses `band_1`.
Writing computes the result and replaces an existing destination.
For several compatible bands in one TIFF, convert the selected Dataset to a
band array first: `image.gs.to_array().gs.to_cog("output/image.tif")`.
[Read and save data](io.md) explains other formats and return values.

```python
result.close()
scene.close()
```

Keep file-backed sources open until all work depending on them finishes.

## When you need more than one raster

An xarray **DataTree** holds Datasets in a hierarchy. It can keep different
grids or observations separate. GeoSave's STAC operations use a specific
collection → Item → asset layout; you do not need it for this example.
See [catalog creation and loading](../guides/catalogs-and-cuts.md) when you
need to organize dated files.

**Next:** [describe pixel values](attrs.md), [save other formats](io.md), or
[make training chips](../quickstart.md). For constructing a raster from new
pixels, see [raster grids](../concepts/raster-grids.md).
