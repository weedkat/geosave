# Basics

A raster is an image whose pixels have locations. Its bands hold measurements
such as red light, near-infrared light, or elevation. A vector describes shapes,
such as field boundaries, as points, lines, or polygons.

GeoSave lets you work with both through ordinary Python objects. Begin with
[Your first raster](../first-raster.md) for a complete example using a small
image you create yourself.

## Learn in this order

| Guide | What you will learn | Result |
| --- | --- | --- |
| [Raster data](rasters.md) | Create, select, calculate, plot, and group bands. | Image, band, or group of images. |
| [Describe raster values](attrs.md) | Distinguish missing pixels, scaled measurements, and class codes. | Values with explicit meaning. |
| [Read and save data](io.md) | Choose a reader and writer for local or remote files. | Python objects on read; files on write. |
| [Vector data](vectors.md) | Read polygons and convert between vector and raster labels. | Shapes or a label image. |
| [Images to training chips](../quickstart.md) | Save images and labels, then read one small region for a model. | Image tensors and a target. |

## Know the objects

```python
import geosave as gs

scene = gs.open_raster("scene.tif")
red = scene["red"]
patch = scene.isel(y=slice(0, 256), x=slice(0, 256))
```

This fragment assumes a georeferenced file containing a band named `red`.
`scene` and `patch` are xarray Datasets; `red` is a DataArray. Selection keeps
the chosen pixels' coordinates. GeoSave registers `.gs` methods on native
objects, so normal xarray and GeoPandas code continues to work.

| Object | Represents | Everyday use |
| --- | --- | --- |
| `xarray.DataArray` | One variable with dimensions and coordinates. | `scene["red"]` |
| `xarray.Dataset` | Named variables on a raster grid. | `gs.open_raster(path)` |
| `xarray.DataTree` | Named raster groups, which can have different grids. | `gs.tree({"optical": scene, "label": labels})` |
| `geopandas.GeoDataFrame` | Geometries and columns; also catalogs or sample tables. | `gs.open_vector(path)` |
| `torch.Tensor` | Pixel values for a model, without geospatial coordinates. | `scene.gs.to_tensor()` |

Reading can defer pixel work. Converting to NumPy or tensors reads the
selected pixels; writing normally computes the output. Choose bands and the
region first. See [lazy execution](../concepts/lazy-data.md) for details.

Continue with [activities](../activities.md) for complete tasks or
[advanced behavior](../concepts/index.md) for detailed contracts.
