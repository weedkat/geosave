# Basics

A raster is an image whose pixels have locations; its bands hold measurements
such as red light or elevation. A vector describes shapes, such as field
boundaries. GeoSave works with both as ordinary Python objects, so start with
[Your first raster](../first-raster.md).

| Guide | You will learn | Result |
| --- | --- | --- |
| [Raster data](rasters.md) | Create, select, calculate and plot bands. | Image, band or group. |
| [Describe raster values](attrs.md) | Tell missing pixels, scaling and class codes apart. | Values with explicit meaning. |
| [Read and save data](io.md) | Pick a reader and writer for a file. | Python objects; files. |
| [Vector data](vectors.md) | Read polygons and convert to and from pixels. | Shapes or a label image. |

```python
import geosave as gs

scene = gs.open_raster("scene.tif")      # xarray Dataset, one variable per band
red = scene["red"]                       # xarray DataArray
patch = scene.isel(y=slice(0, 256), x=slice(0, 256))
```

`.gs` methods are added to native xarray and GeoPandas objects, so normal
xarray and GeoPandas code keeps working. Reading defers pixel work; selecting
bands and a region first keeps it small.

Next: [activities](../activities.md) for complete tasks, or
[Advanced](../concepts/index.md) for the rules behind the examples (lazy
execution, storage layouts, the interactive explorer).
