# Basics

Start with a small real image. Learn one operation at a time, then combine
the pieces in your own project.

| Step | You will do |
| --- | --- |
| [Install GeoSave](install.md) | Choose the base package or ML extras. |
| [Your first vegetation map](first-raster.md) | Open, plot, calculate NDVI and save. |
| [Raster data](rasters.md) | Choose bands for a vegetation map. |
| [Vector data](vectors.md) | Work with features in a GeoDataFrame. |
| [Describe values](attrs.md) | Read nodata, scaling and class labels. |
| [Read and save data](io.md) | Choose a format and reopen its output. |
| [Store imagery as a catalog](catalog.md) | Publish observations as COGs with a searchable index. |
| [Images to training chips](training-chips.md) | Keep imagery and labels named through tensor conversion. |
| [Explore a map](explore.md) | Inspect raster and vector layers interactively. |

GeoSave adds `.gs` methods to native xarray and GeoPandas objects. Ordinary
selection, arithmetic and other ecosystem operations remain available.
The [visual xarray guide](../concepts/xarray-structure.md) explains Dataset,
DataArray, dimensions and coordinate labels with inspectable examples.

Continue with [Advanced](../advanced/index.md) for larger workflows,
[Concepts](../concepts/index.md) for behavior, or [API Reference](../reference/index.md)
for exact arguments. [Examples](../examples.md) shows complete small tasks.
