# Data API

GeoSave works with xarray objects and GeoDataFrames. Importing `geosave`
registers the `.gs` accessors. These pages document GeoSave additions; use the
xarray and GeoPandas reference for their native methods.

| API | Purpose |
| --- | --- |
| [Raster conversions and time](rasters.md), [regions](region.md) | Assemble arrays, describe coverage, and work with dates. |
| [Dataset](dataset-accessor.md), [DataArray](dataarray-accessor.md), [DataTree](datatree-accessor.md) | Raster accessor methods, including inherited methods. |
| [Vectors](vectors.md) | GeoDataFrame accessor methods. |
| [I/O](io.md) | Readers, writers, format options, and storage. |
| [Chips](chips.md) | Read raster windows and merge predictions. |
| [Transforms](transforms.md) | Grids, cropping, combining, time, masks, and vector conversion. |
| [Features](features.md) | Spectral indices and cloud or shadow masks. |
| [Metadata](attrs.md) | Validated metadata groups. |
| [STAC](stac.md), [catalog operations](catalog.md) | Discover observations, publish assets and load native grids. |
| [Errors and warnings](errors.md) | GeoSave conditions callers can handle. |

`gs.Dataset`, `gs.DataArray`, `gs.DataTree`, and `gs.GeoDataFrame` are native
ecosystem types, not extra containers.
