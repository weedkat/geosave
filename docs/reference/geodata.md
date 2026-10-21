# Data API

GeoSave works with xarray objects and GeoDataFrames. Importing `geosave`
registers the `.gs` accessors. These pages document GeoSave additions; use the
xarray and GeoPandas reference for their native methods.

| API | Purpose |
| --- | --- |
| [Constructors](rasters.md), [anchors](anchor.md) | Create rasters, trees, and targets. |
| [Dataset](dataset-accessor.md), [DataArray](dataarray-accessor.md), [DataTree](datatree-accessor.md) | Raster accessor methods, including inherited methods. |
| [Vectors](vectors.md) | GeoDataFrame accessor methods. |
| [I/O](io.md) | Readers, writers, format options, and storage. |
| [Transforms](transforms.md) | Grids, cropping, combining, time, masks, and vector conversion. |
| [Features](features.md) | Spectral indices and cloud or shadow masks. |
| [Metadata](attrs.md) | Validated metadata groups. |
| [STAC](stac.md), [catalog operations](catalog.md) | Item catalogs, image sources, and readable windows. |
| [Errors and warnings](errors.md) | GeoSave conditions callers can handle. |

`gs.Dataset`, `gs.DataArray`, `gs.DataTree`, and `gs.GeoDataFrame` are native
ecosystem types, not extra containers.
