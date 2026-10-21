# Advanced: concepts and behavior

These pages explain the rules behind the practical examples. Read the topic
related to your next operation; you do not need all of them before using GeoSave.

| Topic | Questions it answers |
| --- | --- |
| [How the library works](data-flow.md) | Which objects enter and leave acquisition, preparation, training, and release? |
| [Raster grids](raster-grids.md) | How do coordinates, CRS, resolution, reprojection, and alignment relate? |
| [Metadata and pixel values](metadata.md) | What do nodata, scale factors, class legends, and metadata joins mean? |
| [Lazy execution](lazy-data.md) | Which calls build a graph, and which read or write pixels? |
| [Storage and file layouts](storage.md) | Which suffixes, file templates, tree layouts and remote options exist? |
| [The interactive explorer](explorer.md) | How do layers, catalogs and large rasters behave on the map? |
| [Catalogs and windows](catalog-windows.md) | What does a table row describe, and how does it find its pixels? |
| [Model input contracts](model-inputs.md) | How do preprocessing, references, context, and tensor transforms fit together? |
| [Training contracts](training.md) | What must a Dataset return, and how do loss, targets, augmentation, and metrics behave? |

Use the [API reference](../reference/index.md) for argument details. Package
ownership and contributor guidance live in [Architecture](../contributing/architecture.md).

For a specific goal, choose an [activity](../activities.md). For a first
introduction to native objects, use [Basics](../basics/index.md).
