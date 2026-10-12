# GeoSave Engine

GeoSave connects geospatial data preparation to PyTorch Lightning training.
Images stay in xarray, vectors stay in GeoDataFrames, and model graphs stay in
PyTorch. You can use each part from ordinary Python without creating a workspace
or running a Prefect workflow.

GeoSave adds geospatial accessors, metadata-aware raster operations, STAC
catalogs and readable training windows. It also provides model composition,
input specifications, Lightning tasks, and native model releases.

## Start here

1. [Install GeoSave](install.md).
2. [Build your first raster](first-raster.md): create an image, compute NDVI,
   save it, and inspect the reopened result.
3. [Prepare training chips](quickstart.md): put imagery and labels in a catalog,
   cut windows, and read a chip as tensors.
4. [Train a model](guides/training.md): supply a torch Dataset to a Lightning
   Module and DataModule.

Examples show their output so you can follow the results without executing the
code. Python snippets use notebook or interactive displays: the final expression
produces the shown result. Training scripts show their log output. Snippets
within a page share their setup; remote examples state when they need network
access or credentials.

## Find your task

| I want to… | Read |
| --- | --- |
| Open, process, or save imagery | [Raster data](basics/rasters.md), [I/O](basics/io.md) |
| Work with polygons or rasterize labels | [Vector data](basics/vectors.md) |
| Describe class labels, nodata, or reflectance scaling | [Raster metadata](basics/attrs.md) |
| Search imagery in a STAC catalog | [STAC data](guides/stac.md) |
| Build a catalog and select training windows | [Catalogs and windows](guides/catalogs-and-cuts.md) |
| Process data larger than memory | [Lazy raster processing](guides/large-data.md) |
| Compose a model or describe its input data | [Models](guides/models.md), [model inputs](guides/model-spec.md) |
| Save or publish a trained model | [Model releases](guides/release.md) |
| Create a project or run an ingestion job | [Workspaces](guides/workspaces.md), [workflows](guides/workflows.md) |

## Understand the behavior

The practical guides cover the rules needed for each example. Read
[Concepts](concepts/index.md) when you need details about grids, metadata,
computation, window selection, model inputs, or training behavior.

Use the [API reference](reference/index.md) to look up a function's arguments,
return value, errors, and supported shapes. Contributor details live in
[Architecture](architecture.md).

GeoSave is in Alpha. The docs describe the current implementation; examples of
features that remain unimplemented are labeled explicitly.
