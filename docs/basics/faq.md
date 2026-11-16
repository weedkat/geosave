# Common questions

## What can I use without training a model?

Open and save imagery or vectors, explore maps, calculate indices, align grids,
search satellite observations and prepare samples.
Start with [your first vegetation map](first-raster.md) or [find a task](../advanced/index.md).

## Do I need GeoSave-specific data containers?

Use native xarray DataArray, Dataset and DataTree objects, or GeoPandas
GeoDataFrames. Importing GeoSave adds `.gs` operations to them.
Normal ecosystem operations remain available.

## Which installation do I need?

The base package covers geodata. Add `ml` for tensors and training,
`hub` for model releases, or `flows` for the Prefect prediction flow.
See [installation](install.md) for commands and optional backends.

## Does opening imagery read every pixel?

File readers can defer pixel reads. Plotting, writing and tensor conversion
read the values they need. See [lazy execution](../concepts/lazy-data.md) for
computation boundaries and chunking.

## Can I use my existing project?

Yes. Use GeoSave from a notebook, script or application. Generated
[workspaces](../advanced/training.md) and the
[three-repository pattern](../mlops/index.md) are optional ways to
organize training and releases.

## What goes into a model release?

A registered model's construction settings and weights, plus its pipeline
settings when included. A pipeline reuses input preparation and maps
predictions back to a raster. See [releases](../advanced/release.md).

## Where are the exact arguments and return values?

Use the [API Reference](../reference/index.md). The
[symbol inventory](../reference/public-api.md) is generated from source;
[documentation maintenance](../contributing/documentation.md#automatic-api-reference)
explains the generator and its scope.
