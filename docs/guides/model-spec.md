# ModelSpec redesign

ModelSpec is disabled while its workflow design is rebuilt. Constructing it or
calling `ModelSpec.load(...)` raises `NotImplementedError` before reading a file
or executing an operation.

The old workflow graph, reference declarations, YAML execution machinery,
`model.ingestion` and `model.processing` have been removed. No replacement
recipe schema is supported yet.

Geospatial template DataModules, raster Datasets and preparation scripts that
require ModelSpec are paused. Model stages and training Modules remain usable
with caller-owned DataLoaders supplying their sample contract. Use native
GeoSave/xarray operations explicitly for preparation.

Model releases save registered stage construction and weights, without a
processing recipe. See [releases](release.md).
