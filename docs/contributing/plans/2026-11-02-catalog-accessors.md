# Catalog accessor ownership

Use native GeoDataFrames and Series with `GeoCatalog` and `GeoCatalogItem`
accessors. A catalog can contain ordinary spatial records; producing STAC
Items requires the additional STAC fields. Series records do not inherit a
GeoDataFrame's CRS, so this change adds no CRS-dependent record operations.

Implementation plan:

- Rename the frame accessor and its source-mirrored tests; update active
  exports, consumers and documentation without compatibility aliases.
- Register the item accessor directly. Accept spatial records and validate
  STAC fields when converting; test boundary records and ordinary columns.
- Consolidate scalar/table STAC representation in vector I/O's `_stac.py`.
  Keep `create_catalog` as the public constructor delegating to this conversion.
- Move raster-to-catalog publication into `io/raster/catalog.py`, keeping
  xarray accessors as the public entry points.
- Verify native objects, lazy raster reads, metadata and persistence with
  focused tests, type checks and lint. Review import laziness and inspect
  surviving references. Preserve unrelated existing working-tree edits.

Search return types, explorer feature expansion, mixed-record publication,
and automatic preservation of a frame's CRS on scalar Series are out of scope.

Validation:

- Affected geodata, explorer, labels, model, registration and import tests:
  759 passed, 6 failed, 24 deselected. All six ModelSpec failures reproduce
  with the task-start source: two tests still index/count accessors, and four
  preparation tests pass the retired `target` argument to the Item reader.
- Source-mirrored publication and record tests after relocation: 8 passed.
- Targeted BasedPyright: no errors or warnings. Ruff passes across source,
  tests and scripts. API reference freshness and Zensical build pass.
- Independent review found no actionable issues; importing core leaves STAC,
  torch and visualization unloaded.

Breaking API changes: `GeoVector` becomes `GeoCatalog`; `GeoCatalogRow`
becomes `GeoCatalogItem`. The native containers and DataTree loading policy
are unchanged. No compatibility aliases are provided.
