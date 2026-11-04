# Native raster I/O Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent ownership slices and a final integrated review.

**Goal:** Replace Scene/RasterFile/assemble with native inventories, transforms and resource-owning loaders.

**Architecture:** Format writers return paths; accessors return lists/group
mappings or native Delayed results. Loaders own resources; transform.combine
combines opened rasters. STAC translates actual files and captured publication
coordinates without synthetic headers.

**Tech Stack:** xarray, Dask, UPath, rasterio, PySTAC, GeoPandas, PyArrow.

**Spec:** ../specs/2026-11-01-native-raster-io.md

## Global constraints

- Preserve the shared dirty checkout and unrelated edits; do not commit.
- No new dependencies, compatibility aliases, notebook edits or ModelSpec redesign.
- Native containers, lazy reads, lossless applicable metadata and deterministic cleanup.

## Review focus

- Closing the final Dataset/DataTree must close all opened assets exactly once.
- A later open/metadata/combine failure must close earlier sources.
- Deferred writer construction must not execute pixel tasks or create files.
- Publication must retain properties/bounds absent from the raster file format.
- Plain mappings must distinguish group inventories from explicit Item assets.

## Task 1: Native read and combine ownership

Files: transform/combine.py, io/openers.py, io/raster/stac.py (read path only),
tests/geodata/io/test_openers.py, tests/geodata/transform/test_combine.py;
remove assemble.py and redistribute its meaningful tests.

- [ ] Reproduce close-callback loss and add failure-cleanup regressions.
- [ ] Move native combination to combine_rasters(rasters, geobox=None, resampling=...).
- [ ] Load paths and STAC assets under ExitStack, transferring closure to results.
- [ ] Remove fake-header replay and move STAC band naming to its proper translator.
- [ ] Verify native direct/STAC reads, laziness and resource lifetime.

## Task 2: Native write results

Files: io/raster/{geotiff,zarr,netcdf}.py and core/raster/{dataset,dataarray,datatree}.py;
matching writer/accessor tests. Coordinate with Task 3 on base.to_catalog.

- [ ] Add failing native-result/deferred-write behavior tests.
- [ ] Return actual UPaths; remove synthetic-header rereads after writes.
- [ ] Return lists/group dictionaries from accessors and native Delayed on compute=False.
- [ ] Preserve preflight and migrate tests to actual readback values/metadata.
- [ ] Verify each format and grouped/deferred writes.

## Task 3: Actual-file STAC publication

Files: stac/{create,translate/item,translate/asset}.py, io/raster/stac.py
(describe_files added after read edits), core/raster/base.py, exports, flow label
I/O, shared test helpers and STAC/publication tests; remove file.py.

- [ ] Add failing source-coordinate preservation and actual-header publication checks.
- [ ] Translate asset paths plus actual Datasets; describe paths under ExitStack.
- [ ] Capture source coordinates by written asset in to_catalog; preserve Collections.
- [ ] Migrate callers and remove wrapper exports/module without aliases.
- [ ] Run full write/catalog/persist/load/close smoke and STAC regressions.

## Task 4: Integration and verification

- [ ] Review ownership slices against /tmp/geosave-native-io-baseline.
- [ ] Update active docs and API reference, preserving historical plans/specs.
- [ ] Run lint/type/docs checks, guide smoke, focused tests and complete suite.
- [ ] Independently review the integrated flow; fix important findings with tests.
- [ ] Record exact results, breaking changes and unchanged baseline failures.

## Execution notes

The user's approval covers the complete I/O redesign described above, including
writer-return breaking changes. Parallel slices have disjoint owned files;
the main implementer owns STAC publication and integration. Existing read/write
modules and tests were copied to /tmp/geosave-native-io-baseline for comparison.
