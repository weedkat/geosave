# Raster path expansion Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the mixed layout pipeline with explicit path expansion and owner-local writing.

**Architecture:** paths.expand_paths handles one Dataset's template only.
GeoDataTree owns groups/global collision checks; GeoTIFF owns format preflight;
Scene owns eager/deferred writing; STAC translation owns observation validity.

**Tech Stack:** xarray, GeoPandas, UPath, Dask, rasterio, PySTAC.

**Spec:** ../specs/2026-11-01-raster-path-expansion.md

## Global constraints

- Keep public writer signatures, results and documented destination semantics.
- No notebooks, compatibility aliases, new dependencies or ModelSpec changes.
- Preserve lazy pixels, coordinates, grids and actual persisted metadata.
- Work in the supplied dirty checkout; preserve unrelated edits and do not commit.

## Review focus

- Expand all groups and reject collisions before writing any group.
- Validate every COG selection before eager or deferred execution.
- A store without time placeholders stays whole without per-date slicing.
- Band splitting must retain nonspatial observation coordinates and bounds.
- Remote paths retain filesystem options; delayed writes read no pixels early.

## Task 1: Dataset path expansion

Files: new io/raster/paths.py and tests/geodata/io/raster/test_paths.py.

Interface: expand_paths(raster: xr.Dataset, template: Location, *, collection:
str | None = None) -> dict[UPath, xr.Dataset].

- [x] Add and run failing tests for destinations, lazy selections, whole stores,
  metadata/coordinate retention, collisions and filesystem options.
- [x] Implement the single-Dataset expansion without format or Scene imports.
- [x] Run the path tests and a small expansion smoke before integration.

## Task 2: Writing, format and publication owners

Files: io/raster/file.py, geotiff.py; core/raster/{dataset,datatree,base}.py;
stac/translate/item.py and matching tests.

Interfaces: Scene.write(files, write, *, collection=None, compute=True) ->
list[Scene]; GeoDataTree._expand_paths(template, storage_options) ->
dict[str, dict[UPath, xr.Dataset]]; geotiff.validate(ds, path) -> None;
stac.translate.item.validate_observation(raster) -> None.

- [x] Add/run failing tests for global collisions, later invalid groups before
  any writes, Scene deferred execution and observation metadata retention.
- [x] Implement the owners above, update public writer bodies and to_catalog,
  and remove Planned/PlannedScene/layout/save without forwarding aliases.
- [x] Move existing layout behavioral tests to their source owners; preserve
  distinct round-trip, header, format and delayed-execution protections.
- [x] Run raster, STAC and catalog regressions with appropriate local runtime.

## Task 3: Documentation and final verification

Files: io package exports/docstrings, docs/reference/io.md, architecture,
catalog docs and generated API references.

- [x] Update active documentation and regenerate API references; preserve older plans.
- [x] Run lint, typing, docs build/coverage, guide smoke and complete pytest suite.
- [x] Review the task-specific changes against the saved working-tree baseline.
- [x] Record exact results and unresolved preceding-refactor failures.

## Execution notes

The user approved the concrete approach and requested planning plus execution.
Continue in this checkout, whose uncommitted refactor is the required baseline.
A copy of the relevant starting files is in /tmp/geosave-paths-baseline for
task-specific review. No repository history or unrelated source will be reset.


Review notes: independent review found missing no-template COG time-value
preflight. Added NaT/malformed-date tree regressions, observed metadata loss
and partial writes, and fixed GeoTIFF validation using its existing time
coverage translator plus explicit NaT rejection. A separate alias-path test
caught relative/absolute destinations targeting the same file; canonical
addresses now detect the collision before any group writes. All-file fill
validation likewise prevents earlier valid groups from being written first.

Ruling: Scene keeps ownership of grouping and pending writes to preserve the
existing public Scene return contract. Path expansion has no Scene or format
knowledge; STAC validity lives in the Item translator. Store destination-folder
semantics are intentionally unchanged, rather than adding another public API
migration to this internal simplification.


Final verification: 1365 passed, 38 failed, 37 deselected, 3 xfailed. All 38
failed test names exactly match the recorded pre-task full run. Ruff, targeted
BasedPyright (zero errors/warnings), docs build/API coverage (1142/1142), and the
catalog guide smoke passed. See the adjacent validation report for commands,
exact failures and compatibility limits. No commits or branch changes.
