# Vector and raster catalog API Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify GDF accessor ownership and expose raster publication as `to_catalog`.

**Architecture:** One stable GeoVector owns GDF operations; format modules own
persistence. Raster publication composes existing layout, writers and STAC
translators, returning a native GeoDataFrame.

**Tech Stack:** GeoPandas, xarray, PyArrow, PySTAC, stac-geoparquet.

**Spec:** ../specs/2026-11-01-vector-catalog-api.md

## Global Constraints

- Preserve unrelated uncommitted work in the supplied checkout; no commits.
- No notebooks, compatibility aliases, new dependencies or ModelSpec redesign.
- No automatic source reuse without a verifiable source contract.

## Review Focus

- Column mutation must not leave catalog-specific cached accessor behavior.
- Catalog export must retain Collections, bbox, asset metadata and href portability.
- Filename collisions must fail before writing any pixels.
- DataTree groups can have different grids and dates.
- A transformed Dataset must write its current pixels rather than reference its old source.

## Task 1: One vector accessor

Files: core/vector/{vector,item,sample}.py, core/register.py, core/ctor/catalog.py,
package exports, io/vector/{geoparquet,stac_geoparquet}.py, explorer callers;
tests/geodata/core/vector and tests/geodata/core/test_register.py.

Produces: GeoVector for all frames; GeoCatalogRow for Item rows;
to_stac_geoparquet(path, collections=None, overwrite=False, storage_options=None).

- [x] Add behavior tests for post-mutation operations and explicit export separation; observe failures.
- [x] Move catalog and ordinary writer methods onto GeoVector; remove subclass dispatch and redundant classes.
- [x] Preserve metadata merging and validate Item operations when called; update readers and visualization.
- [x] Move behavioral tests to their source-owning test modules and run vector/persistence checks.

## Task 2: Raster to_catalog

Files: core/raster/base.py; tests/geodata/core/raster/test_base.py.

Produces: to_catalog(path=None, *, collection=None, id_format=existing default,
datetime=None, overwrite=False, storage_options=None, **options) -> GeoDataFrame.

- [x] Test templated Dataset and DataTree round trips, current transformed values,
  Collection retention, collision preflight and explicit path=None failure; observe failures.
- [x] Compose layout/save with the suffix-selected writer and existing STAC constructors.
- [x] Run new tests and raster layout/writer regression tests.

## Task 3: Callers, docs and verification

Files: source/test callers, public API guides and generated reference pages.

- [x] Migrate removed accessor imports and explicit STAC writes; preserve ordinary vector writes.
- [x] Document public API and limitations, regenerate API reference using the existing script.
- [x] Run focused integration tests, lint/type checks as applicable and the project suite.
- [x] Review the final changes and report exact failures or verification exclusions.

## Execution notes

Ruling: Work in the supplied checkout because the refactor being extended is
uncommitted across source/tests/docs; a new worktree would omit its required
baseline. Preserve those edits and do not commit or change branches.

Ruling: The user explicitly requested planning and implementation together;
execute the concrete design above in this turn rather than introduce another
confirmation gate. Source reuse remains explicitly unsupported.


## Review and verification notes

A read-only independent review checked accessor registration, Item conversion,
Arrow readback, raster template dispatch and export semantics. Its two important
findings were fixed with failing-then-passing behavioral tests: ordinary
GeoJSON/GeoPackage output now refuses Item metadata loss, and store publication
requires locatable grids before creating any assets.

Focused verification: 238 passed, 9 deselected (the configured slow/integration
markers). The complete catalog guide ran as a smoke test: three Items, two
sample groups and readable chips. Ruff passes; targeted BasedPyright including
registration typing reports zero errors and one existing lazy-export warning.
Zensical builds and the generated API reference reports 1149/1149 rendered
coverage. Full-suite results and exact failures are recorded separately in the
validation report beside this plan.

Ruling: ModelSpec preparation and task templates still consume removed sample
and chip APIs from the preceding refactor. Preserve the requested no-redesign
scope; report these failures without restoring compatibility machinery.
