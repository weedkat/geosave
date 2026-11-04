# Unified GeoParquet Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write features and STAC catalogs through one public to_geoparquet method.

**Architecture:** io.vector.geoparquet.write selects the encoder from frame
metadata at write time. STAC-specific invariants remain with its encoder.
The GDF accessor delegates; readers retain metadata on column selections.

**Tech Stack:** GeoPandas, PyArrow, PySTAC, stac-geoparquet, UPath.

**Spec:** ../specs/2026-11-01-unified-geoparquet.md

## Global constraints

- Preserve unrelated dirty-checkout edits; no commits or branch changes.
- No notebook changes, compatibility aliases, new dependencies or ModelSpec redesign.
- Preserve native containers, Collection metadata, portability and ordinary writer options.

## Review focus

- Missing columns or malformed metadata must not downgrade catalog encoding.
- Selected-column reads retain metadata and reject incomplete catalog writes.
- Metadata removed from complete Item-shaped rows produces a clear failure.
- Catalog compression/options must not be silently ignored or override managed metadata.
- Sample keys remain transient; ordinary features retain their existing writer behavior.

## Task 1: Write-time selection

Files: core/vector/vector.py; io/vector/{geoparquet,stac_geoparquet}.py;
tests/geodata/{core/vector/test_vector,io/vector/test_geoparquet}.py.

Interface: to_geoparquet(path, *, collections=None, overwrite=False, **options)
delegates to geoparquet.write with the same keywords. GeoParquetWriteOptions
continues typing storage and format options; STAC enforces its representation.

- [x] Add/run failing behavior tests for marked catalog round trips, damaged
  metadata/fields, lost markers, sample rejection and codec propagation.
- [x] Implement the dispatcher, central STAC validation/options and one accessor method.
- [x] Remove the second accessor method and migrate source/test callers without aliases.
- [x] Run vector and STAC persistence regressions, checking actual footer and Items.

## Task 2: Selected-column readback

File: io/vector/geoparquet.py and matching test module.

- [x] Add/run a failing read-select-write test proving Collection loss cannot occur.
- [x] Restore footer metadata independently of selected columns, with href resolution only when usable.
- [x] Run read/filter/relative-href round trips and feature read regression tests.

## Task 3: Docs and verification

Files: active docs and generated API references, source docstrings/callers.

- [x] Update current API examples; preserve historical plan/spec records.
- [x] Run focused tests, guide smoke, lint, typing, docs/API generation and full suite.
- [x] Obtain independent read-only review and fix important findings with regressions.
- [x] Record exact results and compare the known 38 pre-task ModelSpec/template failures.

## Execution notes

The user explicitly approved this approach and requested planning plus
implementation using skills. Execute in the shared dirty checkout, preserving
its ongoing refactor. Task baseline files are saved under
/tmp/geosave-unified-parquet-baseline for review. No implementation beyond
GeoParquet selection and necessary caller/documentation migration is authorized
by this plan.


Independent review identified store_schema=False dropping required geometry
and STAC footer metadata, plus row_group_size being rejected by the STAC
encoder's underlying ParquetWriter. Regression tests reproduced both. Both
encoders now reject metadata removal, and the STAC encoder uses the dependency's
public Arrow RecordBatchReader input to bound row groups. Managed metadata and
conflicting Parquet options cannot override the catalog encoding.

Targeted verification: 175 vector/catalog/reader tests passed before review
fixes; all 24 tests in the GeoParquet module passed after the fixes. Targeted
BasedPyright reports zero errors with one existing lazy-export warning.


Final result: 1381 passed, 38 failed, 37 deselected, 3 xfailed. The failed test
names exactly match the pre-task recorded run; none were added or resolved.
Docs/API coverage is 1141/1141, the guide smoke passed, and lint/type checks
passed with the existing lazy-export type warning. The adjacent validation
report records exact failures and commands. All current callers were migrated;
historical specs/plans and notebooks were preserved. No commits or branch changes.
