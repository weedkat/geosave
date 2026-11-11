# Raster Extent Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve declared raster geometry and expose it as anchor extent.

**Architecture:** An Extent attrs group owns ACDD WKT/CRS serialization. Raster
accessors write metadata without changing pixels and derive anchors independently
of grid placement; STAC translates the same extent at I/O boundaries.

**Tech Stack:** xarray, ODC Geo, Pydantic, Shapely, PyProj, PySTAC.

**Spec:** ../specs/2026-11-06-raster-extent.md

## Global constraints

- Keep native objects and lazy pixels; no compatibility alias.
- Preserve existing working-tree changes; do not edit notebooks.
- Work in the current workspace because the reviewed implementation is uncommitted.
- Do not commit unrelated changes. User authorized planning and implementation.

## Review focus

- ACDD latitude/longitude order must not move the polygon.
- Extracted bands and detached tree nodes must retain metadata.
- Differing grids must not prevent reading explicitly declared extents.
- Serialization must preserve holes, multipolygons, CRS and laziness.
- Malformed or incomplete extent metadata must fail rather than fall back silently.

## Tasks

- [x] Add failing tests for extent assignment, clearing, independent selection,
  coordinate order, invalid metadata, tree aggregation and NetCDF/Zarr/GeoTIFF
  persistence; run them to confirm missing behavior.
- [x] Add `attrs/groups/extent.py`: `Extent.from_geometry(Geometry)` and
  `to_geometry()`, ACDD aliases, validation and serialization; register the group.
- [x] Add `GeoRaster.write_extent(Geometry | None)` and update anchor resolution.
  Rename GeoRegion's field and update production callers, tests and active docs.
- [x] Retain explicit ODC regions and STAC Item geometry at loading boundaries;
  publish raster anchor extents. Add real I/O tests before implementing changes.
- [x] Run focused workflows, lint changed Python files, and review the final diff.

## Execution record

- Planning: existing attrs groups have no geospatial_bounds owner. Dataset attrs
  do not propagate to selected variables; stamp both intentionally.
- Ruling: stored extent is a declared region preserved by native operations;
  users explicitly change it with write_extent. No implicit clipping is added.
- Ruling: temporal bounds interpretation remains unchanged in this spatial change.

- Verification: initial extent/anchor workflow passed 20 tests; storage round trips
  passed for NetCDF, Zarr and GeoTIFF outside the sandbox. Zarr stalled inside it.
- Review: fixed STAC fallback overwriting per-band extents and retained ODC
  FeatureCollection / geo-interface support; reviewer confirmed both fixes.
- Broad verification: 1108 passed, 16 deselected, 3 xfailed, one stale test caller
  failed on `.geometry`; updated that caller to `.extent`. Full default run recorded below.

- Final default suite: `uv run --no-sync pytest -q --tb=short` completed with
  1579 passed, 2 failed, 41 deselected, 3 xfailed and 144 warnings (213 seconds).
  Remaining failures: `tests/examples/test_notebooks.py::test_catalog_examples_load_chips_and_restore_pixels`
  calls the removed `to_datatree()` from a notebook; and
  `tests/geodata/viz/explore/test_explorer.py::test_static_and_explorer_backends_in_both_orders`
  hits a band/RGB plotting conflict. Neither failing implementation path was
  modified for this extent work. Notebooks remain untouched as instructed.
- Ruff checks and formatting, generated API freshness and `git diff --check`
  pass. Changes remain uncommitted alongside the existing working-tree work.
