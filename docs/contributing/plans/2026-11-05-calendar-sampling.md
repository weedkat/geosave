# Calendar Sampling Implementation Plan

> Ownership correction: [2026-11-06 sampling ownership](../plans/2026-11-06-sampling-ownership.md)
> supersedes the preparation settings and GeoSamples loader described below.
> The interval assignment and calendar grouping contracts remain in effect.


> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace reference-based pairs with independently assigned temporal slots and lazy sample preparation.
**Architecture:** Catalog owns the public request; vector/sampling.py owns spatial and temporal membership operations; GeoSamples owns membership and preparation of its samples using existing raster I/O and transforms.
**Tech Stack:** GeoPandas, pandas, xarray, existing GeoSave temporal operations.
**Spec:** ../specs/2026-11-05-calendar-sampling.md

## Global Constraints
Preserve unrelated working changes; work in this existing integration workspace without commits.
No new wrappers, dependencies, input/target roles, notebook edits or implicit reprojection.

## Review Focus
Inclusive interval ends at bucket boundaries; collection names colliding with columns;
missing middle/edge slots; preservation of actual temporal coverage; lazy file lifecycle.

### Task 1: Membership and windows
- [x] Add failing tests for independent slots and crossing policies.
- [x] Implement read_observations, assign_slots, select_windows and spatial grouping with plain tables/lists.
- [x] Replace catalog sample and remove the old pair/window algorithm.
- [x] Migrate old contract tests and check vector tests.

### Task 2: Prepared samples
- [x] Add numerical lazy reduction/interpolation and lifecycle tests.
- [x] Extend GeoSamples with slot membership, times and to_datatree; use existing reducers and readers.
- [x] Check temporal/vector/I/O integration suites and typing/lint.

### Task 3: Documentation and review
- [x] Update active guides and generated API references; preserve historical plans.
- [x] Review implementation and run focused checks; record limitations and exact counts.


## Verification and review

- Initial planning regressions: 9 failed against the old API, then passed.
- Prepared loading/interpolation: 4 new failures before implementation, then passed.
- Independent review: corrected last-slot-only collection metadata (regression
  failed before fix), removed all-samples timeline scans from single-sample loading,
  and activated temporal planning for explicit nondefault preparation settings
  (regression failed before fix).
- Final focused run: 169 passed, 2 deselected, 6 warnings across vector,
  temporal transforms, STAC catalog, GeoParquet, and model processing tests.
  This is not a full-library or notebook-suite result.
- Ruff passed for vector source/tests and migrated call sites; BasedPyright on
  vector source: 0 errors. Generated API freshness and git diff --check passed.
- No new dependencies. Pixel-gap eligibility uses a NumPy scan through xarray
  rather than requiring optional bottleneck forward/backward filling.
- Decisions: intervals include their end timestamp; zero surviving windows returns
  empty samples; nondefault temporal options activate temporal planning; interpolation
  keeps a conservative window-wide coverage envelope and is not forecasting-safe
  by itself. Wholly absent collections remain empty groups under missing='keep'.
- Breaking changes: pair/tolerance/keep removed; static collections must be explicitly
  selected when collections is supplied. Prepared loading is samples.to_datatree(i);
  samples[i].gs.to_datatree() still reads raw sources. Mixed static/dated products
  must be separated into collections for prepared loading. No notebook files changed.
