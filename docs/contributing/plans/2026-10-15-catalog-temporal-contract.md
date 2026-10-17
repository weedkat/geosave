# Catalog temporal contract implementation plan

> Execute inline with superpowers:executing-plans. User explicitly requested
> planning and implementation in the current session; no additional approval
> handoff. Do not commit unrelated shared-worktree changes.

**Goal:** Make temporal matching a join responsibility and frames pure slicing.
**Architecture:** Series -> matched series -> frames -> chips -> lazy row reader.
**Tech stack:** GeoPandas, pandas, PySTAC, xarray, Dask, existing persistence.
**Spec:** ../specs/2026-10-15-catalog-temporal-contract.md

## Constraints and review focus

- Keep native rasters lazy; preserve acquisition metadata and source grids.
- No compatibility aliases, notebook edits, new dependencies or wrapper types.
- Cover singleton dates, timeless/interval sources, incomplete joins, repeated
  observations, persistence and chip reads. Keep timeline diagrams readable.

## Task 1: Record and test temporal row semantics

- [x] Add `times` to series rows and tests through public catalog operations.
- [x] Add red tests for anchored joins, strict/drop, singleton and interval matches.
- [x] Add red tests for frames slicing without joining or tolerance options.

## Task 2: Implement the three ownership seams

- [x] Rename stacks.py to join.py; join grouped series on a collection timeline.
- [x] Replace frames temporal matching with select_frames(times/selection slicing).
- [x] Update row reading for sampling time and acquisition_time lazily.
- [x] Update accessors, exports and FramesSpec to the new contracts.

## Task 3: Migrate and verify connected workflows

- [x] Migrate source consumers, fixtures, persistence, frame/chip/merge tests.
- [x] Improve docstring and guide diagrams; update reference links.
- [x] Run focused red-green tests, connected catalog/reader/model smokes, Ruff,
  focused BasedPyright and API-reference checks; record unrelated failures.
- [x] Obtain an independent final review and address material findings.

## Accepted warning policy (user steering)

Default joins warn and drop incomplete sampling positions. Invalid Item rows,
ambiguous series and unusable frame rows warn and are skipped without discarding
other valid data. Strict matching remains opt-in. Invalid caller arguments and
wrong operation stages still raise. Anchor observations are preserved verbatim;
only the other collections undergo matching. ModelSpec.prepare forwards explicit
tolerance/mode and declares its prepared Collection definitions.

## Verification and review

The independent reviewer identified anchor composites being rematched against
their own overlapping intervals. The regression failed first and now passes:
anchor selections are retained, and only other collections are matched.
The reviewer also requested prepare-level tolerance/mode; these are now forwarded
and covered by asynchronous-source and lazy-persistence smokes.

The shared checkout was preserved without commits. The catalog guide executes
end to end. Focused Ruff and BasedPyright checks pass; the only remaining API
reference freshness differences are the pre-existing attrs-module documentation.

Final connected run: 265 passed (catalog, row reader, model/window declarations,
model preparation, STAC publication and persistence). Additional missing-CRS and
missing-sampling-metadata checks pass. The guide's Python blocks execute end to
end and show both sampling and acquisition timelines. Ruff and focused typing
pass. Notebook files were not modified.
