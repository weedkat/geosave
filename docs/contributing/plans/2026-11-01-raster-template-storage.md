# Separate raster templates from storage

**Goal:** Make `path` an exact destination unless a separate relative `template`
is supplied, in which case `path` is the parent location.

**Design:** `expand_paths(raster, template, *, collection=None)` returns relative
strings mapped to lazy Dataset selections. Raster writers own UPath conversion,
filesystem options, joining, and final collision detection. Exact destinations
are never formatted. No new wrapper values or compatibility execution paths.

The approved approach changes the public API as well as the helper. Merely
changing helper types would leave the public `path` argument ambiguous.

## Tasks

- [x] Add failing contracts for relative expansion, escaping the parent,
  literal braces, remote options, and public template-based round trips.
- [x] Separate expansion and destination resolution; update Dataset, DataArray,
  DataTree, catalog writers and their real consumers. Preserve ModelSpec data
  flow while migrating its calls to the explicit storage/template boundary.
- [x] Update examples and docs (excluding notebooks), run relevant tests and
  Ruff, and review the change against the pre-task working-tree snapshot.

## Verification

Keep coordinate, metadata, laziness, collision and delayed-write protection.
Exercise exact local destinations, remote parents with options, mixed dated and
undated groups, normalized destination collisions, and write/read round trips.
Reject absolute paths, URLs and parent traversal in relative template output.
Do not compute raster pixels during expansion or destination resolution.

## Execution notes

Work in the existing checkout because this change depends on its uncommitted
source migration. Do not commit or discard unrelated changes. The baseline for
review is `/tmp/geosave-path-baseline`, rather than HEAD.

## Validation notes

- Relative expansion and explicit-template round trips: 13 contracts failed
  against the old implementation and passed after separating the boundary.
- Independent review: no important runtime findings; the stale API-reference
  member it identified was migrated to the writer-side resolvers.
- Repository Ruff and API-reference freshness checks pass. Documentation builds;
  Zensical reports that strict mode is unsupported.
- Geodata plus the affected model/flow tests: 1168 passed, 6 failed,
  21 deselected, 3 xfailed. The same six ModelSpec failures reproduce against
  the pre-change source and tests (35 passed, 6 failed): retired sample accessor
  indexing/length and `GeoCatalogRow.to_xarray(target=...)`. These are outside
  the path/template change and remain unmodified.
- Plain GeoTIFF template smoke round-tripped pixels under a literal-brace parent.

- Final focused raster/ingest run: 195 passed, 3 deselected.
- Default suite: 1427 passed, 38 failed, 37 deselected, 3 xfailed. All 38
  failing test names match the baseline reproduction (77 passed, 38 failed,
  5 deselected) across model, generated recipes and the CLI walkthrough.
  Beyond the six sampling failures above, these concern the removed
  `ModelSpec.chips` contract in generated recipe code/configuration.
- Final error-message smoke: 1 passed, 58 deselected. The multiple-date
  GeoTIFF error now directs callers to add `{time}` to `template`.
- Breaking API: embedded filename templates move to the keyword `template`;
  `ModelSpec.ingest` forwards that keyword too. `expand_paths` now accepts
  strings and returns relative string keys; group storage resolution lives in
  `writers.resolve_group_paths`. Notebook files were left untouched.
