# Explorer state and lifecycle implementation plan

**Goal:** Fix catalog toggle state, release page subscriptions, and make Identify
follow the map's drawn layers.

**Architecture:** Keep native Panel widgets and Param state. The table owns its
subscriptions; Explorer owns page lifetime and drawn-layer policy. Catalog
loading remains in VectorLayer, and viewport I/O remains in read_scene_pixels
through the public accessor.

**Spec:** Approved review in this conversation and
`../specs/2026-11-03-explorer-raster-loading.md`.

## Constraints

- Preserve existing uncommitted work and native raster readers.
- Keep the padded single-window cache and HoloViews rendering hooks.
- No new tile server, CatalogLayer, compatibility aliases or dependencies.
- Work in the current checkout: the implementation depends on its uncommitted
  refactor. Do not commit unrelated work.

## Tasks

- [x] Add failing behavioral checks for prechecked/copied catalogs, a source
  column named raster, independent footprint/raster toggles, and detached tables.
  Make the table own initialization, collision-safe display columns and release.
- [x] Release custom color-control subscriptions and evict removed layer styles;
  retain one page per Explorer rather than rebuilding subscribers on display.
- [x] Make Identify use drawn raster eligibility and unique layer labels;
  verify comparison/date exclusion without executing excluded raster reads.
- [x] Protect current selections from obsolete catalog-load failures and
  invalidate pending loads on removal/closure; test controlled delayed reads.
- [x] Run focused regressions, the visualization suite with bounded diagnostics,
  viewport reader checks, and lint on changed files. Review the final diff.

## Review focus

Reentrant failure rollback, filtered/sorted row identity, removing an active
table, repeated close/display, and pending loads completing after removal.

## Validation ledger

Implemented with behavioral regressions. Initial regression run reproduced five
failures covering stale ticks, column collision, table detachment and Identify.
Review added checks for reusing a removed catalog and accessing a closed table.
A final teardown check reproduced UI updates during pending-load rollback;
subscriptions now detach before that rollback.

- Visualization suite: 110 passed, 8 slow tests deselected, 3 warnings.
- Viewport readers: 6 passed, 20 deselected, 2 warnings.
- Local server lifecycle: 5 passed, 3 browser/notebook tests deselected.
- Ruff passed on all changed Python files; tracked diff whitespace check passed.
- Final Explorer/server rerun after teardown ordering: 49 passed,
  3 browser/notebook tests deselected, 3 warnings.

Ruling: execute async catalog/server checks outside the sandbox. The sandbox
stalled at the asyncio selector after worker completion; the same catalog test
passed outside it. Controlled pending-result tests use asynchronous events to
exercise ordering independently of thread wakeup behavior.

Ruling: canceling a removed catalog unchecks unfinished loads and preserves rows
already opened, so adding that layer again cannot display a permanently pending
check. Closing only invalidates background results; it cannot stop running I/O.

Ruling: table lookup returns None when its cache has been released, without
changing reactive open-layer state on a destroyed session's page.

Browser interaction remains unverified because Playwright is not installed.
No viewport reader, padded-cache algorithm, or raster rendering hook changed.
