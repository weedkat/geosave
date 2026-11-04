# Explorer reactivity and performance implementation plan

> Implement one stage at a time using the executing-plans skill. Each stage must leave a usable application. Implementation results are tracked in [the execution report](2026-11-04-explorer-reactivity-results.md).

**Goal:** Make ownership and reactive dependencies understandable, eliminate unnecessary work, and improve viewport reuse without losing existing reader and session guarantees.

**Architecture:** Explorer owns application state and layer membership. Persistent Panel components display that state; HoloViews owns viewport streams and map updates. Layers own rendering and disposable pixel caches; geodata I/O owns reading pixels.

**Stack:** Panel, Param, HoloViews, GeoViews, Bokeh, native xarray/GeoPandas.

**Prior design:** [Raster loading](../specs/2026-11-03-explorer-raster-loading.md), [native HoloViz explorer](../specs/2026-10-16-native-holoviz-explorer.md), and [completed lifecycle fixes](2026-11-03-explorer-state-lifecycle.md).

## Target pattern

```python
class LayerDetails(Viewer):
    layer = param.ClassSelector(class_=Layer, allow_None=True, allow_refs=True)

    @param.depends("layer")
    def view_info(self):
        return format_info(self.layer.get_info()) if self.layer is not None else ""

    def __panel__(self):
        return self._layout

# Construct the pane once in __init__; Panel consumes the method reference.
self._info = pn.ui.HTML(self.view_info)
# The parent supplies a native parameter reference.
details = LayerDetails(layer=explorer.param.active_layer)
```

This is the proposed interface, to smoke-test in stage 2. Decorated display methods return plain content or an already-owned component. Do not add `watch=True` to methods Panel evaluates. `@param.depends(..., watch=True)` remains appropriate for fixed state transitions; explicit `watch`/`unwatch` remains appropriate for changing sets of layers and disposal. Decorators alone do not make callbacks faster.

Keep `viewer.period`, `viewer.freq`, `viewer.layers`, and `viewer.active_layer` in this iteration. A Timeline object is deferred: reorganizing temporal methods and narrowing dependencies delivers the immediate benefit without a public parameter migration. The user delegated this choice to the recommendation.

## Preserve these decisions

- Keep one persistent Map, HoloViews pane, DynamicMap, and its streams. Preserve framing, stream anchoring, hover scoping, and RGB output type guarantees.
- Preserve `read_scene_pixels` in `geodata/io/stac.py`, including window/overview selection, nodata, reprojection, credentials, and resource closure. Viz chooses a display grid; it does not implement another asset reader.
- Keep padded viewport reuse and quantized resolution. A contained pan already avoids pixel reads and image-array replacement.
- Do not recreate tables, style widgets, or subscriptions during every view evaluation. Keep the established release ordering and obsolete-load guards.
- Keep catalog behavior on VectorLayer. Do not reintroduce CatalogLayer, generic controllers, forwarding aliases, or a tile server.
- Keep footprint checkboxes independent from raster checkboxes. Table filtering/sorting must address original rows correctly.
- Keep computed xarray data lazy outside the requested operation. A small output image does not guarantee small source computation.
- Preserve unrelated worktree edits; do not use this refactor to reorganize static plotting or I/O.

## Measured baseline

Two processes ran the [diagnostic probe](explorer-reactivity-baseline/probe.py); raw reports are [run 1](explorer-reactivity-baseline/run-1.json) and [run 2](explorer-reactivity-baseline/run-2.json). Panel 1.10.0, Param 2.4.2, HoloViews 1.23.2, GeoViews 1.15.1, Bokeh 3.10.1.

Fixture: a 256×256 lazy RGB dataset with 64×64 chunks, two polygons, and a materialized full Panel page in a Bokeh Document. Five warmed repetitions per ordinary operation; one zoom-away/return pair per process. Timings include diagnostic instrumentation and synchronous Python work. There is no browser or network, and PlotSize uses the server-side fallback.

| Operation | Median ms, run 1 / run 2 | Work per action |
|---|---:|---|
| Raster opacity | 64 / 63 | 9 menu builds, 2 legend formats, 1 period update, 0 pixel reads |
| Vector color | 79 / 80 | 9 menu builds, 2 legend formats, 1 period update, 1 vector render |
| Active layer | 49 / 51 | 9 menu builds, 0 map draws or pixel reads |
| Cached pan | 60 / 57 | 1 map draw, 0 pixel reads, 0 image-array updates |
| Zoom in | 1,158 / 5,446 | 1 pixel read, 175 Dask tasks, 1 image-array update |
| Return to original zoom | 1,161 / 3,989 | 1 pixel read, 127 Dask tasks, 1 image-array update |
| Hidden raster pan | 40 / 38 | 0 pixel reads or image-array updates |

Counts agree across runs; zoom timings are too variable and undersampled for a latency claim. One retained pixel Dataset occupies 66,816,320 bytes in run 2. Each zoom emits an image-array update containing 22,259,524 raw array bytes. Neither number measures websocket traffic or total retained memory; rendered, contrast, and overview arrays add memory. Model count changes are not evidence of model identity stability or absence of leaks.

Reproduce the diagnostic snapshot from the repo root:

```bash
uv run --no-sync python docs/contributing/plans/explorer-reactivity-baseline/probe.py > /tmp/explorer-baseline.json
```

The snapshot instruments current private methods and will need adaptation after refactoring. `scripts/measure_explorer.py` currently imports removed `read_viewport`; repair that benchmark before using it as evidence.

## Stage 1 — Establish reliable performance and behavior gates

**Files:** `scripts/measure_explorer.py`, `scripts/smoke_explorer.py`, existing tests under `tests/geodata/viz/`.

- [x] Replace the obsolete benchmark reader import with the actual RasterLayer and native catalog-row path. Preserve its HTTP range accounting.
- [ ] Cover two source types separately: a tiled COG with overviews through a catalog row, and a computed lazy xarray raster. Record requested grid, source tasks, HTTP requests/bytes, pixel reads, image updates, and process memory.
- [ ] Record cold construction separately from warm interaction. Run 20 warm repetitions in five processes for opacity, selection, cached pan, outside-cache pan, zoom A→B→A, band/time changes, hide/show, and resize. Report median and p95 with fixture and viewport dimensions.
- [x] Materialize the full page. Add browser measurements of interaction-to-display latency and actual websocket payloads; label Python callback timing separately.
- [x] Exercise both catalog checkboxes through filtering, sorting, copying, load failure, removal while loading, and close. Reuse surviving contract tests before adding any missing coverage.

**Gate:** A reproducible baseline for each source type; no browser launch in ordinary tests. Browser checks belong to the existing slow/server workflow. No refactor speed claim until the same harness measures both versions.

## Stage 2 — Make Explorer and component lifetimes readable

**Files:** `explore/explorer.py`, `explore/page/sidebar.py`, `explore/page/table.py`; corresponding Explorer/page tests.

- [x] Arrange Explorer as parameters → construction/display → public operations → state queries → layer reconciliation → temporal transitions → serving/disposal. Keep methods with their callers and document only non-obvious ownership constraints.
- [x] Split `_watch_layers` into subscription reconciliation and active-selection repair. Keep one membership transition coordinating these operations. Catalog loading remains on VectorLayer; comparison choices are derived from raster membership.
- [x] Smoke-test the proposed LayerDetails parameter-reference interface against the installed Panel version, including replacement of the active layer and setting it to None.
- [x] Verify disposal of Param reference subscriptions before extending this pattern to other Viewers. After release, source changes must neither update destroyed UI nor retain removed layers; document the supported release mechanism alongside the existing explicit watcher cleanup.
- [x] Convert LayerDetails information and selected-style display to named decorated methods. Create pane/layout shells once; return formatted content or cached style controls. Keep cache pruning and callback disposal owned by LayerDetails.
- [x] Split AttributeTables membership reconciliation from selection following in `page/table.py`. Selection must reuse the existing table and its filters.

**Gate:** Selection displays the correct information/style/table with no raster reads. The map and table model identities survive repeated selection. Removing a selected layer and closing during a pending load remain safe. Run focused tests and the stage-1 benchmark; a pure rearrangement should not change work counts.

## Stage 3 — Remove unnecessary reactive work

**Files:** `explore/explorer.py`, `page/sidebar.py`, `page/table.py`, `page/map.py`, `layers/vector.py`.

- [x] Replace broad `layer_setting_changed` consumers incrementally with relevant parameter references or explicitly owned dynamic subscriptions. Remove the global signal only after its last consumer is migrated; do not introduce a second permanent event bus.
- [x] Migrate Map's `_settings` HoloViews Params bridge as part of that change. Request one map redraw per logical transition; exclude status/error-only changes and derived `fitted_clim` feedback. Preserve stream cleanup and verify that period changes affecting multiple layers do not trigger intermediate redundant draws.
- [x] Make period choices depend on dated raster membership, visibility where currently relevant, and frequency. Opacity, colormap, vector color, and selection must not scan dates.
- [x] Build each LayerList group's entries once for changes affecting its contents. Derive the selected index and visibility from retained entries; selection must not rebuild entries. Subscribe to parameters actually used by labels, summaries, and actions.
- [x] Make each legend depend on its owning layer's relevant style values. Changes to another layer must not format that legend. Opacity need not format legends unless the displayed legend actually uses opacity.
- [x] Narrow VectorLayer's draw-cache key to geometry, visible rows, and rendering style. Include viewport/resolution for viewport-dependent dense rendering. Catalog error/loading/raster membership alone must not invalidate polygon rendering.

**Dependency targets:**

| Change | Work allowed | Work excluded |
|---|---|---|
| Opacity | Apply map opacity | Menu rebuild, date scan, unrelated legends, pixel read |
| Active layer | Active menu index, details, table selection | Entry rebuild, map draw, pixel read |
| Raster band/time | Selected raster pixels and legend | Unrelated vector render/legend |
| Catalog raster checkbox | Load/cancel selected raster, reconcile membership | Footprint selection change |
| Catalog footprint checkbox | Polygon selection/render | Opening raster assets |
| Cached pan | Viewport-dependent map work | Pixel read or image-array replacement |

**Gate:** Diagnostic counts meet the table. Regression tests protect visible behavior and absence of unintended reads; private callback counts remain benchmark diagnostics, not brittle unit tests. Repeated select/remove/close cycles leave no growing subscriptions or retained removed layers.

## Stage 4 — Separate comparison state and server mechanics

These are two independently reviewable changes; neither is a prerequisite for caching.

- [x] Move comparison state from `page/comparison.py` into `explore/comparison.py`. Comparison owns available choices and selected side; keep its controls/layout in `page/comparison.py`. Explorer retains `comparison`; `Explorer.select_drawn_rasters()` remains the single combined visibility/date/comparison eligibility calculation used by Map and Identify. Preserve selection repair. Do not move a UI-owning class wholesale and call that separation.
- [x] Move server registry/port/stop mechanics into existing `explore/server.py`. Keep Explorer's `serve`, `stop`, and `close` entry points and existing arguments; Explorer still owns session copies and aggregation of drawn regions. Avoid a lifecycle mixin or a general service container. Only introduce an owning server class if functions cannot express its resource lifetime clearly.

**Gate:** Comparison and Identify agree about displayed rasters. Session copies remain independent, region aggregation works, and closing/removing while loading cannot mutate a destroyed document. Run non-browser server tests, then the separate slow browser/notebook cases in suitable environments.

## Stage 5 — Optimize raster reuse within a memory budget

**Files:** `layers/raster.py`, benchmark script, focused raster-layer tests. Preserve `io/stac.py` ownership.

- [ ] First measure grid dimensions and overfetch at real browser sizes. Compare current padding against smaller padding and a maximum display-grid area. Assess sharpness, pan hit rate, memory, and transfer size together; do not silently sacrifice resolution to improve timing.
- [x] Prototype a per-layer byte-bounded LRU of computed viewport pixels. Match selection, quantized resolution, grid alignment, and containing extent. Key all pixel-affecting selectors, including RGB/band, time, and nonspatial axes. Retain rendering-style changes outside the pixel key.
- [ ] Account for all retained buffers, avoiding double-counting shared arrays. Evict least-recently-used windows; release entries when a layer is removed/closed. Never cache live widgets, Viewer instances, open readers, or credentials globally with `pn.cache`.
- [x] Benchmark budgets of one, two, and three observed viewport footprints before choosing a default. Define over-budget behavior explicitly: retain no oversized extra entry, and still permit a current view to render.
- [ ] Test A→B→A reuse when both windows fit, eviction, selection invalidation, hidden rasters, and close. Verify exact grid/nodata/band results, not only hit counts.

**Gate:** A→B→A performs no new source read on the return when A is retained. Contained pans still send no new image array. Memory respects the documented cache budget plus measured current-render overhead. A zoom return may need image transfer even when source pixels are cached. Adopt only if benefits survive realistic COG and computed-raster measurements.

## Stage 6 — Conditional responsiveness and UI improvements

Start after cache/overfetch evaluation; adopting the LRU is not a prerequisite. Asynchronous reads improve responsiveness, not necessarily total computation time.

- [ ] If uncached reads block interaction materially, prototype bounded background pixel reads with one latest desired request per layer. Coalesce viewport bursts, discard stale results by generation, and apply results on the owning document thread. Workers never mutate Panel/Bokeh objects. Cancellation must not claim to stop an already-running raster read.
- [ ] Keep the previous image visible while loading, expose loading/error status, and distinguish the requested viewport from the image currently displayed. Preserve catalog-load and pixel-read lifecycle ownership separately.
- [ ] Add concise tooltips explaining “Footprint” and “Raster” checkboxes, a retry action for failed loads, and a “Fit visible layers” action. First verify which already exist; avoid duplicate controls.
- [ ] Evaluate an automatic/manual viewport refresh option only if remote/computed read costs justify it. Defer cache controls and performance diagnostics from the ordinary UI unless users need to make a meaningful choice.

**Gate:** Browser interactions remain responsive during a deliberately slow read; the last request wins; removed/closed layers cannot publish late results. Checkbox state and errors remain understandable without reading implementation details.

## Verification and completion

For each stage, inspect existing tests first, add only missing observable contracts, run the focused tests, then interaction count/identity smoke checks. Repeat the full timed comparison after stages 3 and 5, or when a regression appears; avoid five-process timing runs for simple method moves. For the integrated result:

```bash
uv run pytest tests/geodata/viz -m 'not slow'
uv run pytest tests/geodata/viz -m slow
uv run ruff check src/geosave/geodata/viz tests/geodata/viz scripts/measure_explorer.py
```

Run relevant `read_scene_pixels` tests if its calling contract changes; changing its implementation is outside this plan. Keep server/browser/notebook requirements explicit rather than silently skipping them or treating Python timing as visual validation. Update `docs/concepts/explorer.md` and `docs/contributing/architecture.md` with final ownership and measured limits.

Review focus: filtered/sorted catalog row identity (stage 1), parameter-reference replacement (stage 2), viewport-sensitive dense vector invalidation (stage 3), destroyed-session callbacks (stages 4/6), and memory accounting across shared pixel/render buffers (stage 5).

Recommended execution order: stages 1–3 first; comparison/server changes independently; caching next; asynchronous loading only if evidence warrants it. Every stage has its own review and stop point. No broad rewrite is required to obtain the first reduction in UI work.

## Documentation references

- [Panel parameter dependencies](https://panel.holoviz.org/en/docs/latest/how_to/param/dependencies.html): pass decorated methods to Panel so it owns reevaluation.
- [Param dependencies and watchers](https://param.holoviz.org/en/docs/latest/user_guide/Dependencies_and_Watchers.html): declarative dependencies and explicit lifecycle subscriptions have different uses.
- [Panel reactivity](https://panel.holoviz.org/en/docs/latest/explanation/api/reactivity.html) and [user experience](https://panel.holoviz.org/en/docs/latest/how_to/best_practices/user_experience.html): reuse components and avoid unnecessary model reconstruction.
- [Panel asynchronous callbacks](https://panel.holoviz.org/en/docs/latest/how_to/callbacks/async.html) and [memoization](https://panel.holoviz.org/en/docs/latest/how_to/caching/memoization.html): evaluate worker/document boundaries and cache scope explicitly.

Planning checks: two instrumented full-page smoke runs completed. A separate installed-Panel smoke passed decorated display, parameter-reference replacement, None selection, and stable layout checks using a native HTML pane; the `pn.ui.HTML` integration remains a stage-2 check. The diagnostic probe passes Ruff. No production refactor was made for this plan. Browser latency, real network transfer, and post-refactor performance remain unmeasured.

## Execution status (2026-11-04)

The core reactivity/ownership refactor and bounded viewport reuse are implemented.
See the [execution report](2026-11-04-explorer-reactivity-results.md) for exact
checks, measured improvements, compatibility changes, and deferred conditional
work. Retry remains the existing checkbox action, clarified in the UI. Async
viewport workers, altered padding, and the complete browser timing matrix were
not implemented; do not interpret the core completion as those gates passing.
