# Visualization Ownership Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` for inline execution
> after design/plan review, or `superpowers:subagent-driven-development` if the
> user chooses delegation. This draft does not authorize implementation.

**Goal:** Give visualization state, callbacks and caches clear owners while
preserving catalog browsing and native overlay composition.

**Architecture:** Explorer coordinates layers and layout. Map, Comparison and
Attributes own their respective rendering, interaction state and native table.
Layers retain data-specific behavior and source caches; docks supply controls.

**Tech stack:** Existing HoloViews, Panel, Param, panel-material-ui, Datashader,
xarray, GeoPandas, PySTAC and GeoSave readers. No new dependencies.

**Spec:** [Visualization ownership proposal](../specs/2026-10-15-viz-ownership-design.md).

## Global constraints

- Catalog browsing, search, row toggles and overlays remain core behavior.
- Follow the existing split: GeoCatalog for STAC-GeoParquet Items;
  GeoStackCatalog for grouped series/stacks and their frame/chip selections.
- Native raster/vector data are shared by reference; pixels are never deep-copied.
- Raster I/O remains lazy beyond the explicitly computed overview.
- Tabulator remains the column-filter parser.
- Functions name actions; variables name their values; keep names concise.
- No compatibility aliases, generic event bus, plugin registry or resource manager.
- Preserve unrelated working-tree changes and ignore notebook files.
- Do not close native input objects supplied by callers.

## Review focus

- A business `sources` column must not turn a plain feature dataset into a catalog.
- A removed/re-added layer must not retain a viewer or receive duplicate callbacks.
- Composition after opening catalog rows must preserve source/layer associations.
- Duplicate labels and second-resolution dates must remain distinct choices.
- Native filters must retain original row indices through pagination and selection.
- A vector catalog footprint must never be mistaken for its vector asset data.

## Work that can proceed independently

The catalog split is a prerequisite for catalog-specific integration, not for
all visualization work. Keep two tracks:

| Can proceed against native rasters/vectors now | Integrate after the catalog split |
| --- | --- |
| Extract Comparison state, choices and swipe layout | Recognize GeoCatalog and GeoStackCatalog in layer construction |
| Extract Map viewport, basemap and stream ownership | Catalog row opening, cache associations and frame/chip semantics |
| Release raster/vector dock caches and owned subscriptions | Catalog-specific cloning and toggle integration |
| Own native attribute tables and row-filter synchronization | Vector-asset catalog reading and its capability contract |
| Smoke native hvPlot numeric raster/vector rendering | Changes to geodata catalog recognition or persistence |

Start with Comparison as the smallest ownership extraction. Its inputs are
Layer/RasterLayer objects and dates; it does not need either catalog accessor.
Then isolate Map and plain raster/vector lifecycle work. Keep existing catalog
integration tests as guards; defer changes to their domain contract to Phase 0.

Before changing the renderer, compare native hvPlot `rasterize=True` with the
current shaded RGB path on numeric bands, classes, composites and mixed geometry.
Check numeric hover/colorbars, viewport streams, nodata, categorical kernels and
actual lazy window reads. A Leaflet backend is a separate frontend experiment:
it must prove a needed interaction and a raster delivery path before replacing
the current HoloViews/Bokeh map. Do not introduce multiple map backends merely
to make the ownership extraction configurable.

Planning smoke results on 2026-10-15: hvPlot over projected Spatialpandas polygons
with `rasterize=True` returned an Image DynamicMap, PlotSize/RangeXY streams,
numeric values `[1, 2, nan]`, a Bokeh hover tool and a colorbar. Changing the range
changed image bounds. Direct GeoPandas hvPlot geographic plotting required
GeoViews, which is not installed in the inspected environment. Switching one
rendered DynamicMap from Image to RGB reproduced its fixed-element-type assertion.
These are feasibility results, not a complete renderer replacement: preserve
mixed geometry, classes and the existing source-window I/O contracts in later tests.

## Phase 0: integrate the GeoCatalog/GeoStackCatalog contract

This precedes catalog-specific integration and the vector-catalog reader design;
the isolated visualization work listed above can proceed independently.
The existing catalog-split plan owns the geodata implementation. This phase
checks its contract and integrates visualization; it does not create a competing
split implementation. GeoStackCatalog is planned but absent from the currently
inspected source tree.

**Files to inspect:**
- `src/geosave/geodata/core/{dataframe,vector,catalog,sample}.py`.
- `src/geosave/geodata/sampling/{schema,series,join,frames,chips,merge}.py`.
- `src/geosave/geodata/viz/explorer/layers/{__init__,catalog}.py`.
- `tests/geodata/core/{test_catalog,test_sample}.py` and
  `tests/geodata/sampling/{test_catalog,test_series,test_join,test_frames,test_chips}.py`.

**Required outcome:** a state/capability matrix mapped onto the two agreed
accessors, canonical recognition at geodata, and integration tests before
extending catalog loading.

- [ ] Trace Item -> series -> joined -> frames -> chips, including empty tables.
  Record each method's valid input states, required fields, output state, metadata
  preservation and refusal/warn/drop policy. Keep table methods and row readers
  distinct. Include serialization, spatial/time query, concatenation and explore.
- [ ] Assert Item tables use GeoCatalog, ordinary features use GeoVector, and
  series/joined/frame/chip tables use GeoStackCatalog. Cover transitions and
  persistence round trips, including empty tables.
- [ ] Verify method ownership follows the split: GeoCatalog owns Item export and
  STAC-GeoParquet, and produces GeoStackCatalog through grouping; GeoStackCatalog
  owns joining, frame/chip selection and grouped reconstruction. Do not retain
  invalid Item-only methods through inheritance or compatibility aliases.
- [ ] Add a regression fixture for an ordinary feature GeoDataFrame with
  `sources = ['business metadata']`; it must retain GeoVector behavior rather than
  become a catalog. The planning smoke reproduced the current misclassification.
- [ ] Test complete/partial grouped schemas and column subsets against the same
  canonical recognition rules used by accessor registration and row reading.
  Check cached accessors after column mutations before promising dynamic retyping.
- [ ] Test method applicability for all five states, including grouping an already
  grouped table, joining/framing after selection, chipping twice, and concatenating
  different grouped stages. Review any proposed behavior change separately from
  preserving an existing contract; do not normalize all cases to one exception.
- [ ] Choose the minimal canonical detection/validation functions in geodata's
  catalog schema and delegate from accessors and viz, following the existing split
  implementation. No parallel viz classifiers and no additional per-stage wrapper
  hierarchy beyond GeoCatalog and GeoStackCatalog.
- [ ] Update layer construction to recognize both accessor types as browsable
  catalogs. Pin a regression proving a GeoStackCatalog does not fall through to
  ordinary VectorLayer rendering after the split.
- [ ] Prove exploration does not regroup selected frames/chips or discard their
  sampling/source/grid relationship. Keep the existing raw-Item grouping behavior
  explicit and covered before changing it.
- [ ] Add the asset-kind axis to the matrix. Do not advertise chips, raster grids,
  raster temporal grouping or raster readers on vector catalogs without a proven
  compatible contract. Unsupported operations must explain their preconditions.
- [ ] Review this contract before implementing catalog-specific parts of Phase A;
  revise the vector-opening milestone to follow it. Plain raster/vector tests and
  Map/Comparison extraction do not wait for the catalog implementation.

## Phase A: ownership refactor

### Task 1: pin lifetime and overlay contracts

**Files:**
- Modify `src/geosave/geodata/viz/explorer/app.py`.
- Modify `src/geosave/geodata/viz/explorer/layers/{base,raster,vector,catalog}.py`.
- Modify `tests/geodata/viz/explorer/test_app.py` and `test_layers.py`.

**Interfaces:**
- `Layer.clone(copies: dict[Layer, Layer] | None = None) -> Layer`.
- `Explorer.remove(layer: Layer) -> None` becomes the membership-removal path.
- `Explorer.__mul__` returns a new composition with fresh application bindings.

- [ ] Add regression tests for dark-viewer composition preserving the original
  theme, date, selected rows and stack. Assert native raster/vector objects are
  shared, and cloning alone schedules no Dask computations.
- [ ] After the catalog split, add a catalog test that opens rows before composition; the composed stack
  must contain the same copied layers returned by its copied catalog cache.
  Unticking in the composed viewer must leave the original map unchanged.
- [ ] Run these tests before implementation and confirm the current reset/sharing
  failure; do not infer independence merely from different Explorer identities.
- [ ] Implement each concrete layer's cloning with its own constructor/settings;
  use the identity dictionary for nested opened layers. Never copy Param watcher
  internals, widgets, native datasets or input feature buffers.
- [ ] Route menu removal through `remove`; preserve the existing catalog menu
  behavior proposed in the spec. Keep removal distinct from hiding/unticking.
- [ ] Run `uv run pytest tests/geodata/viz/explorer/test_app.py
  tests/geodata/viz/explorer/test_layers.py tests/geodata/sampling/test_series.py`.

### Task 2: own the native attribute table

**Files:**
- Create `src/geosave/geodata/viz/explorer/attributes.py`.
- Modify `explorer/app.py` and `explorer/docks/table.py`.
- Create `tests/geodata/viz/explorer/test_attributes.py`.
- Update relevant table tests in `test_docks.py`.

**Interfaces:**
- `Attributes(layer: VectorLayer, *, theme: Theme)` owns one native Tabulator.
- `Attributes.table` exposes that native widget; no parallel filter expression type.
- `Attributes.close() -> None` detaches its owned subscriptions.
- `Explorer.get_attributes(layer: VectorLayer) -> Attributes` returns the single
  existing owner for that layer or creates it.

- [ ] Test that two requests for the same layer return the same owner/table.
  Test area `>= 1.5` plus search `rice` yields original row `[2]` and preserves
  the selected source rows. Retain the remote-pagination coverage.
- [ ] Test removal detaches table-to-layer and layer-to-table callbacks; changing
  a retained old table must not modify an active Explorer or reopen a source.
- [ ] Move native table construction and synchronization out of the dock. Track
  subscriptions explicitly. Use native column filters plus the layer predicate;
  use the verified plain-predicate refresh if reactive filter teardown cannot be
  expressed through public dependency interfaces.
- [ ] Leave the dock responsible for choosing the displayed vector, its title
  and placement. Remove its competing per-layer table cache.
- [ ] Run `uv run pytest tests/geodata/viz/explorer/test_attributes.py
  tests/geodata/viz/explorer/test_docks.py tests/geodata/viz/explorer/test_app.py`.

### Task 3: make cleanup explicit at its owners

**Files:**
- Modify `explorer/app.py` and `explorer/docks/{base,layers,style,search,inspection}.py`.
- Modify lifecycle tests in `test_app.py`, `test_attributes.py`, `test_docks.py`.

**Interfaces:**
- `Dock.release_layer(layer: Layer) -> None` and `Dock.close() -> None`.
- `Explorer.close() -> None`, safe to call twice.

- [ ] Write weak-reference tests that build real controls/tables, remove a layer,
  drop external view references and collect garbage. The removed layer must be
  unreachable when the test retains neither it nor its native view objects.
- [ ] Retain a layer separately, remove/re-add it, and assert one callback/update
  per edit. Closing one composed viewer must leave the other responsive.
- [ ] Record and release GeoSave-created watchers at their actual owners. Evict
  dock caches on removal and all caches on viewer close. Avoid bindings that
  capture an Explorer through per-layer legend/control callbacks.
- [ ] Do not treat `pn.Param.object = None` as proof of disconnection. Verify real
  effects and reachability; do not inspect or alter private Panel registries.
- [ ] Preserve caller-owned data resources and intentional data caches on a
  separately retained Layer. If native widget callbacks still retain a viewer,
  resolve that specific binding before continuing, rather than weakening tests.
- [ ] Run the explorer tests and a repeated open/toggle/remove/close smoke.

### Task 4: move viewport rendering into Map

**Files:**
- Move `viz/map.py` to `viz/explorer/map.py` and `viz/raster.py` to
  `viz/explorer/raster.py`; update all source and test imports.
- Modify `explorer/app.py` and `explorer/layers/{base,raster,vector}.py`.
- Move viewport tests to `tests/geodata/viz/explorer/test_map.py` and raster
  sampling tests to `tests/geodata/viz/explorer/test_raster.py`.

**Interfaces:**
- Keep `View`, `create_tiles` and `Overview` behavior at their new owners.
- `Map(*, basemap: str, height: int, theme: Theme)` owns map panes and streams.
- `Map.set_layers(layers: Sequence[Layer]) -> None` consumes draw-ordered layers.
- `Map.zoom(bbox: BoundingBox) -> None` and `Map.close() -> None`.
- `Map.create_pane(element: hv.DynamicMap) -> pn.viewable.Viewable` applies current
  framing and style to a comparison element.
- `Map.view` exposes View; `Map.point` is a Param tuple of Web Mercator click
  coordinates or None. Explorer subscribes to it for layer identification.

- [ ] Pin pan/zoom/resize/click behavior, including zoom then add-layer retaining
  the current view, and source-layer identification at the current pixel tolerance.
- [ ] Move stream creation, basemap composition, framing and pane construction
  from app.py to Map. Keep the dependency one-way: Map must not import Explorer
  or catalog loading. Use Layer only as a type where runtime imports would cycle.
- [ ] Remove the old modules without aliases and update mirrored tests/doc links.
- [ ] Run the map, raster, layer and app tests; retain nodata, categorical kernels,
  lazy full-resolution windows and Image/class/RGB switching checks.

### Task 5: let Comparison own comparison

**Files:**
- Create `src/geosave/geodata/viz/explorer/comparison.py` and matching
  `tests/geodata/viz/explorer/test_comparison.py`.
- Modify `explorer/app.py` and `explorer/docks/compare.py`.

**Interfaces:**
- `Comparison(map: Map)` owns `enabled`, `left`, `right` Param state.
- `Comparison.set_layers(layers: Sequence[Layer]) -> None` rebuilds valid
  raster/date choices, preserving valid selections.
- `Comparison.swap() -> None`, `Comparison.close() -> None` and `__panel__`.
- `Explorer.comparison` exposes this single owner to the controls dock.

- [ ] Move and extend tests for duplicate names, second-resolution dates, side
  removal, swapping, empty/one-raster cases and keeping the viewport while
  enabling/disabling swipe. Comparison must not mutate the main layer stack.
- [ ] Move raster-specific choice construction and swipe rendering into this
  module. The dock builds widgets from Comparison parameters. Remove duplicated
  comparison parameters/methods from Explorer without compatibility aliases.
- [ ] Use Map.create_pane for framing/basemaps instead of copying those rules.
- [ ] Run comparison, map, dock and app tests together.

### Task 6: verify and document Phase A

**Files:**
- Update current visualization responsibilities, public references and TODO.txt.
- Update source-mirrored tests and affected catalog/accessor callers.

- [ ] Run `uv run pytest tests/geodata/viz tests/geodata/sampling
  tests/geodata/core tests/test_layering.py`.
- [ ] Run Ruff/format and BasedPyright on affected source/tests; report exact
  scope and remaining diagnostics. Use suitable local socket/network access for
  Zarr and remote schema validation before diagnosing environment failures.
- [ ] Serve a local catalog viewer and check source toggles, vector overlays,
  filtering, identification, comparison and repeated removal in a browser.
- [ ] Document direct import/API moves, clone semantics and lifecycle rules.
  Update the cache-lifecycle TODO only when behavioral evidence supports it.
- [ ] Review the task diff against the initial workspace state; preserve all
  unrelated edits. Commit only explicit task files if committing is requested.

## Phase B: raster and vector catalog opening

This is a feature extension with an additional geodata reader contract, not a
file-move task. Resolve it after the Phase 0 catalog contract and ownership design
are reviewed, using the following investigation before its implementation plan.

- [ ] Confirm that “vector catalog” means rows referring to vector datasets,
  rather than feature visibility within one already loaded GeoDataFrame.
- [ ] Construct local raster and vector catalogs, the vector one referring to a
  small real GeoParquet file. Smoke reading the data through the existing STAC,
  storage and vector-reader modules. No I/O implementation belongs in a dock.
- [ ] Establish source kind/asset selection from actual catalog metadata; require
  explicit input if metadata cannot distinguish the intended data asset. Include
  relative hrefs, a data asset plus thumbnail, and multiple vector assets.
- [ ] Define the native vector row-reading interface at the existing geodata/STAC
  seam, supported formats and storage options. Do not add a generic viz reader
  registry, merge feature sets silently, or invent a temporal-vector container.
- [ ] Generalize catalog caches to Layer values and the Explorer membership
  coordinator to both RasterLayer and VectorLayer. Preserve raster grouping.
- [ ] Test tick/untick independently for raster and vector sources, default vector
  placement above imagery, cache reuse, filtering without deselection, and a
  failing source leaving the catalog browsable and the current map intact.
- [ ] Write the Phase B implementation plan against the proven reader contract.

## Execution recommendation

Implement isolated visualization tasks inline, one at a time, after review.
Start with Comparison, then Map and plain-layer lifecycle/table ownership; adapt
Task 1 composition work when those owners are clear. Catalog-specific steps wait
for Phase 0. The tasks share lifecycle and constructor interfaces, so parallel
implementation would create unnecessary overlap. Review the complete change
before starting Phase B.

Planning probes have run, and nine existing accessor/transition tests passed.
The production implementation and proposed new regression tests have not been
written or executed as part of creating this plan.
