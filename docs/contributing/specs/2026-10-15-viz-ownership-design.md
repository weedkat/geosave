# Visualization ownership proposal

Status: draft for review; no implementation changes authorized by this document.

Related: [current visualization responsibilities](2026-10-15-viz-responsibilities.md)
and [implementation plan](../plans/2026-10-15-viz-ownership.md).

## User workflow

```python
viewer = raster_catalog.gs.explore()
# Tick catalog rows to open imagery; untick them to remove that imagery.
viewer.add(boundaries)
combined = viewer * prediction
```

Catalog browsing remains a central feature. Catalog footprints, search, date
filters, attribute tables and row toggles remain available throughout the
refactor. Removing a catalog means the existing layer-menu action; it does not
mean removing catalog exploration from GeoSave.

The requested direction includes vector catalogs. Before defining their opening
API, align with the established GeoCatalog/GeoStackCatalog split below.
An already loaded feature GeoDataFrame remains an ordinary vector dataset;
catalog footprint geometry does not make its referenced assets vector data.
Raster and vector catalog opening must use native geodata readers, not readers
implemented inside visualization widgets.

## Prerequisite: catalog methods and variants

This prerequisite applies to catalog-specific changes. Comparison, viewport
rendering, plain raster/vector table ownership and callback/cache cleanup can
be refactored independently while retaining existing catalog integration tests.

The user clarified the existing direction on 2026-10-15:

- **GeoCatalog:** STAC Item rows complying with STAC-GeoParquet.
- **GeoStackCatalog:** grouped source rows produced by series grouping and
  collection stacking/joining; subsequent frame/chip selections retain this
  accessor and their operation-specific invariants.
- **GeoVector:** ordinary loaded feature geometry and properties.

These are accessors over native GeoDataFrames, not replacement data containers.
The visualization refactor consumes this split; it must not implement a competing
catalog taxonomy or keep grouped rows under GeoCatalog for convenience.

```python
items.gs                                    # GeoCatalog
series = items.gs.group_series()
series.gs                                   # GeoStackCatalog
stacks = series.gs.join_collections(on="optical")
stacks.gs                                   # GeoStackCatalog
frames = stacks.gs.select_frames(2)
chips = frames.gs.chips(128)
```

GeoCatalog owns Item/Collection metadata, STAC-GeoParquet persistence, STAC export
and the transition into series. GeoStackCatalog owns grouped source semantics,
collection matching, frame/chip planning and reconstruction. Its persistence is
GeoParquet preserving the grouped schema; it must not advertise STAC Item export
for rows that are no longer Items. Exact class inheritance and implementation
details remain with the existing split design, not this visualization plan.

There are two independent questions:

1. **Row state:** Item, single-collection series, joined collections, selected
   frames, or chips. This determines source/time/grid selection semantics.
2. **Referenced data:** raster, vector or an explicitly supported combination.
   This determines which reader and operations apply.

GeoVector owns operations on loaded feature geometry. GeoCatalog and
GeoStackCatalog expose their respective source-description contracts. GeoSample
owns reading one supported catalog row. Viz consumes those contracts instead of
defining another set of catalog variants. A `sources` column alone is not a
GeoStackCatalog contract.

Current behavior checked before the planned split; the matrix is evidence to map
onto the two target accessors, not the proposed public class hierarchy:

| Row state | Group series | Join collections | Select frames | Chip | Row reader |
| --- | --- | --- | --- | --- | --- |
| Items | Produces series | Requires grouping | Requires grouping | Requires grouping | Reads the Item |
| Single-collection series | Warns and returns unchanged | Matches collections | Requires a dated timeline | Requires a reference grid | Preserves native source grids |
| Joined collections | Warns and returns unchanged | Not single-collection input | Slices established sampling positions | Requires a reference grid | Preserves established source correspondence |
| Selected frames | Warns and returns unchanged | Refused after selection | Refused after selection | Produces chips | Reads selected positions |
| Chips | Warns and returns unchanged | Refused | Refused | Refused | Honors the chip grid and padding |

This table records the current raster workflow, not a promise that pixel/grid
operations apply to vector assets. `to_xarray()` is the row reader: Item rows
return a Dataset and sample rows return a DataTree. Select a collection's node
to obtain its Dataset. Item persistence and grouped persistence also differ:
Item rows can reconstruct STAC Items; grouped rows must preserve their source
selection schema instead of masquerading as Items.

Nine focused accessor/transition tests passed during planning. A smoke produced
all five states and confirmed the current implementation still resolves them to
GeoCatalog and CatalogLayer; GeoStackCatalog was not present in the inspected code.
However, an ordinary feature table containing only a business `sources` column
also resolves to GeoCatalog. The current `sources`-presence predicate is weaker
than the actual grouped-row contract.

The first milestone therefore integrates the existing two-accessor split, maps
operation-by-state contracts to their intended owner, and tests malformed/partial
schemas, empty tables, column subsets, persistence and mixed-stage concatenation.
Schema and validity rules belong in geodata, with operation-specific validation
at its actual owner. Viewer code delegates to them. Do not introduce a separate
class for every intermediate stack state or duplicate the rules in the layer factory.

Layer construction must recognize both GeoCatalog and GeoStackCatalog as
browsable source catalogs. It currently checks only GeoCatalog; after the split,
that check must not accidentally render a GeoStackCatalog as an ordinary vector
dataset. Item opening may group sources where the existing workflow requires it;
stack/frame/chip opening must preserve the row's declared selection and use its
existing row-reader path.

Consistency means a method retains one meaning and has explicit preconditions;
it does not mean every catalog variant supports every method. Preserve intentional
warn/drop policies for dirty source data, and distinguish those from invalid
operation order. Settle vector-catalog capabilities only after this matrix is
reviewed, especially where no raster reference grid exists.

## Evidence from the current implementation

Planning smokes on 2026-10-15 reproduced these results:

- After removing a vector layer, Explorer removes its own layer watcher, but
  Style and AttributeTable retain the layer in their caches. A weak reference
  remains live after the local layer reference is dropped and GC runs.
- Composing a dark Explorer using `*` constructs a light Explorer and changes
  the original shared layer's theme to light.
- Tabulator header filtering followed by a plain `VectorLayer.filter_rows`
  predicate and an explicit refresh preserves original row positions: an area
  filter plus text `rice` produces original row index `[2]`.
- Setting a native `pn.Param` pane's object to None does not fully disconnect
  an existing slider: changing the old slider can still change the layer.
  Cleanup cannot be inferred from clearing a layout or changing that property.

HoloViews documentation defines `*` as overlay composition and `+` as a layout.
A local HoloViews smoke also confirmed that a new Overlay retains the input
elements by identity. That establishes overlay intent, not a lifecycle contract
for an application with tables, catalogs and Python callbacks.

## Approaches considered

1. **Move files only.** Put map and raster helpers under Explorer. This improves
   navigation but leaves caches, callbacks and comparison state scattered.
2. **Assign ownership to existing domain objects and three focused modules.**
   Explorer coordinates membership and layout; Map, Comparison and Attributes
   own their actual state and callbacks. Recommended.
3. **Introduce a generic application store, event bus and plugin registry.**
   This adds interfaces without a demonstrated need. Excluded from this plan.

## Proposed hierarchy

```text
viz/
  plot.py                    # static figures
  shapes.py                  # shared geometry preparation
  style.py                   # shared colors and presentation defaults
  explorer/
    app.py                   # composition, membership and layout
    map.py                   # View, tiles, Map and viewport/click streams
    raster.py                # Overview and raster window sampling
    comparison.py            # Comparison state and swipe rendering
    attributes.py            # one native attribute table and its bindings
    layers/                  # raster/vector/catalog behavior and data caches
    docks/                   # controls and placement of the owned objects
```

Move the existing root map/raster modules; do not leave forwarding modules.
Keep time synchronization in Explorer initially: it is short coordination across
the current layer membership. Keep catalog selection-to-membership coordination
there too. Extract either only if implementation reveals a substantive separate
invariant, not to reach a target line count.

## Ownership rules

| Owner | State and behavior | Does not own |
| --- | --- | --- |
| Explorer | layer membership/order, active layer, shared time, theme, dock placement; coordinates catalog selection | raster reads, native table predicates, swipe construction |
| Map | basemap, current View, map panes, RangeXY/PlotSize/Tap streams, framing and click coordinates | catalog loading, table selection, comparison choices |
| Comparison | enabled/left/right parameters, raster/date choices, swapping and swipe panes | main layer membership or source reads |
| Attributes | one VectorLayer's Tabulator, header filters, native selection links, visible row indices and their subscriptions | opening catalog sources or choosing the active layer |
| RasterLayer | raster representation/settings, Overview cache and identification | page widgets or closing caller-owned raster data |
| VectorLayer | geometry/settings, text filtering, spatial queries and selection | table construction or header-predicate parsing |
| CatalogLayer | catalog footprints, date predicates, source-to-opened-layer cache and selected-source opening | Explorer membership or page layout |
| Dock | widgets for one task, layer-specific widget caches and subscriptions it creates | domain data loading or duplicate copies of Comparison/Attributes state |

Each object that subscribes to a parameter or stream retains the handle needed
to detach that subscription. Each cache has one named owner and a release rule.
Use native Panel/Param/HoloViews mechanisms; do not add a generic resource manager.

## Lifetime

`Explorer.remove(layer)` is the single membership-removal path, including calls
from `run_action`. It detaches Explorer's own subscription, releases that layer's
Attributes object and asks docks to release their cached controls. It then
repairs the active layer, comparison choices and map composition.

`Explorer.close()` releases its subscriptions, Map, Comparison, Attributes and
docks. Calling close twice is harmless. It never calls close on native xarray
or GeoDataFrame inputs supplied by callers.

`Dock.release_layer(layer)` releases its own cache entry and owned subscriptions;
`Dock.close()` releases the dock's subscriptions and caches. A dock without
layer-specific state has no per-layer work.

Native widget objects can contain internal reference cycles. Release tests must
drop the real view references and measure reachability with weak references and
GC, as well as prove that old callbacks cannot modify an active Explorer. Do not
manipulate private Panel callback registries. A retained caller reference to a
Layer may legitimately retain that layer's data caches.

Catalog toggles keep their current meaning: tick opens selected data, untick
removes that data from the map while the catalog stays available. Hiding a
loaded layer changes visibility only. The proposed existing Remove behavior
keeps already opened data layers on the map if the catalog itself is explicitly
removed; it releases the catalog's association with this Explorer. This is a
proposal, not an instruction to remove catalog browsing from the interface.

## Tables and filtering

Explorer obtains one Attributes object per vector layer through
`get_attributes(layer)`. The table dock chooses which one to display. Multiple
calls retrieve the same owner, rather than creating competing filter owners.

Tabulator remains the column-filter parser. Search and catalog dates remain
layer predicates. Attributes bridges native `current_view.index` to
`filtered_rows`, using original positional indices. Filtering never changes the
selected source rows or silently closes opened data.

Explicit subscription handles are required for selection and filter refresh.
If a native bidirectional binding cannot be fully detached through its public
interface, use two explicit Param watchers for that binding. The predicate
refresh smoke supports using `table.add_filter(layer.filter_rows)` with owned
refresh subscriptions rather than an untracked reactive function filter.

## Overlay composition

`viewer * other` remains an overlay expression: it returns a viewer with the
additional data drawn above existing data. It does not mutate the original
viewer's membership, theme or current date while constructing the result.

Native raster/vector data are shared by reference; pixels are never deep-copied.
For application-specific state, create fresh layer settings and fresh callbacks.
The proposed `Layer.clone(copies=None)` contract copies presentation/selection
state and uses a plain identity dictionary to preserve catalog-to-opened-layer
relationships. Catalog caches and the active stack must refer to the same copied
layers, including when a source was already opened before composition.

This is an application-state policy, not a claim that HoloViews deep-copies its
elements. A custom Layer must define cloning to participate in this composition
path. Direct reuse of a live layer across independently managed tables is not a
new supported contract in this refactor.

## Raster and vector catalogs

The ownership refactor must remove the assumption that every catalog-opened
layer is a RasterLayer: the target is `open_selected() -> list[Layer]` and
`opened_layers -> list[Layer]`. Raster collections keep their existing temporal
grouping and lazy reading behavior. Vector assets produce native GeoDataFrames
and then VectorLayers; their feature geometry is distinct from catalog footprints.

Vector loading is a second milestone after the ownership refactor. First smoke
a small STAC/GeoParquet catalog referring to a real vector file through existing
storage and vector readers. Establish asset selection, URL resolution, CRS and
stable source keys before settling a new public row-reader signature. Multiple
vector files remain separate layers initially; do not silently merge features or
invent temporal-vector behavior. Ambiguous or unsupported assets require an
explicit selection/error; catalog footprint geometry cannot identify asset kind.

Default placement is raster imagery below opened vector overlays, catalog
footprints above both. Reordering within the data layers remains available.
Turning off a source row removes only its associated layers, not unrelated
overlays. Failure to open a row leaves the catalog available and reports the
source error without committing a partially updated selection to the map.

## Acceptance

- Canonical schema and method preconditions distinguish catalog state from asset
  kind, including ordinary vectors with a business `sources` property.
- Items resolve to GeoCatalog; series/joined/frame/chip tables resolve to
  GeoStackCatalog, including after round trips. Both remain explorable.
- Existing accessor workflow and catalog browsing survive the refactor.
- Removed layers leave no references in viewer-owned caches and no live
  callbacks into that viewer; repeated remove/re-add does not multiply updates.
- Closing one composed viewer does not disconnect another viewer.
- Composition preserves source-layer identity relationships without computing
  or copying native data.
- Native header filters, text/date filters, original row positions, selection
  and map identification remain consistent.
- Numeric/class/RGB switching, categorical resampling, nodata, lazy window reads,
  unsorted dates and sub-minute timestamps retain their existing tests.
- Tests mirror the final source modules; notebook files remain untouched.

## References

- [HoloViews composing elements](https://holoviews.org/user_guide/Composing_Elements.html)
- [Current visualization responsibilities](2026-10-15-viz-responsibilities.md)
- [Implementation plan](../plans/2026-10-15-viz-ownership.md)
