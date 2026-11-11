# Native STAC publication and GeoCatalog

Status: core implementation and Python caller migration implemented, 2026-11-07.
The obsolete publication path has been replaced. See the implementation plan for
verification results and the remaining notebook migration limitation.

Implementation order and stage gates: [implementation plan](../plans/2026-11-07-stac-publication.md).

## Confirmed decisions

- Initial asset publication is COG-only; do not implement Zarr/NetCDF STAC asset
  publication or speculative format abstractions in this refactor.
- Use readable functions with blank lines separating meaningful steps. Inspect
  existing operations before adding helpers. Reuse concrete `gs.validate` for
  xarray structural validity and extend its checks when needed. Keep path/ID
  collisions and Collection conflicts with the operations that own those values;
  do not duplicate structural validation in publication helpers.
- Remove the old raster publication implementation before building replacements;
  this removal is applied. No compatibility wrappers or temporary replacement.
- Return native PySTAC Items and GeoDataFrames. `create_catalog` accepts an Item
  or a sequence of Items and replaces
  `from_items`; no new public Asset or Collection constructor.
- `to_item()` saves the selected raster assets and returns exactly one Item.
  `to_catalog()` partitions Items explicitly, uses that publication path, then
  converts Items through stac-geoparquet Arrow to GeoDataFrame.
- Dataset Item partitioning uses explicit dimension selection (`split_item` in
  the latest examples); Dataset asset partitioning distinguishes variables through
  `split_vars`. DataArray uses `split_asset=("band",)` by default, following the project's
  raster band convention. Dataset variable partitioning stays separate.
- `{var}` names a Dataset variable; `{band}` names an actual band coordinate.
  No pseudo-dimension selector called `var`.
- Retain `format_id`, explicit `id` and `format_paths`. Formatting never changes
  partitioning. Check available labels, actual rendered collisions, safe paths and
  unique Item IDs across the complete batch before any raster write.
- `split_item` defaults to `("time",)`. If the whole input has no time
  dimension, publish one Item; `split_item=()` explicitly requests one Item.
  Dataset `split_vars` defaults to `True`, retaining each variable's bands.
  One output uses asset key `asset`; multiple outputs use variable names, qualified
  by relative node paths for trees. Default Item IDs derive from observation time or interval plus other
  Item-splitting coordinate values. Default paths are `{id}/{asset}.tif`, validated
  exactly like custom templates; never silently append counters.
- Use native DataTree traversal. The selected subtree declares the grouping for
  one Item; do not group by parent paths or infer scenes from timestamp tolerance.
- Assets in one Item may have different grids and acquisition times. Metadata
  must describe those differences; no implicit reprojection or time matching.
- Unselected dimensions are untouched. Explicit slicing of an unlabeled dimension
  returns native selections without manufacturing labels. A template requesting
  such a missing label fails before writing.
- Dataset variable splitting retains each variable's internal band dimension.
- DataTree publication targets a selected observation-series root whose grid
  nodes share observation coordinates. Do not implement unequal-time union,
  static-asset replication or temporal matching; the latest user direction
  supersedes those earlier proposals. Different spatial grids remain supported.
- Item time uses a common instant or the combined known coverage interval;
  explicit input takes precedence. If no usable time exists, require input.
- Different Collection IDs may coexist. Reject conflicting definitions of the
  same ID and identify the differing fields. Do not automatically merge extents.
- Saved files supply storage metadata; explicit/xarray semantic metadata is kept
  separately. Additional customization uses native PySTAC after construction.

Independent raster writers, native STAC readers and existing Item-table conversion
remain available. Shared path expansion still supports those writers; it will be
replaced with focused selection and naming operations, not cloned into three
mixed-responsibility expand functions.

## Proposals still needing decisions

Recommendations below are not user-approved requirements. Metadata precedence,
default naming and exact public parameters are still open. Semantic property
precedence and Collection comparison rules are confirmed below. Approved behavior is listed above.
Do not infer approval from examples or from the earlier feasibility probes.

## Caller flow

```python
# Proposed API: writes raster assets and returns one native PySTAC Item.
item = selection.gs.to_item(
    "output/scene",
    id="scene-001",
    split_vars=True,
    format_paths="{var}.tif",
)

# Native customization remains available before tabular conversion.
item.add_asset(
    "thumbnail",
    pystac.Asset("https://example.org/preview.jpg", roles=["thumbnail"]),
)
catalog = create_catalog([item], collections=[collection])

# Convenience publication: same Item-producing path, then conversion.
catalog = dataset.gs.to_catalog(
    "output",
    split_item=("time",),
    split_vars=True,
    format_id="{collection}_{time:%Y%m%d}",
    format_paths="{time:%Y%m%d}/{var}.tif",
    collection=collection,
)
```

These are proposed interfaces, not executable examples of the current API.
`to_item()` performs persistence, as explicitly requested. `to_catalog()`
partitions the input into observations, calls `to_item()` for each, and converts
the results to a native GeoDataFrame. Calling both independently writes twice;
use `create_catalog` when the Items already exist.

## Intent and approaches

Use PySTAC as the metadata model, stac-geoparquet as the tabular codec, and
GeoPandas as the analysis representation. GeoCatalog is an accessor over that
representation, not a new storage format or an alternative Item model.

The selected direction is xarray selection → persisted assets → PySTAC Items →
Arrow → GeoDataFrame. Building metadata directly from xarray before saving is
possible, but does not reliably describe the storage encoding. Using rio-stac's
one-source constructor repeatedly would require undoing its Item-level metadata
choices when assembling multiple sources. Instead, use its raster inspection
approach as prior art for one small multi-source constructor. Reuse dependency
capabilities; do not copy the entire rio-stac implementation.

No new asset wrapper, source dataclass, Collection constructor, or parallel
STAC schema is proposed. First implementation supports the current COG
publication workflow. Zarr/NetCDF publication and a general vector/sampling
redesign are separate work.

## Independent layers

| Owner | Input → output | Responsibility |
| --- | --- | --- |
| Native `DataTree.subtree_with_keys` | selected subtree → relative paths and nodes | Traversal, including data-bearing roots and parents |
| Concrete DataArray/Dataset `gs.iter_slices` | explicit dimension names → labels and native selections | Lazy coordinate selection; no templates, storage or STAC |
| Dataset variable selection | Dataset → selected variables | Separate from dimension slicing; preserve observation coordinates |
| `io/paths.py` and existing template utilities | labels + templates → relative destination names | Naming and collision checks; no xarray type dispatch |
| `io/formats/geotiff.py` | selections + locations → persisted locations | Encoding and raster persistence; no STAC |
| `io/stac.py::create_stac_item` | named existing raster locations + explicit metadata → `pystac.Item` | Inspect files and assemble an Item; no xarray or catalog accessors |
| `stac/item.py`, `band.py`, `extensions/` | inspected raster metadata → native Asset/extension fields | Small internal construction steps; extensions package owns extension imports |
| concrete raster accessors and `io/stac.py` | selected xarray → saved assets → Items | Publication orchestration and explicit semantic metadata translation |
| `stac/catalog.py` | Items ↔ Arrow ↔ GeoDataFrame | In-memory conversion; no raster I/O and no `.gs` callbacks |
| `io/formats/stac_geoparquet.py` | table + collection definitions ↔ file | File metadata, relative href handling, format conformance |
| `core/vector/catalog.py` | native frame → native frame/Items | Thin user-facing operations; delegates conversion and Collection reconciliation |

Independence means a layer works with native values without its caller. It does
not mean eliminating ordinary one-way function calls. A writer must not call
`rows.gs.to_items()` to reach its own underlying conversion implementation.

## Selection and naming

The user's decisions are `format_id`, `format_paths`, optional explicit `id`,
`create_catalog(items, collections=...)`, and default single-output asset key
`asset`. Replace obsolete public names; do not add compatibility aliases.
An explicit `id` overrides `format_id` for one Item. Bulk publication must detect
repeated IDs within a Collection before writing; a fixed ID cannot name many
observations. Default IDs derive from the observation time or interval plus
other Item-splitting coordinate values; default paths are `{id}/{asset}.tif`.

Use native grouping internally; do not introduce a public `iter_slices`
accessor or return coordinate-label dictionaries that duplicate the raster:

```python
for time, dataset in ds.groupby("time"):
    # When the group is a singleton, retain its time as a scalar coordinate.
    dataset = dataset.squeeze("time", drop=False)

for group, node in scene.subtree_with_keys:
    if not node.data_vars:
        continue

    dataset = node.to_dataset(inherit="all_coords")
```

The code above illustrates native mechanisms, not permission to squeeze a
multi-entry group. GroupBy preserves duplicates together; publication must check
whether each output can be represented as a COG. Do not impose global uniqueness
on xarray coordinates or silently drop/average/mosaic duplicate entries.
Multiple grouping dimensions can introduce a stacked dimension; the feasibility
study must guide implementation so writer-facing shape and metadata stay explicit.

Preserve attrs, encoding, spatial coordinates and observation metadata. Variable
partitioning differs from coordinate grouping. Dataset variables retain their
internal bands; DataArray defaults to grouping assets by its band coordinate.
Unselected dimensions remain intact. Unlabeled dimensions must not acquire
invented coordinate metadata, and templates requiring unavailable labels fail.
Use value-specific local names: `dataset`, `array`, `item`, `asset`, `path`.

`GeoRaster` declares `expand_rasters` as an abstract method. Concrete accessors
return a DataFrame of labels and native, lazy raster slices. Dataset and DataTree
accept `split_vars=False` to retain Dataset slices, including CF bounds and
shared metadata. Tree expansion carries each node's effective inherited attrs.

Publication composes this interface twice: observation dimensions first, then
asset dimensions and variable separation. Its internal `expand_assets` operation
returns one flat table indexed by observation. It does not rename arrays, convert
them to Datasets, interpret templates, or write files. Selecting a Dataset
observation retains its static variables through normal xarray selection; tree
nodes still require matching observation coordinates.

Naming consumes explicit `variable`, `path`, and dimension columns. A sole asset
is `asset`; multiple assets use the configured split labels. Node paths remain
separate from asset names throughout formatting. Only the GeoTIFF preparation
step converts native arrays into the Dataset representation used by its writer.
`_raster_groups` and `write_groups` are removed; publication and direct GeoTIFF
writing consume the same inventory.

Native tree traversal remains unchanged. Shared-observation validation belongs
in the concrete `gs.validate` methods, not a separate validation subsystem.

Publication uses explicit split configuration to partition Items and assets.
Templates only name the resulting outputs. The slicer does not parse templates. `{var}` identifies a Dataset variable;
`{band}` identifies a band coordinate within an array. This distinction is now
confirmed. Dataset variable partitioning keeps each variable
whole, including its bands; do not silently reinterpret `{band}` as `{var}`.
IDs use observation labels; asset-only fields must not accidentally create Items.
A path lacking `{time}` does not merge distinct observations into one Item.

Validate the actual rendered names, not merely the presence of splitting fields.
`{time:%Y}` can collide for two observations in one year despite containing time.
A single observation does not require a time field in its path. Diagnostics should
show the colliding selections and suggest distinguishing fields/precision or
omitting custom formatting. Missing/ambiguous template values fail before writing.
Default names must be tested by the same checks as custom names.

`format_paths` remains `str | None` for all three xarray interfaces. The supported
DataTree structure does not need per-node template mappings.

One output has asset key `asset`; multiple Dataset variable outputs use variable
keys. Tree outputs qualify those keys with relative node paths, such as
`optical/red`. DataArray dimension-split key details remain to be finalized. Output filenames do not implicitly define
asset identity. A tree's `{group}` is the relative node path, not its Collection ID.

Selecting a subtree declares the observation boundary for `to_item`; nodes can
have different grids. Include data-bearing parents and children. Do not guess
observation groups from parent paths or equate grid identity with observation
identity. `to_catalog` may split a selected observation series by time, but cannot
combine unrelated branches merely because timestamps match. Grid nodes within the selected observation-series root share observation
coordinates. No unequal-time grouping or static replication is required. Use
`gs.validate` at publication to enforce this contract for user-constructed trees.

## Standalone Item constructor

Proposed initial surface, subject to the remaining decisions below:

```python
def create_stac_item(
    sources: Mapping[str, Location],
    *,
    id: str,
    input_datetime: datetime | None = None,
    properties: Mapping[str, Any] | None = None,
    geometry: Mapping[str, Any] | None = None,
    collection: str | None = None,
    collection_url: str | None = None,
    asset_roles: Sequence[str] | None = None,
    asset_media_type: str = "auto",
    asset_hrefs: Mapping[str, str] | None = None,
    with_proj: bool = False,
    with_raster: bool = False,
    with_eo: bool = False,
    geom_densify_pts: int = 0,
    geom_precision: int = -1,
    storage_options: StorageOptions | None = None,
) -> pystac.Item:
    ...
```

`sources` maps unique asset keys to readable files. `asset_hrefs`, if supplied,
overrides advertised locations for matching keys without changing what is read.
Recommended href rule: resolve local source paths to absolute locations and
require absolute advertised overrides. Returned Items and in-memory catalogs
must not depend on the process working directory. The file writer may relativize
copied hrefs against the output file location; the reader resolves them against
the input file location. Standalone conversion of external Items resolves
relative hrefs from their self href, or rejects unresolved relative references.
Never mutate caller-owned Items while rewriting links.
`asset_roles` applies the same list of STAC role strings to all source assets;
per-asset customization uses the returned native Assets. `geometry` is an
optional longitude/latitude GeoJSON geometry describing the observation's
footprint; otherwise derive a footprint from the source bounds. Compute the
Item bbox from the final geometry. Geometry does not claim a nodata-derived
valid-pixel footprint.

`properties` contains Item properties, including `start_datetime` and
`end_datetime` for intervals; it is not an arbitrary band configuration.
Reject inconsistent instant/interval inputs. No implicit current time or
year-1/year-9999 interval. If no usable time is known, require explicit caller input.

Projection metadata belongs to each Asset when grids differ. The first asset
must not silently define the grid of all other assets. Raster metadata describes
saved dtype, nodata, scale and offset. EO metadata is populated only when known;
do not invent wavelengths or common names from ambiguous band labels.

The three flags request extraction, not merely extension declarations. Default
inspection reads raster headers, not pixels. Statistics/histograms are deferred
from the initial API; adding them later requires explicit pixel-read controls.
Arbitrary extension declarations, additional assets and specialized metadata use
native PySTAC afterward. A native Item constructor alone does not prove schema
conformance; validate representative complete results in tests.

## Raster publication

The orchestration order is:

1. Select one observation and retain its explicit semantic metadata.
2. Slice explicitly, partition variables, format names and preflight the complete destination inventory.
3. Write COGs and obtain their actual output locations.
4. Construct the Item from those files.
5. Apply xarray-only semantic metadata through existing STAC metadata helpers.
6. Return the native Item, or collect Items and convert them to a frame.

Saved-file dtype, transform, nodata and other storage facts are authoritative.
Keep differing semantic values on their respective assets, promote only shared
values to the Item, and let explicit Item properties override inferred semantic
properties. Do not use semantic overrides to misdescribe stored pixels.
Observation time, declared footprint,
descriptive properties, band semantics and optional Collection definitions come
from explicit input metadata. Do not forward an opaque xarray header to the
standalone constructor or depend on private TIFF tags to retain this context.
Do not reuse `encoding['source']` as the publication destination.

`DataArray` owns its one-array selection, `Dataset` owns variable selection, and
`DataTree` owns traversal. Shared publication code handles one already selected
observation; it must not dispatch on all three xarray types.

`to_item()` returns one Item for the selected logical grouping. The earlier
proposal to reject differing asset timestamps is withdrawn: STAC permits
asset-specific temporal metadata and Item time ranges. A common observed instant
across all assets is not a STAC requirement. Item time uses a common instant or combined coverage interval unless explicitly
overridden. Catalog partitioning follows `split_item`; do not infer a
same-date constraint from the word observation. Mixed-resolution assets can share
an Item without resampling.

A dimension not requested for slicing remains untouched, with or without
coordinates. Native xarray positional selection does not add a coordinate label
for an unlabeled dimension. Unlabeled dimensions contribute no coordinate label to the slice metadata. Iteration positions and coordinate metadata are
different values; neither is an implicit observation timestamp.

References checked: [xarray data structures](https://docs.xarray.dev/en/v2026.02.0/user-guide/data-structures.html)
and [STAC 1.1 Asset additional fields](https://github.com/radiantearth/stac-spec/blob/v1.1.0/commons/assets.md#additional-fields).

Full catalog destination preflight must occur before the first `to_item()` write,
so a later naming collision cannot cause a partial batch. Runtime I/O or later
schema failures can still leave completed files; no transactional rollback is
promised. A returned Item refers to completed writes. Deferred publication would
need a different return contract and is outside the initial proposal.

## Collections and concatenation

Use `pystac.Collection` directly; there is no need for `create_stac_collection`.
Preserve the optional Collection definition already carried in xarray metadata.
A collection ID on an Item and a complete Collection definition are distinct.
Do not invent a license or description when only an ID is known. Collection
definitions travel alongside Items into `create_catalog` and the file
footer; Item JSON is not a container for the complete Collection JSON.

The installed PySTAC 1.14.3 smoke shows that a Collection and its clone compare
unequal with `==`, while their serialized dictionaries compare equal. Neither
object equality nor ID equality answers whether two definitions can be merged.

An explicit `collection=pystac.Collection(...)` at raster publication overrides
the carried xarray definition without mutating either. Attach a native Collection
association to returned Items. `create_catalog(..., collections=...)` accepts
explicit definitions and may collect already resolved native associations; it must
not fetch remote Collection links implicitly. Conflicting definitions entering
`create_catalog` use reconciliation rather than silent override. Conversion back
through `to_items` restores associations from the frame metadata.

Introduce one reconciliation operation in `stac/catalog.py`, used by concat and
publication. It accepts native Collections and returns a mapping of IDs to
copied native Collections. Do not expose a general-purpose equality framework.

Confirmed conflict rules:

- Different IDs coexist. Missing definitions do not erase supplied definitions.
- Equal definitions for the same ID collapse to one without mutating inputs.
- Conflicting license, providers, summaries, assets, extension fields or
  descriptive links raise a useful conflict error; never silently take the first.
- Exclude `self`, `root`, `parent`, `child` and `item` navigation links from
  definition comparison only. Compare other links and metadata. This exclusion
  does not authorize deleting or merging links. Do not resolve Collections over
  the network.
- Differing Collection definitions are rejected with field-level differences,
  including differing extents; no automatic extent or summary merge. Compare
  definitions within each Collection ID; distinct IDs may coexist.

Collection extent has spatial and temporal parts. A world bbox is an explicit
choice, not the PySTAC constructor default. Do not synthesize global coverage
when carrying an existing Collection. Preserve its stated definition on subsets.
Duplicate `(collection, id)` keys are rejected at Item-table construction and
publication boundaries. Generic GeoDataFrame concatenation is still available;
it does not promise preservation of STAC metadata.

## GeoDataFrame and file contract

Keep `create_catalog(items, *, collections=()) -> GeoDataFrame` and
`to_items(frame) -> list[pystac.Item]` as standalone conversion operations.
Use stac-geoparquet's Arrow codec, GeoPandas Arrow conversion and Arrow-backed
nested columns. Preserve metadata under `frame.attrs['stac-geoparquet']` through
GeoSave operations; arbitrary pandas operations are not a metadata guarantee.

The frame is an editable analysis view. Shapely geometry and an omitted redundant
bbox column are acceptable in memory. Serialize through the STAC writer, which
restores file requirements. Column presence alone never proves conformance.

The linked specification defines one row per Item, typed flattened properties,
nested assets/links, geographic geometry, and a bbox struct. Validate the complete
Arrow/file representation and its separate GeoParquet and STAC metadata; include
Collection definitions under the plural `collections` footer key. Keep STAC,
GeoParquet and STAC GeoParquet version numbers distinct. Reject property-name
collisions instead of silently dropping properties.

Source: [STAC GeoParquet specification](https://github.com/radiantearth/stac-geoparquet-spec/blob/main/stac-geoparquet-spec.md).

Round-trip tests must cover timestamps, intervals, nulls, heterogeneous asset
keys, extension fields and relative hrefs. A spatial reprojection in the analysis
frame must be normalized when converting back to Items. A row converted in
isolation must retain or explicitly receive CRS context; a Series does not
reliably supply it. Empty catalogs need a deliberate typed schema because the
dependency's Item parser cannot infer one from no Items.

## Layer-by-layer delivery

Implement and review these stages independently; see the linked plan for files,
interfaces and checks. Do not start by rewriting `to_catalog`.

1. Lazy DataArray/Dataset slicing and native tree traversal contract.
2. Explicit partition orchestration, default/custom naming and batch preflight.
3. Standalone saved-raster Item construction, using real COG smoke cases.
4. Collection reconciliation and `create_catalog` in-memory conversion.
5. STAC GeoParquet read/write compliance and actual file round trips.
6. Concrete raster `to_item` followed by `to_catalog` orchestration.
7. GeoCatalog integration, caller migration and obsolete implementation deletion.

The user explicitly chose deletion before replacement; old raster publication
was removed. Track broken dependents until rebuilt. Shared independent writers
remain; do not remove them incidentally. No compatibility aliases. Preserve
unrelated working-tree changes and sampling behavior.

## Evidence and remaining decisions

Fresh local smoke checks during this design review passed Items → Arrow →
GeoDataFrame → Arrow → Items and actual temporary Parquet round trips. Data and
file footer metadata matched between the original and reconstructed Arrow
tables. Geometry field metadata had equivalent CRS JSON with different whitespace,
so byte equality of every field's metadata is not a useful semantic assertion.
The smoke also confirmed that native GeoPandas writing does not serialize our
STAC metadata from frame attrs, and reading does not restore those attrs. The
format adapter therefore has real work to own. Empty dependency input raised
`ValueError`, confirming the need for an explicit empty-table path.

Existing feasibility artifacts in
`../benchmarks/2026-11-07-xarray-items/` establish the in-memory codec path for
seven examples with zero Dask tasks. Their URLs are illustrative: they do not
prove file persistence or extension validation. A separate saved-COG investigation
found why write-before-inspect matters: an encoded uint16 raster was described
using its float32 xarray dtype by the current builder.

## Open decision register

### GroupBy feasibility findings

Reproducible probe: `../benchmarks/2026-11-07-xarray-items/groupby_feasibility.py`.
Run from the repository root with `PYTHONPATH=. .venv/bin/python` followed by that
script's repository path. This is a study, not a new library implementation.

- Native time grouping and targeted squeeze preserved the grid, encoding and
  Dataset-level time bounds with zero Dask tasks during grouping.
- Native traversal/grouping preserved distinct grids with shared times.
- A plain grouped two-band Dataset wrote a COG with matching raw values/transform.
- A grouped Dataset retaining `time_bounds(time, edge)` as `time_bounds(edge)`
  failed in the existing rioxarray-backed COG writer with CoordinateValidationError.
  Extracting a variable as a DataArray instead loses that auxiliary coordinate.
  Resolve metadata capture and writer-facing coordinate handling before chaining
  publication onto this path. Dropping bounds in the probe was a diagnostic control,
  not an approved fix or evidence that a metadata round trip succeeds.
- Duplicate timestamps produce a multi-entry group that cannot be squeezed;
  publication must validate COG representability rather than globally forbid
  duplicate xarray labels. No implicit
  averaging, mosaicking or first-entry selection is permitted.
- Existing `gs.expand_rasters` methods already implement positional expansion
  into a DataFrame. Their retention/replacement must be assessed rather than adding
  another public `iter_slices` interface by default.

Earlier iterator-plus-label-dictionary examples are now superseded proposals:
native selections retain their coordinates. Evaluate native GroupBy/selection
internally; do not duplicate coordinate metadata in another return container.

Resolve these explicitly; defaults and error behavior are part of the interface.

| ID | Decision | Earliest stage |
| --- | --- | --- |
| D1 | Settled: slice unlabeled dimensions without inventing labels; missing template values fail. | 1 |
| D2 | Approved: time/interval plus other split-coordinate values for default Item IDs; `{id}/{asset}.tif` for default paths; actual collision checks and no automatic counters. | 2 |
| D3 | Approved: DataArray `split_asset=("band",)`; Dataset `split_vars=True` retains internal bands; `split_item=("time",)` with the agreed no-time fallback. | 2 |
| D4 | Settled: common instant or combined coverage interval, explicit time precedence, require input when no usable time exists. | 3 |
| D5 | Semantic precedence confirmed: saved-file storage facts; per-asset differing values; shared Item values; explicit Item semantic properties override inference. Still finalize exact constructor options and Collection override details. | 3 |
| D6 | Settled: ignore self/root/parent/child/item links for comparison only; reject other same-ID conflicts including extent; distinct IDs coexist. | 4 |
| D7 | Latest direction: shared observation coordinates within each selected root; differing grids allowed. No temporal union or static replication. Define the corresponding `gs.validate` check. | 6 |
| D8 | COG-only publication confirmed. Still review completed writes before returning, footprint/href rules and overwrite/runtime failure behavior. | Before affected implementation |

D2/D3 are approved. Multiple
split dimensions must not introduce silent duplication, missing values, temporal
matching or invented observation metadata. There is no static-replication or
per-node-template feature in the current scope.

The relative-href base is defined by the proposed absolute-in-memory/file-relative
rule above. Full core/extension and file-schema validation remains a test gate;
constructor success alone is not evidence of compliance.

These are design decisions still open for discussion, not completed features.

## DataTree construction review

Live `io/odc.py::load_datatree` partitions one matched Item sequence into band/grid
families, passing the same Items and grouping settings to each load. The intended
selected-root workflow therefore shares observation coordinates while grids differ.
Native xarray can represent more general trees, and a full catalog tree can contain
separate Collections. Neither requires publication to reconcile unrelated series.
Validate the selected root rather than adding handlers for unsupported mixtures.


## Confirmed implementation clarifications

- A Collection reconstructed from `stac_collection` may remain in memory without
  a self href. Explicit native `collection=` takes precedence over raster metadata.
- `set_self_href` assigns a document address; it does not save or upload anything.
  Preserve genuine existing Collection URLs. Never invent a JSON Collection URL
  pointing at a Parquet file. Embedded definitions live in the Parquet footer.
- A standalone JSON export must assign and save actual Collection documents
  before validating the serialized Item links. In-memory PySTAC object graphs
  can contain unresolved document locations.
- Dataset bounds use the CF `time.bounds` reference (attrs or decoded encoding)
  and a two-vertex bounds variable. DataArray uses start/end coordinates; explicit
  `.gs` conversions translate between the representations.
- Calendar cells keep their actual shared boundaries, without subtracting a
  microsecond. Temporal selection retains its existing inclusive-overlap behavior;
  changing that independent filtering policy is outside this refactor.
- Raster inventory expansion and publication share native dimension grouping.
  Repeated labels remain grouped; only singleton split axes are squeezed.
  Inventory position columns remain inspection metadata for unlabeled dimensions,
  and never become invented raster coordinates or publication identities.
- Formatting validates already-selected outputs. Explicit `id` overrides
  `format_id` without evaluating the unused template.
