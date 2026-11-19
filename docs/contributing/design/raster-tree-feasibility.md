# GeoDataTree spatial contract and publication feasibility

Status: feasibility study and proposals, 2026-11-17. No runtime contract has
changed. The shared-GeoBox restriction was not adopted. This study informs
the [raster-tree specification](raster-tree.md); it does not supersede its
approved grouping, metadata extraction, split, merge or concat decisions.

Review update: the user approved Option C's responsibility split: independent
raster leaves and a GeoDataTree interface separate from GeoRaster. The
[Option C refinement](#option-c-refinement) records that direction. Individual
method contracts are tracked in the main specification. The seven core workflow
choices have since been approved there; this study's experiments and alternatives
remain evidence rather than the current decision ledger. Earlier
conditional recommendations for a one-GeoBox tree are retained as study history.

Publication clarification: one staged Item represents one observation or
derived product. Independent static/reference rasters are excluded from dated
observation Items. Earlier probes that attached elevation to imagery Items
remain evidence of permissive low-level I/O, not examples of the approved
staging contract. A successful write or reload does not establish observation
membership.

## Finding

A shared-GeoBox tree is feasible and still has a purpose beyond stacking or
STAC staging: it preserves separately described Datasets over one spatial
domain. However, a shared grid does not make those Datasets automatically
mergeable, temporally synchronized, ready for a model, or writable as one file.

Publication should not require a tree under either spatial policy. Its input
relationship is explicit Item-to-asset membership. Reading a catalog assembles
new analysis data; it is not the inverse of publishing an xarray hierarchy.
These distinctions matter more than changing a validator.

## Evidence and limits

The reproducible probe in `scripts/experiments/raster_tree_contract.py` uses
native xarray and existing GeoSave functions. It implements no proposed public
methods. Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/geosave-study-mpl .venv/bin/python scripts/experiments/raster_tree_contract.py
```

It writes synthetic files and `results.json` to a new
`/tmp/geosave-tree-contract-*` directory. Some probes deliberately assert current
rejections or deficiencies; they are a dated study, not desired regression
behavior to preserve after redesign.

Checked with xarray 2026.4.0, odc-geo 0.5.1, odc-stac 0.5.2, Dask 2026.3.0,
PySTAC 1.14.3 and Zarr 3.2.1.

| Probe | Observed result | Implication |
| --- | --- | --- |
| One GeoBox, separate timelines and static data | Tree construction, grid validation and effective extraction executed zero Dask tasks; extraction shared source pixels | Grouping is structural and can remain lazy |
| Merge the same leaves into one Dataset | Exact merge rejected different time indexes; outer merge grew labels from one date to two and changed uint8 to float32 | A common grid does not remove coordinate-namespace differences |
| Repeated variable name with different values | Native merge rejected it | Namespaced leaves have value independently of spatial differences |
| Chipping with leaf-local spatial coordinates | Chip construction executed zero tasks; tensors had image `(2, 2, 2, 2)`, label `(1, 1, 2, 2)`, static `(1, 2, 2)` shapes | Spatially aligned inputs can retain different non-spatial shapes |
| Request shared time context or temporal sampling | Both rejected different leaf timelines | Temporal checks remain operation-specific |
| Put shared spatial coordinates on the root | Native construction rejected a different child grid; current chipping then failed because the root retained uncropped coordinates | Coordinate placement and transformations need an explicit design; one-grid validation alone is insufficient |
| Zarr hierarchy round trip | Group order, effective attrs, dimensions, grids and pixel values preserved | Structural persistence serves a different purpose from catalog publication |
| Earlier publication probe without RasterStac | Two Items, five COGs: two image files, two SCL files and one shared elevation file; GeoParquet reopened successfully | Mechanically possible; the imagery/elevation pairing violates the subsequently agreed observation-membership contract |
| Earlier dimension-subgroup staging probe | Two explicit Items accepted dated image leaves and one shared elevation href; zero Dask tasks; effective source attrs preserved | Low-level asset addition does not enforce observation membership; the static pairing is superseded |
| Load that catalog onto one target grid | Image values and packing preserved; SCL nearest resampling matched the expected values; classification flags restored | An aligned analysis result can be assembled from independently gridded assets |
| Compare source and loaded structure | Static elevation gained a time dimension; source group title and hierarchy were not reconstructed | Catalog loading is not an object-identity round trip |
| Load two half-pixel-shifted sources at the same CRS/resolution through the native-grid loader | One 4 by 5 output replaced two 4 by 4 source grids | Current CRS/resolution grouping does not guarantee original pixel-grid preservation |

Band names and classes in the publication probe use existing `with_eo=True`
and `classification.write_band` capabilities explicitly. The existing default
publication route does not yet wire both consistently. The study does not
claim that an unmodified high-level publisher provides the new workflow.

The earlier dimension-subgroup probe checks metadata staging only; it writes no asset
files. The static Dataset retains `(y, x)` dimensions and its own coarse grid,
while image assets describe their finer grid. Inherited Dataset attrs are not
automatically copied into Item properties. These structural checks do not make
the static raster part of the image observation. The scripts retain the earlier
experiment for reproducibility; they are not acceptance tests for the new
publication contract. Observation-membership enforcement is not implemented.

The first sandboxed Zarr run stalled. A bounded diagnostic stopped after 25
seconds in Zarr's asynchronous I/O loop. The same probe completed outside that
sandbox. This was an execution-environment limitation, separate from the
reproduced parent-coordinate crop failure.

Existing publication and effective-extraction checks also passed: 20 tests,
with no skips or deselections, covering `test_raster.py`, `test_publish.py`, and
the four `test_to_dataset_*` cases in `test_datatree.py`. This is not a complete
suite, remote-storage, model-training or performance validation.

## Observation identity and duplicate time labels

The positional splitting direction is now approved; publication metadata
validation remains open. A bounded local probe with odc-stac 0.5.2 wrote two
4-by-4 rasters and described them as different Items with the same datetime.
Their first pixel values were 1 and 2. It loaded both through GeoSave's current
`stac.load_dataset` using the same output GeoBox and lazy chunks:

```python
grouped = stac.load_dataset(items, geobox=grid, chunks={}, groupby="time")
separate = stac.load_dataset(items, geobox=grid, chunks={}, groupby="id")
```

| Operation | Observed result |
| --- | --- |
| Load with `groupby="time"` | One output plane, one time label; first pixel 1 after default fusion |
| Load with `groupby="id"` | Two output planes, identical time labels; first pixels 1 and 2 |
| Plan either lazy load | Zero Dask pixel tasks; pixels were explicitly computed afterward for verification |
| Inspect default loaded coordinates | Time, spatial coordinates and GeoSave period coordinates; no source Item ID coordinate |
| Native `separate.groupby("time")` | One group containing two positions |
| Native positional `isel(time=i)` | Two separate 2D slices, each retaining the same scalar time |
| Current `RasterStac.from_dataset(separate)` | Fails when squeezing the length-two label group |

This matches the [ODC grouping contract](https://odc-stac.readthedocs.io/en/latest/_api/odc.stac.load.html):
grouping determines which Items contribute to one pixel plane. Installed source
uses each input index for `groupby="id"`, and the first grouped Item's nominal
datetime as the output label. The string `"id"` does not imply an Item-ID
coordinate is returned. Source Items and loaded observations need not have a
one-to-one relationship.

The same probe added static `height(y, x)` to the loaded Dataset, then selected
`dataset.isel(time=0)[["height"]]`. Height stayed 2D but the selected Dataset
carried the scalar time coordinate. Checking time presence alone would therefore
accept a static input as dated. A one-off native merge emitted an xarray warning
about its future default `compat`; the observed assertions all passed.

Approved direction: preserve each existing dimension position
when structurally splitting, even if labels repeat; leave pixel grouping/fusion
with explicit loading or preparation. Treat the scalar time as a description
of each prepared observation. Across independently prepared branches, retain
explicit publication membership instead of matching repeated times or assuming
the same positional ordering. Without `names`, dimension labels become node
names and duplicates raise; explicit `names` permits repeated labels while
keeping child names unique. Item IDs belong to publication.
The alternative of requiring a persistent observation-ID coordinate was not
selected; it would add a metadata contract to loading and derived products.

### Native child promotion

A separate native-DataTree probe compared effective metadata before and after
`tree.drop_nodes("s2").assign(tree["s2"].children)`. The children were promoted,
the original tree retained its groups and pixel arrays were shared. However,
the removed group's `title` attr and scalar `time` coordinate were absent from
the promoted leaves. The generic native operations do not implement GeoSave's
agreed effective-metadata preservation. A small parent-scoped operation can
reuse native tree mechanics while preserving this metadata and checking name
collisions. The approved interface is now `parent.gs.ungroup(name)`, selecting
one direct child group and returning a new containing subtree.

## Why retain a tree on one grid?

A Dataset already supports differently typed variables and per-variable units,
packing and flags. Different units alone do not justify a tree. Image/SCL
file separation can be expressed by selecting two Datasets from one Dataset.

A tree earns its place when Dataset groups have independent identities:

```text
analysis                         one shared spatial GeoBox
├── observations                 Dataset: dated red, nir; source attrs
├── reference                    Dataset: class on a different time axis
├── predictions                  Dataset: class; model/run attrs
└── terrain                      Dataset: static height
```

Here, `reference/class` and `predictions/class` are distinct variables. The
group titles and provenance have separate owners. Flattening requires decisions
about names, coordinate alignment and attrs. Shared parent attrs still provide
one update location, with effective extraction preserving local overrides.

Native DataTree supplies hierarchy, selection, traversal and grouped storage.
GeoSave adds the raster structure contract, effective attrs and reviewed
restructuring operations. For a flat set of uniquely named variables with one
timeline, stay with Dataset. Tree construction should remain optional.

This role is dataset composition. Tensor channel stacking happens later within
a chosen Dataset. STAC asset association happens separately at publication.

## Renewed conversion contracts

These are proposed responsibilities, not final new method signatures.

| Conversion | Required meaning | Explicitly not promised |
| --- | --- | --- |
| Dataset to tree | Wrap intact under a name, or partition variables using approved ordered groups | Item splitting, resampling, automatic tensor/channel meaning |
| Selected node to Dataset | Existing `node.gs.to_dataset()`: local data, inherited coords and effective attrs | Gathering descendants or flattening a tree |
| Several leaves to one Dataset leaf | `merge` for distinct variables; `concat` for corresponding variables along a stated dimension | Automatic name repair, time alignment, unit conversion or mosaicking |
| Dataset snapshots to published Items | Explicit Item metadata and named asset Datasets; write assets and describe their stored representation | A required tree, implied Item identity, or preserving arbitrary hierarchy |
| Items to catalog | Native Item table and explicit persistence | Pixel loading or analysis-grid choice |
| Catalog selection to Dataset | Load chosen assets/bands and observations onto a selected output grid through ODC | Reconstructing original nodes, Dataset attrs, or static-variable dimensionality |
| Catalog selections to tree | Independently load named Dataset groups onto one target GeoBox, then construct a tree | One automatic interpretation of all collections, asset keys or time axes |
| Source asset to Dataset | Open a selected raster href to retain that file's grid and bands | Recovering an original multi-file or time-cube hierarchy |
| Tree to/from grouped storage | Preserve hierarchy and applicable metadata using Zarr/NetCDF | Catalog discovery or STAC semantics |

For `merge` and `concat`, the final output remains a tree with a Dataset leaf;
extract that leaf when a Dataset is wanted. The approved local extraction
method must not silently acquire descendant-combination behavior.

The main routes are therefore:

```text
source Datasets / selected tree leaves
                │ explicit Item metadata + asset membership
                ▼
         files + STAC Items → catalog
                                │ select rows/assets, grid, time grouping
                                ▼
                      analysis Dataset(s)
                           │ optional named organization
                           ▼
                      analysis tree
                           │ leaf selection or explicit merge/concat
                           ▼
                         Dataset

original tree ────────────── Zarr / NetCDF ────────────── restored tree
```

The direct Dataset route should remain available. A catalog reader need not
construct a tree just to extract a Dataset immediately afterward.

## Caller composition before adding another loader

For one collection and one timeline, load a Dataset and then partition its
variables. For different collections or independent timelines, load groups
separately. The latter is already expressible without a new loader abstraction:

```python
image = image_rows.gs.to_dataset(
    geobox=grid,
    bands=["red", "nir"],
    chunks={},
)

reference = reference_rows.gs.to_dataset(
    geobox=grid,
    bands=["class"],
    resampling="nearest",
    chunks={},
)

tree = xr.DataTree.from_dict({"image": image, "reference": reference})
```

This is a composition sketch, not a tested external catalog example. It shows
where target-grid choice and group identity belong. A future catalog-to-tree
convenience should reuse these loading paths and an explicit naming recipe,
rather than invent a second asset-reading or temporal-grouping implementation.
Do not design a general query-plan object or nested configuration language now.

Publication has a different input relationship:

```python
assets = {
    "image": image_10m,
    "scl": scl_20m,
}
```

Each value is a prepared file-sized Dataset. The source could be a standalone
Dataset or an extracted tree leaf. Item identity and temporal metadata are
provided separately. Multi-Item publication adds explicit membership for each
Item; image and SCL must belong to the same observation within each Item.
Independent static/reference inputs remain separate from those Items. The final
public spelling, write-result association and observation checks remain open.

## Alternatives at the responsibility seam

| Design | Benefits | Costs and excluded behavior |
| --- | --- | --- |
| Dataset plus named mapping only | Smallest additional interface; sufficient for flat image/mask exports | Callers manage group attrs, nested selection, transformations and grouped persistence; gives up the agreed hierarchy behavior |
| One-GeoBox GeoDataTree | One spatial reference across an independently described Dataset hierarchy; fits existing chipping and shared spatial tensor context | Multiple regions/resolutions remain separate objects; alignment is explicit before composition; storage readers and transformations need migration |
| General raster hierarchy, with a smaller accessor | Keeps independent source grids; structural grouping remains useful | Must reconsider inheriting the single-raster `GeoRaster` interface; remove or explicitly scope whole-tree spatial properties; aligned consumers retain their checks |

The third option is a real redesign alternative, not necessarily more spatial
fallback logic. It would make the tree primarily structural and keep spatial
processing at leaves or aligned consumers. It is preferable to claiming every
general hierarchy is one raster while returning ambiguous spatial properties.

Recommendation: if GeoDataTree is intended for spatial analysis composition,
choose the one-GeoBox contract and make trees optional. Keep general source
collections in native containers and STAC. Do not justify the restriction only
by current chipping code or use the tree as mandatory publication staging.

If retaining arbitrary source hierarchies as first-class GeoSave processing
objects is a requirement, choose the smaller general accessor instead. A
shared-grid requirement would then move complexity into callers rather than
remove it. This product-scope decision remains for review.

## Migration consequences of choosing one GeoBox

1. Define the invariant across raster leaves: exact CRS, affine transform and
   spatial shape/extent. Keep units, dtype, packing and time axes independent.
   Existing `same_grid` permits different extents and is not sufficient.
2. Keep native xarray mutation and file opening native. Validate the GeoSave
   contract at relevant operations; construction cannot permanently guarantee
   it after arbitrary native edits. Do not silently resample a loaded hierarchy.
3. Audit crop/reproject and inherited coordinates before enabling the contract.
   A transformation must update all affected coordinate owners and leave one
   output GeoBox. The observed root-coordinate failure needs a separate fix.
4. Separate publisher asset membership from tree traversal first. The current
   `RasterStac.to_cog` calls `item_node.gs.region`; enforcing a single-grid
   `region` immediately would break its existing mixed-grid publication route.
5. Preserve catalog-to-Dataset loading and add no automatic tree restoration.
   Review the misleading native-grid guarantee: CRS/resolution grouping alone
   cannot preserve shifted grids. Exact source-grid access is a different path.
6. Keep tensor values per leaf and spatial context shared. Keep temporal
   agreement checks when shared time context or temporal sampling is requested.
   Leaf representation and feature-unit rules remain separate decisions.
7. Verify replacement publication, direct asset reads, catalog assembly,
   effective attrs, lazy transforms and grouped persistence before retiring
   RasterStac. Existing tests are evidence to migrate, not a reason to retain
   conflicting meanings.

## Option C refinement

The structural mismatch is visible in the current inheritance:

```text
GeoRaster                     spatial meaning of one raster
├── GeoDataArray
├── GeoDataset
└── GeoDataTree               hierarchy forced to expose singular properties
```

`GeoDataTree.geobox` currently tries common-grid validation and returns None
when it fails. Inherited `crs`, `is_georeferenced`, `bounds`, `resolution` and
`transform` then depend on that result. Supporting more special cases in those
properties is not required to retain native multi-grid hierarchy.

The proposed interface separation is:

```text
GeoRaster
├── GeoDataArray               one array's spatial properties
└── GeoDataset                 one Dataset's spatial properties

GeoDataTree                    structure and operations over raster leaves
```

No new wrapper, prepared-tree class or strictness mode is needed. A native
DataTree may happen to have aligned leaves; alignment is a relationship checked
when an operation needs it. It is not the identity of the container.

Native coordinate inheritance still applies: shared dimensions/indexes must
align with ancestors. Independent grids/timelines belong on separate branches
without a conflicting parent coordinate. Keep the agreed raster-leaf convention
and effective-attrs rule; neither requires siblings to share a spatial grid.

### Proposed method ownership

| Surface | Option C treatment |
| --- | --- |
| Named construction, variable/dimension split, dissolve | Structural tree operations; preserve leaf pixels and applicable metadata |
| `validate()` | Check GeoSave's structure and each raster leaf; no universal same-grid requirement |
| `node.gs.to_dataset()` | Keep implemented local extraction with inherited coords/effective attrs |
| `merge` / `concat` | Validate relationships among selected leaves, then perform the explicit combination; do not silently resample |
| Singular `geobox`, `crs`, `transform`, `resolution`, `bounds`, spatial-axis properties | Remove from the general tree interface; inspect an extracted Dataset or explicitly establish a common grid for a consumer |
| `crop(region)` / `reproject(target)` | Tree-owned traversal delegates to Dataset operations; inputs may be independent; native inheritance constraints must still be preserved |
| `crop_to_grid(target)` | Each selected leaf must be compatible with the target pixel lattice; differing input extents do not inherently require reprojection |
| `to_tensor()` pixels | Convert independently, keyed by full relative leaf path; no global spatial check for pixels alone |
| Requested shared tensor grid/time context | Require the relevant relationship among participating leaves; reject disagreement rather than introduce fallback schemas |
| Temporal `sample` | Keep an explicit synchronized-window operation with its own time-axis requirement; do not redefine validity of other trees |
| Implicit footprint/timeline aggregation | Do not imply one raster extent or one observation axis; initially keep aggregation with consumers that need it, rather than add a generic summary subsystem |
| Zarr/NetCDF and per-leaf raster writing | Preserve structure or write explicitly prepared leaves, respectively |

For callers needing different spatial context per model input, the existing
Dataset tensor conversion can be called on individual extracted leaves. A new
tree-wide per-input context configuration is not necessary to support general
tree storage or pixel-only conversion. Numerical feature layout/units remain
their separate unresolved contract.

This reduces the tree's interface while retaining useful operations. A
geographic crop does not require globally aligned pixels. Chipping does. Both
can operate on the same native container under different stated requirements.

### Focused evidence

The study script now includes `general_tree_probe`. It constructed nested
10 m image and 20 m SCL leaves with attrs on two ancestors. Construction,
validation, effective extraction, geographic crop planning and explicit
reprojection planned zero Dask tasks. The source arrays and grids remained
unchanged. A common geographic crop produced 2 by 2 image pixels and 1 by 1 SCL
pixels. Existing Dataset conversion produced independent tensors shaped
`(2, 4, 4)` and `(1, 2, 2)`. Explicit nearest-neighbour reprojection to the image
GeoBox matched expected repeated SCL values.

This is evidence for the mechanics; tree pixel-only conversion still has its
old global check. The earlier parent-coordinate crop failure remains relevant
under either design. One exploratory run emitted Rasterio's
`NotGeoreferencedWarning` during execution despite the checked output grids and
values; this study does not diagnose that dependency warning.

### Consumer changes

The concrete consumers of a single tree GeoBox are `RasterChipper`, shared
spatial tensor context (`read_context` and `read_centroid`), and the chunked
ML prediction path. They should request a common GeoBox from an explicit
validation/resolution operation over their inputs. Reuse one small mechanism
instead of repeating agreement logic in those callers; its final public or
internal spelling is not decided here.

`RasterStac.to_cog` currently uses `item_node.gs.region`. Replacement publication
must compute the selected assets' Item footprint independently of a tree's
single-raster properties. This is also necessary for accepting standalone
Datasets and multiple trees as sources.

Catalog loading can produce one Dataset on an analysis grid, or independently
loaded Dataset groups organized in a tree. Option C permits the latter to keep
different grids. Exact source preservation still requires a loader that actually
preserves source grids; the observed CRS/resolution grouping limitation is not
resolved by changing the result container. Neither route restores arbitrary
published hierarchy without an explicit recipe.

The user approved Option C with this smaller interface. It retains geographic
differences without claiming one raster-wide spatial identity. This approves
the ownership split, not a blanket expansion of every operation to every
possible grid configuration. The main specification now records the seven
approved core workflow decisions and lists the remaining interface details.

## Dependency references

- [xarray hierarchical data](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html): parent-child alignment, independent sibling time axes and coordinate inheritance. Local probes use the installed version rather than assuming documentation matches it exactly.
- [ODC STAC load](https://odc-stac.readthedocs.io/en/latest/_api/odc.stac.load.html): constructs a Dataset; output-grid selection and observation grouping are loading decisions.
- [STAC Item specification](https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md): Item metadata and named asset associations.
- [STAC projection extension](https://github.com/stac-extensions/projection): projection fields can describe assets independently when they differ from Item defaults.

Live GeoSave owners checked: `core/raster/datatree.py`, `dataset.py`,
`utils/context.py`, `sampling/chipper.py`, ML dense preprocessing/prediction and
chunked prediction, `stac/raster.py`, `publish.py`, `load.py`, vector catalog
conversion, GeoTIFF batch writing, and grouped Zarr reading/writing.
