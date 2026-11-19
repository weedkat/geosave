# GeoDataTree raster structure and separate STAC publication

Status: core tree restructuring and explicit STAC publication implemented;
remaining interfaces listed below. Checked 2026-11-17.

This specification consolidates the agreed redesign. GeoDataTree is independent
of STAC: native raster leaves, effective metadata, group/split/merge/concat/ungroup,
spatial preparation and numerical conversion. Each raster leaf owns its grid;
parent-coordinate repair is not part of the contract. Publication accepts a native Item
and explicit asset Datasets. Implementation status is not implied by approval;
see the delivery ledger at the end.

Latest review: Option C is approved. GeoDataTree owns a general hierarchy of
raster Datasets and will have its own interface, separate from the single-raster
GeoRaster base. Different leaves may have different grids; operations and
consumers check alignment when needed. A shared GeoBox is not tree validity.
Static variables stay once beside a subgroup of dimension slices; separate
`merge`/`concat` methods are approved.
All seven subsequent core workflow choices are approved: native subtree scope,
replacement of selected restructuring inputs in the returned tree, exact
combination alignment, shared-only tree tensor context, synchronized temporal
sampling, explicit named Dataset assets per Item, and composed loading before
adding a catalog-to-tree interface. Remaining details are listed separately.
Concat also requires matching input variable sets. The caller now explicitly
selects a node and merges/concatenates all its children; this supersedes the
earlier per-operation input-path proposal. The dimension subgroup layout is
approved and must be supported by STAC staging through explicit asset
membership. Each staged Item represents one observation or derived product;
independent static/reference rasters must not be grouped into its dated assets.
This supersedes the earlier shared-elevation example.

The latest method review approved decisions 2A through 6A: dimension slices keep
scalar coordinates; merge/concat consume immediate raster-leaf children and
collapse the selected group; Dataset attr conflicts use native `combine_attrs`
with `no_conflicts` by default; publication accepts a native Item plus asset
Datasets; numerical conversion permits channels with different physical units.
Implementation status is recorded below; approval alone does not imply delivery. The subsequent review
approved positional splitting: each existing dimension position remains a
separate slice even when labels repeat. Without `names`, dimension labels supply
the node names and duplicate labels raise. Explicit `names` supplies one unique
name per position and permits repeated coordinate labels. The approved removal
operation is `parent.gs.ungroup(name)`, targeting a direct child group.
Publication checks available scalar times/periods and explicit membership;
position and copied timestamps cannot establish provenance.

The RGB discussion is resolved: preserve original Dataset attrs on its source
group when partitioning, and variable attrs on variables. Do not infer which
children need particular attrs or automatically relocate/filter RGB metadata.
Effective extraction inherits source metadata; consumers decide whether a
particular declaration applies to the extracted variables. Explicit local
overrides remain available.

The [spatial-contract feasibility study](raster-tree-feasibility.md) records
the tested alternatives, current transformation gaps and proposed publication
and loading contracts. Its shared-grid alternative is retained as study
history; the Option C refinement records the chosen responsibility split.
Trees are not mandatory STAC staging, and catalog loading does not automatically
restore a published tree hierarchy.

## Problem statement

Imagery and ancillary rasters need different treatment. Sentinel-2 reflectance
and SCL classes may have different grids, dtypes, packing and resampling rules.
Putting them in one band stack or file is not always appropriate. Repeated
caller-written variable splitting and save loops duplicate this knowledge.

The retired `RasterStac.from_dataset()` made several decisions at once:
it splits an Item dimension, assigns Item IDs, separates variables into assets,
and constructs an Item/asset hierarchy. Those publication decisions are too
early for general raster organization, layer composition and model preparation.

Shared metadata needs one authoritative update location. Persistently copying
Dataset attrs into every leaf makes later parent edits ineffective. Native
xarray inherits coordinates, but not Dataset attrs.

## Solution

Repurpose `GeoDataTree` as the accessor contract for a hierarchy of geospatial
raster Datasets. Keep native `xr.DataTree` as the container. Tree methods own
construction, raster-node restructuring and explicit spatial operations.
GeoDataTree no longer inherits singular spatial properties from GeoRaster.
DataArray and Dataset retain that single-raster interface.

STAC staging/publication separately associates named asset Datasets with Items.
These Datasets can be standalone or extracted from prepared tree leaves.
Publication uses raster writers and assigns Item identity, asset membership
and catalog metadata. STAC does not dictate the general tree hierarchy or
silently restructure rasters during publication.

```text
Dataset(s)
    -> named raster tree
    -> explicit split / dissolve / combine / spatial preparation
    -> prepared raster tree
         -> display layers
         -> model preparation -> tensors
         -> explicit STAC Item membership -> assets + Item table -> GeoParquet
```

These are different consumers of the same raster leaves. A leaf is not
intrinsically a file, STAC asset, model input or display layer.

### Decision and implementation status

| Concern | Agreed direction | Current implementation |
| --- | --- | --- |
| Container | Native DataTree with a GeoSave raster structure contract | Independent accessor validates raster leaves and rejects data-bearing parents |
| Dataset conversion | Named whole-Dataset node by default; explicit variable groups optionally partition it | `Dataset.gs.to_tree(name="ds", groups=None)` is implemented |
| Metadata ownership | Shared Dataset attrs on grouping nodes; local attrs override | `node.gs.to_dataset()` implements effective extraction |
| Restructuring | Subtree scope; merge/concat replace selected inputs in a new tree with exact alignment; optional dissolution | `group(groups)`, `split(dim, names=...)`, `merge`, `concat` and `ungroup` implemented |
| Spatial validity | Independent raster leaves; GeoDataTree separate from GeoRaster; alignment checked by operations | Implemented; shared-grid consumers use an explicit internal validation helper |
| Tensor conversion | Named independent leaf tensors; shared context checks only when requested; representation details open | Independent pixel tensors; shared context validates before reads; Dataset numerical stacking has separate dimension/dtype checks |
| STAC publication | Separate explicit Item/asset association consuming Datasets or extracted leaves | `stac.publish_item` writes explicit assets; RasterStac and Dataset.to_catalog removed |

## User stories

1. As a raster user, I want named whole-Dataset construction, so that wrapping data preserves its time dimension.
2. As a raster user, I want named variable groups, so that imagery and SCL can remain separate.
3. As a raster user, I want independent grids in one tree, so that construction does not resample sources.
4. As a raster user, I want independent timelines and static rasters, so that synchronization stays deliberate.
5. As a raster user, I want shared Dataset attrs on a parent, so that I update them once.
6. As a raster user, I want local attr overrides, so that specific descriptions coexist with shared defaults.
7. As a raster user, I want effective Dataset extraction without pixel reads, so that metadata inspection stays lazy.
8. As a raster user, I want splitting with or without its parent, so that I control the resulting hierarchy.
9. As a raster user, I want group dissolution, so that I can promote children without changing their pixels.
10. As a raster user, I want ordered variable combination, so that the resulting Dataset has deliberate band order.
11. As a raster user, I want explicit temporal concatenation, so that node names do not determine array axes.
12. As a raster user, I want restructuring to preserve metadata and source arrays, so that preparing another view does not alter the original.
13. As a model author, I want one named pixel tensor per leaf, so that sibling inputs remain independently prepared.
14. As a model author, I want operation-specific alignment checks, so that validity does not force premature model preparation.
15. As a map user, I want independently gridded display layers, so that visualization preserves source placement.
16. As a publisher, I want explicit Item membership, so that general tree structure does not imply STAC identity.
17. As a publisher, I want written paths associated with source nodes, so that catalog construction does not parse filenames.
18. As a publisher, I want raster metadata round trips, so that stored imagery and SCL retain their interpretation.
19. As a publisher, I want a native Item table, so that catalog persistence remains an explicit GeoParquet operation.
20. As a maintainer, I want small, readable operations with logical spacing, so that each behavior has one maintainable implementation.

## Implementation decisions

### Raster structure and ownership

The target structure has raster Datasets at leaves. Internal nodes organize
descendants and may carry attrs and non-spatial coordinates, but do not hold
raster pixels or supply an inherited grid to a raster leaf.
A root-only raster is valid because its root is also a leaf. A valid raster tree
contains at least one raster; empty organizational nodes do not represent rasters.

Each raster leaf follows the existing two-dimensional geospatial Dataset
contract, including supported optional time/band axes. Its spatial variables
share one georeferenced grid. This does not require all variables to have the
same units, packing or dtype: stacking and output formats impose their own
additional requirements.

Different leaves can have different CRSs, resolutions, alignments, extents and
timelines. They need not overlap. A subtree can scope one scene or site without
making that a universal requirement for the entire tree.

```text
/
├── site_a                       organizational node
│   ├── image                    Dataset[B02, B03, B04], time × y × x, 10 m
│   ├── scl                      Dataset[SCL], time × y × x, 20 m
│   └── elevation                Dataset[height], y × x, static
└── site_b
    └── image                    another grid and timeline
```

The leaf-only rule is enforced by GeoSave validation, not native xarray.
Raster-bearing parents now fail validation; a raster at the root is valid when
it has no children. Native file readers should not silently rewrite
stored hierarchies to make them conform.

### Named construction and grouping

The default conversion is a named node holding the entire Dataset. It is not
one node per variable and does not infer an Item dimension or publication ID.
Optional named groups partition variables underneath the source node.

```text
default                         with explicit groups
/                               /
└── s2                          └── s2                    shared Dataset attrs
    Dataset[B02,B03,B04,SCL]         ├── image             B02,B03,B04
    time remains intact             └── scl               SCL
```

`Dataset.gs.to_tree(name="ds", *, groups=None)` is implemented. `name` gives
the source node name and `groups` maps child names to ordered variable lists.

When groups are supplied, each data variable must appear in exactly one group.
Incomplete grouping raises an error identifying the unassigned variables;
repeated assignments raise an error identifying the variable and its groups.
There is no automatic remainder group, singleton-leaf fallback or implicit dropping of
unmentioned variables. Select a subset with native Dataset selection before
conversion when that is the intended input. This coverage rule also applies to
explicit variable groups supplied when splitting an existing raster leaf.

Grid coordinates remain on each raster leaf when constructing or splitting a
GeoSave tree. Parent ownership applies to shared Dataset attrs, not to a global
grid. Native trees supplied by callers must meet the same leaf-grid policy;
GeoSave transformations do not repair ancestor-owned grids.

Construction with groups must reuse variable-partitioning logic rather than
introduce a second splitting mechanism. This membership rule partitions one
source Dataset; it does not prohibit spatial overlap or matching variable names
in independently constructed raster nodes. Groups use explicit variable-name
lists. Preserve mapping insertion order for groups and list order for variables;
patterns, predicates and automatic sorting are not part of the initial API.

Native DataTree construction remains available for independently gridded
Datasets. Splitting an already aligned Dataset cannot recover original grids.

### Shared attrs and effective Dataset extraction

Shared Dataset attrs belong to the source/group node. Raster leaves keep local
Dataset attrs and their variables' attrs/encoding. Do not maintain synchronized
copies of shared attrs on every leaf.

`node.gs.to_dataset()` is the implemented public materialization operation:

- Call native conversion with all inherited coordinates, including scalar
  coordinates such as observation time and marked properties.
- Merge Dataset attrs from the root toward the selected node.
- Nearer ancestors override more distant ones; local attrs take precedence.
- Return the selected node's Dataset, without combining descendant rasters.
- Keep variable attrs, encoding and pixel arrays unchanged.
- Return a new attrs dictionary. Later parent edits affect subsequent calls,
  not previously extracted Datasets. Nested attr values remain shallowly shared.

Native `node.to_dataset()` is unchanged. Its default inherits indexed
coordinates; its all-coordinates mode includes non-indexed coordinates too.
Neither mode inherits Dataset attrs. Attrs carried by an inherited coordinate
remain attached to that coordinate.

The effective-attrs rule is a GeoSave extension, not a change to xarray. Consumers
that need inherited Dataset attrs should use the accessor operation. Calls that
intentionally inspect local node contents must remain local; replacing every
native conversion mechanically would change behavior.

Non-spatial coordinate inheritance continues to obey native alignment
constraints. Spatial coordinates belong to each raster leaf. The default
`raster_leaves` validation policy rejects inherited grid coordinates before
transformations; shared time/property coordinates remain supported.
Variable grouping retains all source coordinates on each leaf, including
auxiliary coordinates with additional dimensions such as CF time bounds.
Variable dimensions remain unchanged; grouping does not infer metadata applicability.

### Split, ungroup, merge and concatenate

The structural operations return new trees and do not compute, cast, decode,
reproject or normalize pixels as a side effect.

Bulk operations act on the raster leaves of the selected native subtree, for
example `tree["scene"].gs.crop(region)`. Do not add recursive selectors or
patterns to every method. The caller selects the node for merge or concat, and
the operation combines all of that node's children. There is no per-operation
input-path list. This follows the user's clarification of the method scope.

Merge and concat replace their selected source leaves with the result in the
returned tree, preserving unrelated nodes. They do not mutate the source tree
or retain duplicate inputs beside the result. The approved result collapses
the selected grouping node into a raster leaf, rather than requiring a separate
destination argument; see the scoped API contract below.

Selected inputs must have exactly matching spatial coordinates and indexes on
shared dimensions other than the concatenation dimension. Do not perform
automatic inner/outer joins or insert missing values. Duplicate variable names
are errors for merge; corresponding variable names are expected for concat.
Concat requires matching variable sets across all inputs; it does not union
their variables or fill missing observations.
Callers explicitly prepare differing grids or indexes before combination.

#### Scoped merge/concat API contract

Core call shape following the user's requested scope:

```python
merged = tree["scene"].gs.merge()
series = tree["observations"].gs.concat(dim="time")
```

Children means immediate children, all of which must be raster leaves. Reject
grouping children rather than recursively flattening different branches.

Return a new DataTree node containing the combined Dataset and no children. The
selected node supplies the output identity. The original tree is unchanged;
assign the returned node back explicitly when replacing it in the larger tree.
Extraction remains the separate `merged.gs.to_dataset()` operation.

These methods combine raster data only. They do not assign Item IDs, group STAC
assets, write files or assemble catalog rows. A mixed-grid subtree may be valid
for explicit STAC asset association while failing raster merge compatibility.

| Operation | Contract | Interface status |
| --- | --- | --- |
| Split variables | Partition a selected raster leaf into named raster leaves; each variable belongs to exactly one group | Explicit lists, exclusive coverage and caller ordering settled |
| Split a dimension | Split by position, removing the selected dimension while preserving its value as a scalar coordinate on each slice | Use dimension labels when `names` is omitted; duplicate labels raise; explicit names allow repeated labels |
| Ungroup | On a containing node, remove a named child group and promote its children, preserving their subtrees | `parent.gs.ungroup(name)` approved; returns a new containing subtree |
| Split without parent | Partition the leaf and promote the results into its containing node | Agreed option; reuse ungrouping |
| Merge variables | Replace selected leaves with one named Dataset leaf containing their distinct variables | Implemented; distinct variables preserve attrs, dtype and encoding |
| Concatenate | Replace selected leaves with corresponding variables assembled along an explicit dimension | Implemented; matching representations required; descriptive attr policy cannot discard packing |

```text
source                    split, retain parent       split, dissolve parent
/                         /                          /
└── s2 [all variables]    └── s2                     ├── image [B02,B03,B04]
                             ├── image              └── scl [SCL]
                             └── scl
```

Dissolving an existing group changes hierarchy, not raster values. Effective
coordinates and attrs from the removed node must be retained on promoted
children where they still apply, rather than moved onto a parent where unrelated
siblings would inherit them. Destination-name collisions raise an error. The
group identity is intentionally removed; this is not a promise that arbitrary
hierarchy annotations have an automatic destination.

The approved operation is `tree.gs.ungroup("s2")`, called on the containing node.
It targets the direct child named `s2` and returns a modified copy of the
containing subtree. It does not search descendants or return an ancestor beyond
the selected scope. For nested groups, select the containing node first.
`ungroup` replaces the earlier proposed spelling `dissolve`.

Do not silently drop variables, overwrite destination nodes, or repair
incompatible grids by alignment. Preserve source metadata, including RGB
declarations; structural partitioning does not infer applicability or rewrite
references. Consumers interpret those declarations in the selected Dataset.
Native xarray selection and merge/concat operations provide the mechanics;
GeoSave owns the raster-specific checks and structural consequences.

Static variables must not accidentally gain a time dimension through a split
and concatenation round trip. Variables without the split dimension stay once
in a separate leaf. The approved layout groups dimension slices together,
keeping static variables beside that group:

```text
scene
├── static                  height(y, x)
└── time                    grouping node, not a time coordinate
    ├── 2026-11-01          B04, first time slice
    └── 2026-11-11          B04, second time slice
```

Then `tree["scene/time"].gs.concat(dim="time")` combines all dated children.
After replacing that subgroup with the result, `tree["scene"].gs.merge()` can
combine the time series and static variables. This supersedes the earlier flat
layout of static and dated siblings. Observation coordinates belong to the
dated leaves, not their shared parent or the static leaf. The subgroup name is
structural: it is not a coordinate declaration or a STAC Item. `split("time", names=None)` implements this layout; explicit names follow position order.

Each dimension slice removes the selected dimension and retains its coordinate
as a scalar, for example `B04(y, x)` with `time=t1`. Associated observation
coordinates remain available. Concat reconstructs the dimension. This shape
uses each existing dimension position, not groups of equal coordinate values.
Repeated temporal labels therefore produce separate leaves. Loading or explicit
preparation owns pixel grouping/fusion; structural splitting does not repeat it.

Node names are local tree identifiers, separate from STAC Item IDs. When `names`
is omitted, use the selected dimension's coordinate labels as node names and
raise if labels repeat or their resulting node names collide. Do not group equal
labels or generate suffixes. Explicit `names` is an ordered sequence with one
unique child name per source position, allowing repeated coordinate labels.
A mapping keyed by coordinate value cannot distinguish those positions.
`group(groups)` partitions variables; `split(dim, names=...)` slices a required dimension. These are separate methods, with no mode dispatch.

Variable merging preserves independent variable identities: temperature and
elevation may have different units in one Dataset. Dimension concatenation
joins values of corresponding variables; units must be consistent for each
variable across its slices. Equal units alone do not establish compatibility.
Neither operation implicitly converts units or pixel representations. Use
separate `merge` and `concat` methods, rather than an overloaded `combine`.

For Dataset attrs from independent leaves, expose native `combine_attrs`, with
`"no_conflicts"` as the default. Preserve agreeing values and reject conflicts
unless the caller explicitly selects a different policy, such as
`combine_attrs="drop_conflicts"`. This differs from ancestor-to-child precedence.
Descriptive attr policy must not bypass units, packing, flags or grid checks;
concat checks dtype, units, flags, full CF packing and storage recipes, including unsigned interpretation.

### Existing method suite

Option C supersedes the earlier common-property interpretation: a general tree
does not expose one implicit CRS, GeoBox, transform or resolution. Consumers
requiring a common grid will use an explicit agreement check. The table below
records operation responsibilities; implemented interfaces follow these boundaries.

| Surface | Required direction |
| --- | --- |
| Validation | Separate valid raster structure from operation-specific relationships |
| Georeferencing | Validate each raster leaf independently; inspect spatial properties on extracted Datasets |
| Common GeoBox | Explicit operation/consumer requirement, not an implicit tree property |
| Region/time summaries | No implicit raster-wide extent or timeline; consumers own needed aggregation initially |
| Geographic crop | Apply the region to each selected raster in its own CRS; fail if a selected raster does not intersect; never drop leaves implicitly |
| Crop to grid | Slice/pad compatible pixel lattices; do not silently resample |
| Reproject | Require an explicit target; imagery and classes may need different resampling policies |
| Temporal sampling | Synchronized windows require matching timelines among dated leaves; independent timelines remain valid stored data |
| Chipping | Keep the current prepared-input common-GeoBox requirement unless separately redesigned |
| Tree storage | Preserve hierarchy, grids and timelines through Zarr/NetCDF |
| Raster file writing | Validate the chosen format's requirements rather than treating every valid leaf as COG-ready |

Independently georeferenced leaves do not need a common GeoBox. GeoDataTree no
longer inherits the single-raster interface. Consumers check shared grids only
when their operation requires one.

### Layer composition and tensor preparation

Tree grouping organizes datasets. It does not automatically define a tensor
channel axis, a time axis, a batch axis or a map rendering choice.

The current tensor interface returns a pair: a pixel-tensor dictionary keyed by
relative node paths and a separate context dictionary. Preserve this separation;
the earlier discussion's single flat sample dictionary is not the current
accessor return contract.

The agreed extension permits pixel-only conversion on independent grids,
one tensor per raster leaf. Tree context remains shared: requested grid metadata
requires one common GeoBox, and requested temporal context requires the relevant
timeline agreement. Reject disagreement when that context is requested.
Do not add a tree-level per-leaf context schema initially; callers can use
Dataset conversion on extracted leaves when they need individual contexts.
Chipping and model preparation can impose stricter requirements than conversion.
Model input naming belongs to preparation, without silently reducing full paths
to potentially colliding basenames.

Numerical conversion uses its own stacking checks, allowing channels with
different physical units. Dataset owns numerical stacking; tree conversion
delegates to each leaf. Compatible dimensions are required, and dtype conversion
must be explicit when needed. Decoding, normalization and categorical encoding
remain explicit preparation operations. Do not relax file-encoding or
geospatial-array safeguards as a side effect. Dataset `dtype=` is explicit; no tree-level per-leaf dtype/layout mapping has been added.

No global-grid check has been removed as part of the implemented extraction
method. Dataset owns numerical channel conversion separately from file-compatible arrays.

### Separate STAC staging and publication

The agreed separation is between general raster preparation and STAC
interpretation. STAC publication consumes explicit Item/asset membership from
Datasets or selected tree leaves, uses effective Dataset extraction where
applicable and common raster writers, and produces native Items/an Item table.

Each Item represents one observation or derived product with its own temporal
description. Its raster assets belong to that observation, such as imagery and
SCL from the same acquisition, even when their grids differ. A static/reference
raster used alongside imagery for analysis does not thereby belong to the
imagery observation and must not be included in that Item. Matching timestamps
alone also do not establish observation identity.

This is GeoSave's staging contract. STAC itself supports a representative
datetime or a temporal interval, including derived products and composites;
"one observation" does not require an instantaneous acquisition. See the
[STAC Item temporal definition](https://github.com/radiantearth/stac-spec/blob/v1.1.0/item-spec/item-spec.md#datetime).
A raster with `(y, x)` dimensions and scalar observation time is still dated
data. Missing a `time` dimension is not sufficient to classify it as static;
publication checks scalar time/period agreement and explicit membership.

Positional splitting is approved. ODC's default `groupby="time"` can combine several
source Items into one output plane; `groupby="id"` preserves separate planes
whose time labels may repeat. A source Item, a loaded plane and its time label
are therefore different concepts. The per-asset temporal
metadata requirement is not sufficient by itself: selecting a dated slice from
a Dataset can leave a scalar time coordinate on a variable that was static
before selection. Preserve known association when preparing data; do not claim
that time-coordinate presence proves observation membership. The
[observation-identity study](raster-tree-feasibility.md#observation-identity-and-duplicate-time-labels)
records the reproduced cases and the accepted positional direction. Across
independently prepared branches, publication membership remains explicit;
neither repeated timestamps nor matching positional order establish association.

The native Item already carries its publication ID. If an automatic Item-ID
formatter is added later, that belongs to STAC preparation, independently of
tree node naming. The retired wrapper used `id_format`; no formatter
parameter is required on general tree methods by the approved contracts.

The canonical input is an explicit mapping of asset keys to prepared Datasets
for each Item, with explicit Item metadata. Values can come from selected tree
leaves, different branches or standalone Datasets, provided they belong to that
Item's observation. Explicit membership selects assets; it does not waive this
requirement. This replaces subtree-path
selection as the underlying membership interface; no outside-asset override
system is needed. A tree convenience can be assessed later. A parent node or
date-looking path is not inherently an Item.

The approved publication interface accepts a native PySTAC Item carrying the
observation metadata alongside the asset mapping. One call validates and writes
one observation, then returns the completed Item. Reuse `StacItem.from_region`
for construction when appropriate. Multiple observations compose through normal
iteration and existing catalog construction; do not add another Item metadata
wrapper. `stac.publish_item(item, assets=assets, root=destination)` is approved and implemented;
`create_catalog(items)` and `catalog.gs.to_geoparquet(...)` complete the flow.

```text
source Datasets                       explicit publication membership
tree["observation_a/image"] ─────────> Item scene-a: image
tree["observation_a/scl"]   ─────────> Item scene-a: scl
standalone elevation Dataset          separate reference product

                                      result: COG assets + Item table
                                              -> explicit GeoParquet write
```

The shape above comes from the discussed preparation/publication workflow; it
does not require a fixed depth in the general raster tree. If an Item grouping
is dissolved, the caller must supply the new membership rather than rely on
implicit timestamp matching.

#### Publication from dimension-split trees

STAC staging must support the approved `scene/static` plus
`scene/time/<observation>` structure without requiring Item/asset tree depth:

```text
source tree                               explicit Item membership
scene
├── static [height]                       outside these observation Items
└── time
    ├── 2026-11-01 [B04] ──────────────> scene-a: image
    └── 2026-11-11 [B04] ──────────────> scene-b: image
```

The canonical asset values are effective Datasets extracted with
`node.gs.to_dataset()`. For example, one Item's explicitly chosen assets are:

```python
assets = {
    "image": tree["scene/time/2026-11-01"].gs.to_dataset(),
}
```

Item IDs and temporal metadata are explicit publication inputs. The `time`
group is never automatically an Item; a dated leaf may supply an asset directly
without another asset-child level. An observation may also be partitioned into
several leaves and supplied through the same asset mapping.

The static sibling remains available for analysis but is not an asset of these
dated observation Items. Staging must not repeat it, assign it the imagery's
time, or accept its inclusion merely because an explicit mapping requests it.
Publishing a reference product separately requires its own truthful temporal
metadata; the interface for that case remains open. No date is fabricated to
make timeless input publishable.

General tree grouping can retain both static and dated inputs for analysis.
The observation-membership requirement belongs to STAC staging, not GeoDataTree
validity. Shared parent attrs remain available during effective extraction but
are not automatically STAC Item properties.

Staging must not call tree merge/concat just to make assets belong to one Item.
Those methods combine raster data; publication associates independent files.

Ownership is explicit:

- Tree methods prepare raster nodes and their metadata.
- Raster I/O writes the requested representations and reports actual files.
- The STAC mechanism assigns Item IDs, asset membership, footprints, temporal
  descriptions and Collection association, then creates the Item table.
- Catalog persistence writes that table separately.

The agreed tree-writer result maps source node paths to written hrefs.
Publication similarly needs to retain the association between its explicit
asset membership and actual written hrefs; standalone Datasets do not require
invented node paths. Do not parse filenames or independently reconstruct the
writer's output. File layout and Item IDs need not be identical.

STAC storage metadata must describe the written representation, including
packing held in encoding. Shared Dataset attrs are not automatically STAC Item
properties; use the explicit metadata conventions and publication policy.
Initial publication requires explicit Item metadata and asset membership;
there is no automatic unanimous-value fallback or static-asset broadcasting.
The earlier static-asset sharing proposal is withdrawn. The explicit mapping and available scalar time/period validation are implemented;
reference products without observation dates remain a separate design.

Validate known format, membership, metadata and destination constraints before
pixel writes. This is preflight, not a transactional guarantee against failures
during I/O. Reuse the existing batch writer and native PySTAC/catalog mechanisms
where their contracts fit.

The general tree must not become another `RasterStac` hierarchy. The intended
migration removes `RasterStac` as a required public staging container once the
replacement workflow works. It does not remove publication functionality before
that replacement exists or add a new public publication-job object. Migrate
callers through explicit restructuring, including raster-bearing parents,
rather than silently reinterpreting old structures or retaining dual behavior.
Verify the replacement publication round trip before retiring the old route.

Catalog-to-tree loading initially composes existing Dataset/asset loading and
native tree construction. Do not introduce a dedicated catalog-to-tree loader
or named loading-plan interface yet. Dataset output remains the direct loading
path; caller recipes establish whether a tree convenience would remove useful
repetition.

The final approved public entry point is `stac.publish_item`.

### Implementation discipline

Implement one reviewed increment at a time and consult the user before deciding
open behavior. Keep readable native operations, explicit intermediate values
and blank lines between logical steps. Helpers must own a named operation and
remove duplication; avoid a monolithic transform/publish function or speculative
framework of node/asset/job objects.

## Testing decisions

Test through the public accessor and publication interfaces. Do not introduce
another public testing layer. Existing DataTree extraction/transform tests and
STAC raster publication round trips provide the prior patterns.

- Verify construction and restructuring through node paths, Dataset variables,
  grids, coordinates, attrs, encoding and source immutability.
- Observe Dask execution to establish laziness; object shapes alone are not
  sufficient. Keep pixel arrays shared where native structural operations allow.
- Verify root-to-leaf precedence, inherited scalar/indexed coordinates, local
  variable metadata, parent updates on subsequent extraction and unchanged
  native xarray conversion.
- Cover raster roots, metadata-only selected nodes, nested groups and differing
  grids/timelines. Do not claim the new leaf-only validator exists beforehand.
- Check split/dissolve composition and combine/concat round trips for supported
  contracts, including intentional handling of static variables.
- Verify model-input names, ordering, shapes, dtypes and context separately from
  geospatial file metadata. Batch matching tensor keys across samples.
- Publish imagery and SCL on independent grids; reopen files and load the Item
  table. Check pixels, nodata, dtype, packing, flags, band names, Item properties
  and Collection association. Persist and reopen GeoParquet for the final flow.
- Stage the approved dimension subgroup layout with a static sibling. Check
  that only explicit observation memberships create Items, direct raster leaves
  need no extra asset-child level, effective attrs/coordinates survive extraction,
  and the static sibling stays outside the dated Items without source changes.
- Reject mixing an independent static/reference raster into a dated observation
  before writes. Distinguish this from legitimate 2D observation slices carrying
  scalar time; finalize metadata validation rules before implementing the check.
- Test foreseeable publication failures before writes. Report runtime partial
  writes separately from preflight failures.

The implemented `node.gs.to_dataset()` has four focused tests covering metadata
and coordinates without pixel execution, parent updates and independent attrs
dictionaries, selected-node scope, and root-raster conversion.

Feasibility probes during the discussion demonstrated lazy partitioning,
promotion and recombination; explicit temporal split/concat; heterogeneous
per-leaf tensor collation; and local multiband imagery plus SCL publication.
They also exposed missing STAC band-name/classification output, dtype/encoding
compatibility gaps, and static-variable broadcasting. These probes are evidence,
not a complete regression suite or remote/training validation.

## Out of scope

- Changing native xarray inheritance or introducing replacement xarray objects.
- Implicit reprojection, resampling, decoding, normalization or mosaicking during
  structural operations or publication.
- Inferring observation identity solely from timestamps, filenames or tree depth.
- Expanding this redesign into arbitrary non-raster dimensions or a new ML graph.
- Automatic synchronization of previously extracted Datasets with parent edits.
- A new persistence framework, transaction system or public publication-job type.
- Compatibility layers for obsolete Alpha APIs unless explicitly requested.

## Further notes

### Approved core workflow decisions

Option C's responsibility split is settled. Do not reopen shared-grid validity,
exclusive variable grouping, ancestor attr precedence or merge-versus-concat
while resolving remaining method-level contracts. The user selected Option A
for all seven choices below. The alternatives remain as decision rationale,
not outstanding questions.

| Decision and root problem | Option A (approved) | Option B (not selected) | Consequence |
| --- | --- | --- | --- |
| Operation scope: nested trees need a predictable target | Select a native subtree; bulk operations visit its raster leaves | Add recursive selectors/patterns to every method | User clarified merge/concat consume all children of the chosen node; no input-path list |
| Restructuring output: merge/concat inputs may remain alongside their result | Replace selected inputs in the returned tree, leaving unrelated nodes intact | Add the result and retain every source leaf | A: structural inverse of splitting; source tree remains unchanged; B creates duplicate participation in later export/processing |
| Combination alignment: matching CRS/resolution can hide different extents and time indexes | Require exact spatial alignment and exact indexes on shared non-concatenated dimensions | Permit automatic outer/inner joins and missing-value insertion | A initially: callers explicitly prepare grids/indexes; duplicate variable names in merge fail, corresponding names in concat are expected |
| Shared tensor context: independent leaf pixels need not have one transform or timeline | Keep shared context and reject disagreement when that context is requested | Add a tree-level per-leaf context schema | A initially: existing Dataset conversion provides independent contexts without a new configuration model |
| Temporal sampling: independent timelines cannot share positional indexes | Keep synchronized-window sampling requiring matching dated-leaf timelines | Independently sample each leaf and define how their windows form output trees | A initially: B needs an explicit observation-matching contract beyond general tree validity |
| Publication selection: tree hierarchy alone does not establish observation membership | Canonical explicit asset-key-to-Dataset membership per Item, using extracted leaves as needed | Subtree membership plus overrides for assets elsewhere | A: handles standalone Datasets and different branches; all assets must belong to the Item's observation, excluding independent static/reference inputs |
| Catalog-to-tree loading: asset groups, output grids and timelines require choices | Compose existing Dataset/asset loading with native tree construction first | Add a catalog-to-tree loader with named loading groups immediately | A initially: validate actual caller recipes before introducing another public orchestration interface |

### Decisions still requiring consultation

Geographic crop membership is settled: raise an error naming non-intersecting
selected raster leaves, leave the source untouched, and do not remove leaves
implicitly. Callers select a relevant subtree before cropping.

Metadata inheritance versus interpretation is settled. SCL was a variable of
the source Dataset; its resulting leaf does not lose source attrs merely because
it cannot render that Dataset's RGB composite. Preserve source Dataset attrs on
the group and variable attrs on variables. Consumers interpret applicability,
and callers may override explicitly. Do not introduce automatic metadata-scope
classification or an RGB-specific partition exception.

Concat input variable membership is settled: matching variable sets are
required; callers explicitly prepare inputs. No automatic union or missing-value
insertion. The dimension subgroup with a static sibling is also approved,
including support from STAC staging through explicit membership. Scalar slices,
immediate-child combination with selected-group collapse, native attr policy,
native Item publication input and separate numerical stacking checks are also
approved. Positional splitting, dimension-label names by default, explicit
`names` for repeated labels, and `parent.gs.ungroup(name)` are approved too.
Concat rejects differing dtypes/storage recipes. The later grid clarification
supersedes inherited-grid materialization: every raster leaf owns its grid, and
transforms never update parent coordinates.

| Topic | Delivery / remaining work |
| --- | --- |
| Construction and restructuring | Implemented: complete exclusive grouping, positional dimension split, default label names, explicit names, static sibling layout, merge, concat, ungroup |
| Metadata | Effective extraction and parent ownership; crop-to-grid clips extents at their existing owner; representation attrs survive descriptive attr policies |
| Spatial operations | Leaf-owned grids enforced by `raster_leaves`; native mapping for transforms; explicit consumer alignment rules |
| Validation ownership | `raster/rules` holds existing DataArray, Dataset and DataTree rules; concat uses `compatible_concat(dim)` before native assembly |
| Numerical conversion | Different units allowed; matching dimensions and explicit dtype preparation required; tree-level per-leaf dtype/layout parameter shape remains unapproved |
| Publication | Implemented native `publish_item`, one observation, explicit asset map, preflight, actual written href association, projection/packing/names/class metadata |
| Observation metadata | Available scalar time/period checks and explicit membership implemented; copied timestamps cannot prove provenance |
| Tree batch writer | Standalone node-path-to-href writer is not implemented: entry point and destination policy remain unsettled; publisher asset-key mapping is separate |
| Catalog-to-tree | Dedicated API deferred by agreement; compose loading with native tree construction |
| Other publication | Reference products without dates, alternate formats, deferred/transactional writing and derived Collection extent updates still need separate design |

Ancestor attrs use nearer-node precedence; sibling Dataset attrs use native
`combine_attrs`. Neither decides observation identity or variable representation.

### Superseded proposals

- One variable per node as the default Dataset conversion: replaced by a named
  whole-Dataset node with opt-in groups.
- Time/Item splitting during initial conversion: replaced by explicit structural
  preparation and separate publication membership.
- Persistent shared attrs copied to every leaf: replaced by parent ownership
  and on-demand effective Dataset extraction.
- Rejecting all ancestor/leaf attr conflicts: replaced by nearer-node precedence.
- Always forbidding root raster data: a root that is a leaf remains valid.
- Solving general raster organization by expanding RasterStac into another
  raster-operation wrapper: replaced by the GeoDataTree contract.
- Requiring one GeoBox for every valid tree: explored but not adopted; Option C
  keeps independent leaves and removes the single-raster tree interface.
- Subtree paths as canonical STAC membership: replaced by explicit named asset
  Datasets per Item, with tree extraction as one source of those Datasets.
- Input-path lists on merge/concat: replaced by selecting a native node and
  combining all its children; the tree grouping itself defines membership.
- Flat static/dated sibling layout: replaced by a dimension-slice subgroup next
  to the static leaf so concat can consume every child of the selected group.
- Reusing static elevation as an asset of each dated imagery Item: rejected;
  analysis inputs can coexist in a tree, while staging groups assets of one
  observation or derived product only.

### Delivery ledger

Implemented: named wrapping; exclusive variable partitioning shared by constructor
and group; effective Dataset extraction; scoped merge/concat/ungroup; positional
split with default/explicit names; independent raster validation; native spatial
transforms on leaf-owned grids; Dataset numerical stacking; explicit Item/asset publication;
COG and catalog round trips, including imagery/SCL and encoded storage metadata.

Removed obsolete RasterStac and automatic Dataset catalog conversion after the
replacement round trips passed. No compatibility layer is retained. Loading keeps
its Dataset contract; no hierarchy or publication identity is inferred.

The table above lists the remaining interfaces and reasons. They are not being
dismissed for lack of callers. Historical foundation-only results below are kept
as evidence and superseded by the completion verification recorded afterward.

### Structural cleanup audit

Latest clarification: grid coordinates belong to each raster leaf. Grouping
nodes retain shared attrs and non-spatial coordinates. This supersedes the
previously approved support for materializing inherited grids during transforms.
The previous `_transform_leaves` helper was introduced during this redesign;
it was not retained RasterStac code. It is now removed entirely.

| Code / responsibility | Flag | Result |
| --- | --- | --- |
| Optional `groups`/`dim` split dispatcher | Removed | `group(groups)` partitions variables; `split(dim, *, names=None)` requires a dimension |
| `_split_dimension` forwarding layer | Removed | Dimension splitting has one public entry point |
| Generic `_transform_leaves` interface | Removed | Each spatial method uses native `map_over_datasets` directly |
| Ancestor grid scans, coordinate stripping and custom reconstruction | Removed | Leaf-owned grids are a validity requirement; no repair or normalization occurs |
| Generic effective-attrs comparison/override machinery | Removed | Operations preserve the existing metadata owners |
| Default raster structure/grid checks | Validation rules | `rules.datatree.raster_leaves` runs automatically through `.gs.validate()`; `same_grid` and `same_geobox` are additional operation rules |
| Inline concat compatibility checks | Moved to validation rules | `rules.datatree.compatible_concat(dim)` checks immediate children, grids, indexes, variables and representation; the accessor owns assembly and metadata preservation |
| Shared extent updates during crop-to-grid | Retained with native Dataset metadata operation | Every leaf targets the same GeoBox, so group extents are clipped with `Dataset.gs.clip_extent` at their existing owner; no coordinates change on groups |
| Positional name uniqueness/length checks | Retained | Prevent overwritten duplicate labels and preserve one name per position |
| Shared variable partitioning | Retained | Dataset construction and leaf grouping use the same complete-coverage rule |
| Merge, concat, ungroup and consumer alignment rules | Retained | Agreed raster operations, independent of STAC staging |

The old direct native mapping implementation skipped group Dataset metadata.
A local probe reproduced a correct cropped grid with a stale shared footprint.
The revised crop-to-grid maps the existing Dataset extent operation over grouping
metadata too. This preserves shared ownership without a generic transform helper
or copied parent attrs. Crop and reprojection leave grouping metadata unchanged.

Tests now exercise leaf-local grids, inherited non-spatial metadata, detached
subtrees, rejection of inherited spatial coordinates, and clipped extents at the
original metadata owner. Tests requiring automatic inherited-grid repair have
been replaced, rather than retained as compatibility requirements.

### Foundation verification, 2026-11-17

The focused raster, sampling, tensor-context, STAC and inference checks passed
235 tests, with two integration tests deselected by the repository defaults.
Additional migration checks passed 44 tests; two sampling regressions verify
unsorted/repeated positions and rejection of different timelines. Preservation
checks cover lazy arrays, variable encoding, CF bounds, inherited metadata,
group order, exact merge alignment and NetCDF/Zarr round trips.

The full default run completed with 985 passed, 16 failed, nine skipped and
31 deselected. Three obsolete tree-property/error-message assertions were then
migrated and retested. All other 13 failures reproduced against the saved
pre-increment source: mixed reflectance/SCL writing into one COG and existing
ML template sample/context contracts. The full suite is not green. Scoped type
checking retains two diagnostics that also reproduce on the baseline, in
Dataset metadata-target typing and the pipeline's optional store return.
Ruff and generated API freshness checks pass.

### Completion and simplification verification, 2026-11-17

The completed publication migration passed imagery/SCL COG, classification,
encoded storage and catalog round trips. Before the grid simplification, the
full default suite completed with 1013 passed, 12 failed, nine skipped and
31 deselected. The remaining failures are the previously reproduced ML template
sample/context failures; the original mixed imagery/SCL publication failure is
resolved by explicit separate assets.

After separating group/split, removing `_transform_leaves` and adopting the
default `raster_leaves` policy, the affected raster, sampling, publication,
catalog and inference suite passed 391 tests, with two integration tests
deselected. The standalone tree suite passed 56 tests, with those same two
deselections. These runs cover local grid ownership, lazy transforms, metadata
ownership and the changed rejection contract. Scoped DataTree type checking and
Ruff pass; generated API references are current. The full default suite was not
repeated after this final simplification, and its known failures remain open.
