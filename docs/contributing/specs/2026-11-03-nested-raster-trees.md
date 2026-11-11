# Raster trees: explicit Collections and nested raster groups

Updated 2026-11-04 after the flexible-inventory feasibility study.

```python
tree.gs.expand_rasters()  # Inspect every raster node, independent of publication scope.
tree.gs.to_catalog("output", collections=["observations/s2", "terrain/dem"])
tree["observations/s2"].gs.to_catalog("output", collection="s2")
```

## Representation and scope

Raster Datasets can appear at any depth, including the root and parents with
children. Each Dataset's raster variables share an affine grid: CRS, transform
and shape. Time remains a coordinate or dimension. Xarray's parent/child
alignment rules still apply; independent grids belong in sibling branches.

```text
/
├── observations
│   └── s2                         selected Collection root
│       ├── reflectance            red(time, y, x), nir(time, y, x)
│       │   └── qa                 cloud(time, y, x)
│       └── classification         scl(time, y, x)
└── terrain
    └── dem                        selected root with elevation(y, x) data
```

`collections=["observations/s2", "terrain/dem"]` selects relative node paths.
Their final path components become Collection IDs (`s2` and `dem`). The selected
node's own data and all descendants belong to that Collection. Attributes supply
metadata; they do not establish boundaries. Short names are not searched
recursively, and hierarchy depth has no special meaning.

`collection="s2"` assigns the whole input tree to one Collection ID, including
root data. Supply either `collections` or `collection`, never both. Tree COG and
catalog publication require an explicit selection. Dataset publication keeps its
existing optional Collection argument and carried STAC metadata behavior.

Unknown paths, duplicate Collection IDs, overlapping roots and unassigned raster
nodes fail before writes. To publish only a subset, first select that subtree.
This prevents incomplete selections from silently discarding data.

`gs.datasets` enumerates native raster Datasets by relative path (`.` for the root).
`gs.expand_rasters()` returns individual variable observations and metadata without assigning
Collection membership. There is no separate `inspect_grids` method.

## Inventory

Expansion returns a fresh pandas DataFrame with one row per selected raster:

| Column | Meaning |
|---|---|
| Expanded dimension names | Selected coordinate labels, or integer positions when unlabeled |
| `raster` | Native lazy DataArray retaining coordinates and spatial metadata |
| `time`, `start_datetime`, `end_datetime` | Applicable scalar temporal metadata only |
| `variable` | Dataset member name; added by Dataset and DataTree |
| `path` | Relative node path; added only by DataTree |

`dim=None` expands time when present; `dim=()` keeps complete variables.
Static variables are not broadcast across another variable's observations.
Metadata-only nodes produce no rows. Empty results retain applicable columns.
Reserved output names cannot also name expanded dimensions at that container
level. Spatial metadata is read through each raster's accessor and native grid.
Inspection does not require georeferencing; callers request structural checks
separately through `validate`. No raster pixels are computed.

The DataFrame is an inspection result, not mutable configuration. Publication
uses native Datasets by their paths and resolves Collection membership separately.

## Publication

One planner in `io/paths.py` validates selected raster nodes, applies Collection
attrs as defaults beneath node attrs, selects time/bands and checks all output
paths together. Intermediate organizational attrs are not implicitly merged into
raster metadata. COG writing and STAC publication use the same plan; STAC retains
Collection membership separately from the selected Dataset's attributes.

| Field | Value |
|---|---|
| `{collection}` | Collection identifier |
| `{group}` | Path within the selected Collection root; `.` for its own data |
| `{grid}` | Short node label |
| `{time}` | Selected datetime, supporting strftime formatting |
| `{band}` | Selected variable name |

`{res}` is replaced by `{grid}` without an alias. Repeated short labels require
`{group}` or another distinguishing field. All collisions, invalid structure,
unsafe relative paths and predictable Item conversion errors fail before writes.
Templates control file selection, not Item identity.

The default remains one COG per variable and date. Publication groups these
assets into one Item per Collection/date, carrying each asset's spatial metadata.
Duplicate asset keys are rejected even when file paths are unique. Temporal
identity policy is unchanged by this refactor.

`to_catalog()` returns an in-memory Item table; `to_geoparquet()` saves it.
Reopening STAC groups assets by exact grid and therefore does not recover the
original organizational paths. Native Zarr/NetCDF preserve the tree structure.
Group enumeration order is not persistent identity.

## Reading and remaining boundaries

STAC readers build Collection/grid trees with time axes. Actual GeoBoxes determine
asset grouping; names such as `r10m` are readable labels, with deterministic local
suffixes when labels collide. Files retain their own georeferencing; conflicting
asset metadata produces a warning rather than overwriting it.

`to_dataset()` merges compatible nodes sharing one exact grid, including
nested organizational groups. It takes no grid-selection or resampling options;
call `tree.gs.align(how=reference, resampling=...)` explicitly before
converting mixed grids. This work does not redefine all tree
transforms, observation identity, or per-variable resampling keys for arbitrary
hierarchies. GeoZarr convention encoding/decoding and STAC mappings for Zarr stores
remain out of scope.

## Validation

Workflow tests cover nested inventory and path expansion without pixel reads,
parent and root data, repeated node labels, explicit Collection paths, unsupported data
rejection before writing, inherited time coordinates, COG/GeoParquet reconstruction,
and native Zarr hierarchy preservation.
