# cf-xarray capability review for GeoSave

Status: research and adoption checklist, not an implementation specification.

```python
import cf_xarray

ds.cf.axes
ds.cf.coordinates
ds.cf.get_associated_variable_names("red")
ds.cf[["red"]]                 # Dataset can retain supporting variables
quality.cf == "cloud"          # Boolean DataArray
quality.cf.is_flag_variable    # Recognition, not complete validation
```

## Evidence and version boundary

This review covers every user-guide chapter linked from the [documentation index](https://cf-xarray.readthedocs.io/en/latest/index.html), the introduction, FAQ, roadmap, API inventory and release history. The per-page manifest below distinguishes reading from execution. It does not claim that all generated API pages, every linked external example, or every possible CF representation were executed.

GeoSave's installed version was verified as **0.11.3**. Rendered `latest` pages were consulted, and complete official documentation source was read at upstream revision [`ee13cb860717299ca8f66d35461e5e50910f3d47`](https://github.com/xarray-contrib/cf-xarray/tree/ee13cb860717299ca8f66d35461e5e50910f3d47/doc). Rendered search caches differ in age; upstream main can differ from the installed release. The installed source is the authority for local behavior. Release 0.11.3 added UGRID support-variable propagation; 0.11.2 added topology propagation and grid-mapping performance changes. [Releases](https://github.com/xarray-contrib/cf-xarray/releases)

The package explicitly does **not** implement the complete CF data model. Its purpose includes shared CF interpretation helpers so downstream packages can stop duplicating parsing. Unit-aware arithmetic and reprojection remain other packages' responsibilities. “Fully adopted” therefore means that GeoSave delegates applicable interpretation and preserves the relationships it needs—not that GeoSave invokes every available `.cf` method. [Roadmap](https://cf-xarray.readthedocs.io/en/latest/roadmap.html)

## Data model: values, declarations and relationships

```text
Dataset
├── coordinates
│   ├── east(col) / north(row)      values
│   │   └── attrs                  axis, standard_name, units
│   ├── longitude(row, col)         auxiliary geographic coordinates
│   └── time(time)
│       └── attrs/encoding         bounds reference, as applicable
├── data variables
│   ├── image(time, row, col)
│   │   └── attrs/encoding         references to CRS, measures, ancillary data
│   ├── time_bounds(time, nv)       supporting data; not another image band
│   ├── cell_area(row, col)         supporting data
│   └── crs()                      CRS definition in its attrs
└── attrs                          dataset-level description/provenance
```

Coordinate values are variables; `axis`, `standard_name`, `units` and related declarations describe those variables through attrs. These are complementary parts of the model. Axes (`X/Y/Z/T`) and coordinate kinds (`longitude/latitude/vertical/time`) have distinct discovery criteria. Discovery properties return lists because multiple matches are legitimate. The guide's “Axes or Coordinate?” subsection is still a TODO, so it cannot settle GeoSave's grid-selection policy. [Axes and coordinates](https://cf-xarray.readthedocs.io/en/latest/coord_axes.html)

Dataset selection can retain associated variables even if they originally were data variables. A DataArray can attach only dimension-compatible coordinates; bounds with an additional dimension and multi-variable geometries need Dataset representation. A scalar semantic selector errors on ambiguous matches; a list selector can return all matches. Consequently `ds.cf["red"]` is not merely a renamed `ds["red"]`. [Selection](https://cf-xarray.readthedocs.io/en/latest/selecting.html)

**GeoSave implication:** attrs groups remain the owner of coherent declarations. Adoption must also audit variable placement, references and extraction; replacing attrs lookups alone is insufficient. Each operation must distinguish raster bands from bounds, measures, ancillary arrays and CRS/topology carriers.

## Capability inventory

### Discovery and semantic operations

`.cf.axes`, `.coordinates`, `.standard_names`, `.cell_measures`, `.cf_roles`, `.bounds` and related properties expose discovered names. `.cf.keys()` and membership queries support inspection. Existing variable names may be mixed with semantic names. This provides two legitimate integration styles: resolve a name then call native xarray, or use a wrapped `.cf` operation directly. GeoSave need not introduce a second semantic-name resolver. [How to use](https://cf-xarray.readthedocs.io/en/latest/howtouse.html), [quickstart](https://cf-xarray.readthedocs.io/en/latest/quickstart.html)

The introduction demonstrates semantic slicing, reductions, rolling, resampling, datetime-component groupby and cell-measure weighting. `.cf.sizes` and `.cf.chunks` expose semantic keys; ambiguous matches can error. Grouped objects remain wrappers supporting semantic arguments; ordinary operation results are native xarray objects. Discovery of an auxiliary coordinate does not by itself guarantee a dimension-taking operation can use that coordinate's semantic name: verify the intended shape. [Introduction](https://cf-xarray.readthedocs.io/en/latest/examples/introduction.html)

`differentiate(..., positive_upward=True)` optionally adjusts derivative sign using the coordinate's `positive` attribute. It is a focused physical interpretation beyond argument translation. It does not replace generic pixel transformations. [Differentiate API](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.differentiate.html)

### Flags: recognition, evaluation and grouping

```python
quality.cf.is_flag_variable
quality.cf == "cloud"
quality.cf.isin(["cloud", "shadow"])
quality.cf.flags  # Dataset of named boolean masks in installed 0.11.3

from cf_xarray.groupers import FlagGrouper

ds.groupby(quality=FlagGrouper()).mean()
```

CF flags include exclusive enumerated values, independent bit masks, and combined masks/values. Named comparisons and membership produce boolean arrays; `FlagGrouper` supplies categorical labels from `flag_meanings` for declared values. The guide explicitly cautions that mask and combined-mask interpretation have limited testing. [Flags guide](https://cf-xarray.readthedocs.io/en/latest/flags.html)

In installed 0.11.3, `is_flag_variable` only recognizes the presence of `flag_meanings` plus either `flag_values` or `flag_masks`. It does not validate lengths, dtype, emptiness, legal meanings, or all CF constraints. Actual evaluation has additional checks. `FlagGrouper` specifically requires values and meanings; it is not a generic bitfield-grouping replacement. Its source uses native `UniqueGrouper`, and `reset()` is unimplemented. [Installed-release accessor source](https://github.com/xarray-contrib/cf-xarray/blob/v0.11.3/cf_xarray/accessor.py), [grouper source](https://github.com/xarray-contrib/cf-xarray/blob/v0.11.3/cf_xarray/groupers.py)

**GeoSave implication:** keep `Legend` validation/writing and palette semantics; evaluate complete CF flags through `.cf` where appropriate. Do not replace a broader categorical-protection guard with this recognition property without deciding behavior for palette-only or incomplete metadata. The `.cf.flags` property and `FlagGrouper` are additional opportunities beyond the initially investigated comparisons.

### Bounds and vertices

`.cf.get_bounds(key)` resolves existing bounds; `.bounds` inventories references. `.add_bounds()` returns a Dataset with estimated bounds, using coordinate interpolation and edge extrapolation. The estimates can be inaccurate for curvilinear grids; producer-provided bounds are preferable. None of these estimates recover acquisition coverage. [Bounds guide](https://cf-xarray.readthedocs.io/en/latest/bounds.html)

`bounds_to_vertices` converts CF cell-bound arrays into vertex arrays; `vertices_to_bounds` reverses supported arrangements. The top-level functions return DataArrays; the Dataset accessor conversion returns a Dataset. Check ordering, supported dimensionality and laziness before using them for grid construction. Bounds discovery is a direct parsing candidate; observed-period aggregation remains GeoSave policy. [Bounds-to-vertices API](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.bounds_to_vertices.html), [vertices-to-bounds API](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.vertices_to_bounds.html)

### Grid mappings and associated-variable selection

`.cf.grid_mapping_names` maps projection kinds to variables. `.cf.grid_mappings` returns a tuple of `GridMapping` objects, including the CRS as a `pyproj.CRS`, the defining DataArray and coordinate names. Multiple mappings are supported; a singular mapping lookup can fail when several exist. Selection propagates referenced mapping/coordinate variables. These are interpretation facilities, not reprojection or affine-grid construction. [Grid mappings](https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html), [GridMapping API](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.GridMapping.html)

`get_associated_variable_names(name)` exposes references grouped by relation: ancillary variables, bounds, cell measures, coordinates, grid mappings, grid and mesh. This is important for band extraction, selection and persistence. It should not be mistaken for a universally validated list of “all non-raster variables,” nor a policy for removing unreferenced stale CRS coordinates. [Accessor source](https://github.com/xarray-contrib/cf-xarray/blob/v0.11.3/cf_xarray/accessor.py)

### Metadata enrichment and renaming

`guess_coord_axis()` uses heuristics and returns an object with inferred declarations. It is an explicit repair/enrichment step, not a validator or a substitute for known CRS/grid information. Attribute preservation is necessary for discovery, but retaining attrs after transformations can also preserve stale meaning. [Quickstart](https://cf-xarray.readthedocs.io/en/latest/quickstart.html)

`add_canonical_attributes()` uses the CF standard-name table, can preserve or override existing attributes, and adds history. Without a supplied source it may download the table through optional `pooch`. Assigning units metadata does not convert pixel values. This must not be an invisible ingestion cleanup. [Canonical attributes API](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.add_canonical_attributes.html)

`rename_like()` matches metadata between objects and returns a renamed DataArray/Dataset. It can update relationships, but uniqueness and skipped categories matter. Its prose describes restricted matching while parameters cover broader categories; use a representative source-level probe before replacing any GeoSave rename method. It is not a general schema migration engine. [Rename API](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.rename_like.html)

### Units and vertical coordinates

Importing `cf_xarray.units` configures Pint parsing/formatting for CF-style unit strings and additional aliases. Formatting can produce strings such as `m3 s-2`. Unit-aware computation remains Pint/pint-xarray territory; cf-xarray does not reject every Pint unit unsupported by UDUNITS. This is not a replacement for `Quantity` declaration ownership or pixel scaling. [Units](https://cf-xarray.readthedocs.io/en/latest/units.html)

`.cf.formula_terms` interprets parameter references; `.decode_vertical_coords()` computes physical vertical coordinates from supported parameterizations. It **mutates the Dataset and returns `None`**. The installed docstring lists three ocean formula families and warns of limited testing; actual support should be checked against source for any selected formula. No GeoSave-wide vertical decoding should be inferred from metadata discovery alone. [Vertical guide](https://cf-xarray.readthedocs.io/en/latest/parametricz.html), [decoder API](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.decode_vertical_coords.html)

### Specialized CF serialization

`encode_multi_index_as_compress()` and `decode_compress_to_multi_index()` return Datasets converting MultiIndex dimensions to/from CF compression-by-gathering representation. The encoded result can be serialized by xarray. Sparse-array conversion is explicitly unsupported. These functions do not implement scale/offset packing, and do not replace native xarray codecs. [Encoding guide](https://cf-xarray.readthedocs.io/en/latest/coding.html)

Geometry functions convert Shapely collections to a CF Dataset and back to a DataArray; Dataset-level geometry encoders/decoders preserve collections suitable for xarray storage. Geometry containers require associated arrays across dimensions, so selection uses Dataset form. Limitations include mixed geometry families in one array, unsupported Z, naming conventions and manual linkage when the automatic heuristic does not apply. Polygon-hole association depends on ring ordering. This is relevant only if GeoSave chooses CF vector-cube persistence, not a replacement for GeoDataFrame/GeoParquet or raster polygonization. [Geometry guide](https://cf-xarray.readthedocs.io/en/latest/geometry.html)

### Discrete sampling, SGRID and UGRID

Discrete-sampling support discovers `cf_role` variables such as profile identifiers. The chapter does not establish a complete ragged-array/trajectory reader or structural validator. [Discrete sampling](https://cf-xarray.readthedocs.io/en/latest/dsg.html)

SGRID support parses topology metadata for axis/dimension discovery. UGRID recognizes mesh topology. Selection follows topology/connectivity/coordinate references, retaining compatible coordinates on DataArrays and the broader collection on Datasets. These features do not establish that ODC can reproject every discovered staggered or unstructured grid. [SGRID/UGRID](https://cf-xarray.readthedocs.io/en/latest/sgrid_ugrid.html)

### Plotting, custom criteria and provenance

DataArray plotting accepts semantic axis/hue/facet keys and can interpret vertical direction to choose orientation. The guide says Dataset plotting is unsupported, but a 0.11.3 smoke probe successfully returned a PathCollection from Dataset `.cf.plot.scatter(x="air_temperature", y="b")`. Other Dataset plots remain untested. GeoSave's map rendering, palette handling and display sampling remain separate concerns. [Plotting](https://cf-xarray.readthedocs.io/en/latest/plotting.html)

Custom criteria map semantic aliases to attribute/name regular expressions. `set_options` supports temporary contexts and global changes; the docs prefer contexts to limit side effects. A GeoSave vocabulary for sensor bands is possible, but it requires an explicit ambiguity/selection contract rather than global import-time mutation. [Custom criteria](https://cf-xarray.readthedocs.io/en/latest/custom-criteria.html)

Automatic `cell_methods` and general history tracking are described as future work blocked on upstream changes. Do not delete composite cell-method/provenance policy on the assumption `.cf` already owns it. Canonical-attribute history is a separate implemented behavior. [Provenance](https://cf-xarray.readthedocs.io/en/latest/provenance.html)

## Documentation limitations found during review

- The API overview says flag masks are unsupported, while the flags guide, release history and installed implementation support them. Treat that overview sentence as stale. [API overview](https://cf-xarray.readthedocs.io/en/latest/api.html), [release history](https://cf-xarray.readthedocs.io/en/latest/whats-new.html)
- What's New redirects readers to GitHub releases for recent updates; its embedded changelog is historical. Some old tutorial bug notes likewise require reproduction rather than acceptance as current limitations.
- The guide's axes-versus-coordinates explanation is unfinished. GeoSave must explicitly distinguish an identified coordinate, a dimension, an affine raster grid and a supported operation.
- Internal helpers are listed for contributors, not thereby endorsed as public dependencies. Prefer public accessor methods; do not replace GeoSave glue with imports of `_get_*` internals. [Contributor reference](https://cf-xarray.readthedocs.io/en/latest/contributing.html)

## Adoption review checklist

For each boundary—STAC → ODC → stamping → Dataset → band extraction → stacking → transforms → storage/publication—record the following before authorizing changes:

1. Actual dimensions, coordinate variables, support variables, attrs and encoding before/after.
2. Which `.cf` names and relationships are discoverable; which are ambiguous or absent.
3. Whether extraction preserves required support variables or drops them because of the DataArray model.
4. Which attrs group owns declaration validation/writing; which `.cf` API owns interpretation.
5. Which operation needs an affine grid, one band, a datetime index or a particular numeric representation beyond what `.cf` recognizes.
6. Return type, in-place mutation, network access and pixel-task behavior of the proposed dependency call.
7. What code can be deleted after equivalent behavior is demonstrated, versus new capabilities outside current scope.
8. Round-trip and malformed-metadata probes appropriate to that specific boundary.

This checklist deliberately retains attrs groups. They prevent scattered interpretation and validation; `.cf` should remove duplicated CF parsing/evaluation without dispersing GeoSave's own metadata contracts.

## Coverage manifest

All entries below were read, not merely discovered in navigation. “Source” means full official page source at the revision above. Most rendered pages were also opened. Root-level GeoSave probes are recorded separately below/alongside this note; documentation reading is not numerical validation.

| Page | Coverage |
|---|---|
| [Index](https://cf-xarray.readthedocs.io/en/latest/index.html) | Rendered index and full source; navigation inventoried |
| [Quickstart](https://cf-xarray.readthedocs.io/en/latest/quickstart.html) | Full source |
| [How to use](https://cf-xarray.readthedocs.io/en/latest/howtouse.html) | Full source |
| [FAQ](https://cf-xarray.readthedocs.io/en/latest/faq.html) | Full source; primarily representation/display guidance |
| [Axes and Coordinates](https://cf-xarray.readthedocs.io/en/latest/coord_axes.html) | Full source plus installed criteria via integration review |
| [Selection](https://cf-xarray.readthedocs.io/en/latest/selecting.html) | Full source |
| [Flags](https://cf-xarray.readthedocs.io/en/latest/flags.html) | Full source and installed recognition/grouper implementation |
| [Units](https://cf-xarray.readthedocs.io/en/latest/units.html) | Full source; no unit-conversion numerical validation |
| [Vertical coordinates](https://cf-xarray.readthedocs.io/en/latest/parametricz.html) | Full source and installed API docstring; no formula numerical validation |
| [Bounds](https://cf-xarray.readthedocs.io/en/latest/bounds.html) | Full source |
| [Grid mappings](https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html) | Full source |
| [Encoding/decoding](https://cf-xarray.readthedocs.io/en/latest/coding.html) | Full source |
| [Discrete Sampling Geometries](https://cf-xarray.readthedocs.io/en/latest/dsg.html) | Full source |
| [SGRID/UGRID](https://cf-xarray.readthedocs.io/en/latest/sgrid_ugrid.html) | Full source, checked against current release notes |
| [Geometries](https://cf-xarray.readthedocs.io/en/latest/geometry.html) | Full source; no geometry numerical round trip in this research pass |
| [Plotting](https://cf-xarray.readthedocs.io/en/latest/plotting.html) | Full source; no rendered plot comparison |
| [Custom criteria](https://cf-xarray.readthedocs.io/en/latest/custom-criteria.html) | Full source |
| [History/provenance](https://cf-xarray.readthedocs.io/en/latest/provenance.html) | Full source; identified as planned functionality |
| [Introduction](https://cf-xarray.readthedocs.io/en/latest/examples/introduction.html) | All markdown and code cells read from official source; saved outputs not rerun |
| [API reference](https://cf-xarray.readthedocs.io/en/latest/api.html) | Initial inventory; superseded by the [complete indexed API audit](2026-11-10-cf-xarray-api-audit.md), including all 60 generated pages |
| Generated APIs | Opened rename_like, add_canonical_attributes, differentiate, GridMapping, decode_vertical_coords, bounds_to_vertices, vertices_to_bounds; installed docstrings/source used for associated-variable discovery and flag recognition |
| [Contributing](https://cf-xarray.readthedocs.io/en/latest/contributing.html) | Full source; internal/public distinction reviewed |
| [Roadmap](https://cf-xarray.readthedocs.io/en/latest/roadmap.html) | Full source |
| [What's New](https://cf-xarray.readthedocs.io/en/latest/whats-new.html) | Full source; release notes v0.10.11–v0.11.3 followed |
| External EarthCube, CESM, COSIMA examples | Linked by index; not part of this first-party chapter coverage, not audited |

No production code was changed by this documentation research. Specialized features above remain documented possibilities unless a local experiment or implementation audit explicitly establishes their fit.

## GeoSave implementation audit (live source, 2026-11-10)

This section is an audit, not a recommendation to replace every native operation
with `.cf`. Existing attrs groups remain the central declaration/validation
interface. A fully evaluated integration means each relevant CF capability has
an explicit owner and a checked contract; unused ocean/mesh features need not
be added to GeoSave just to claim adoption.

### Placement contract

```text
Dataset.attrs                 dataset descriptions / retained STAC provenance
Dataset.coords[name].values   actual coordinate values
Dataset.coords[name].attrs    that coordinate's axis, standard_name, units
Dataset.data_vars[name]       measurement or CF support variable
Variable.attrs                semantics / active packed representation metadata
Variable.encoding             storage settings / decoded relationship metadata
```

Coordinates and their attrs are complementary. GeoSave's `_write_cf_axes`
already writes the declarations on coordinate variables, not the Dataset root.
The audit must not infer that all current metadata belongs in coordinates.
Per-band spectral values are presently variable attrs; stacking into one
`band` dimension is a separate representation boundary where differing values
may need per-band coordinates instead of conflict-dropping combined attrs.

### Function-level adoption matrix

Paths below are relative to `src/geosave/geodata/`.

| Current code | CF facility | Adoption / gap | Required check before changing |
| --- | --- | --- | --- |
| `core/raster/utils/grid.py:read_spatial_dims` | `.cf.axes`, `.cf.coordinates` | Adopted for CF fallback; ODC conventional names still win | Contradictory names/attrs; multiple grids; auxiliary axes; curvilinear coordinates |
| `core/raster/utils/grid.py:_write_cf_axes` | CF declaration criteria | Writes known coordinates correctly; guessing is not an equivalent | All CRS types, axis units and retained coordinate attrs |
| `conventions.py:from_cf_bounds` | `.cf.get_bounds`, associated-variable lookup | Still manual; previous implementation report was incorrect | attrs and encoding references; absent/malformed bounds; no implicit estimated bounds |
| `conventions.py:to_cf_bounds` | Native Dataset coordinate construction | GeoSave start/end translation, not replaced by `add_bounds` | DataArray extraction and file time/bounds units |
| `core/raster/base.py:write_time_bounds/time_bounds/select_time/times` | CF time/bounds lookup for discovery | Canonical time and GeoSave periods remain assumed | Multiple time coordinates; scalar time; inclusive observed interval semantics |
| `core/raster/base.py:grid_mapping_coords` | `.cf.grid_mappings` | Partial fit; current method also finds stale unreferenced carriers | Active mapping selection must not replace cleanup discovery |
| `core/raster/dataset.py:spatial_dims/geobox/validate` | Associated-variable relationships | All data variables currently treated as rasters | Bounds, cell areas, ancillary rasters, unrelated variables, multi-grid datasets |
| `core/raster/dataset.py:mask/reproject/composite/_map_variables` | Relationships inform variable ownership | Operations still traverse variables without a CF support-variable contract | Cell areas after warp; bounds after temporal reduction; ancillary data preservation |
| `core/raster/dataset.py:to_dataarray` | Coordinate representation; semantic selection | Combines variable attrs and can drop differing spectral metadata | Dataset → band array → Dataset round trip, per-band declarations |
| `core/raster/datatree.py` | No installed `.cf` tree accessor | Concrete node traversal stays GeoSave-owned | Inherited bounds/mappings and node-local variable ownership |
| `io/odc.py:load_dataset/stamp_band_attrs` | CF discovery of produced raster | Spatial declarations and newly annotated time discoverable; attrs groups remain | ODC grouping/fusion, alias resolution, effective nodata, class/bitflag metadata |
| `stac/raster.py:stamp_item_time` | CF time-coordinate criteria | New Item date annotated; existing unannotated dates left unchanged | Existing file time, scalar/vector representation and composite coverage |
| `io/rasterio.py:open_rasterio` | CF time-coordinate criteria | ACDD/TIFF/filename-created dates remain unannotated | All three time sources and native decode options |
| NetCDF/Zarr readers | Native decode_coords plus `.cf` relationships | Bounds normalized to GeoSave periods; not uniform CF model | Coordinate/data-variable classification and encoding references |
| `attrs/groups/legend.py` | `.cf.is_flag_variable`, `.cf.flags`, comparisons | Keep declaration validation and palette; CF can evaluate conditions | Checker is shallow; combined masks/values; unknown names; serialized attrs |
| `core/raster/dataarray.py:is_categorical` | `.cf.is_flag_variable` | Not an equivalent replacement | Palettes and incomplete legends must remain protected from interpolation |
| `features/cloud_mask.py:scl_valid_mask` | `.cf.isin` | Optional named-class API, not direct numeric-call replacement | Existing numeric input works without flag metadata; do not silently require it |
| `core/raster/utils/statistics.py:class_percent` | `.cf.flags` | Potential condition-generation reuse only | Overlapping bitfields are not mutually exclusive classes; fill exclusion and denominator policy |
| `core/raster/dataarray.py:colorize` / STAC classification output | Legend and palette ownership | CF boolean flag datasets do not replace colors or STAC classes | Combined flags must not be mistaken for enumeration codes |
| Packing conversion / NetCDF encoding / Zarr option filtering | Native xarray codecs and backend contracts | Not cf-xarray replacements | Preserve data representation and backend-specific options |
| Spectral indices / `dataset.rgb` | Custom criteria possible | Existing explicit variable selection/GDAL color roles are different contracts | No automatic STAC common_name → CF standard_name conversion |
| Plotting/time visualization | `.cf.plot` / semantic lookup | Evaluate only where CF selection eliminates code | GeoSave HoloViz grid/time UI is not replaced by xarray plotting wrappers |

### Reproduced local findings

Executable evidence: [cf_xarray_smoke.py](cf_xarray_smoke.py). Run from the
repository root:

```bash
MPLCONFIGDIR=/tmp/geosave-mpl .venv/bin/python docs/contributing/reviews/cf_xarray_smoke.py
```

The script reproduces selected current behavior, including known gaps. Its
assertions are audit observations, not contracts future fixes should preserve.


Installed cf-xarray **0.11.3**, xarray **2026.4.0**. These are small smoke
experiments, not a complete conformance suite. No production files were changed
for this documentation audit.

1. **Flag checker is not full validation.** `is_flag_variable` checks presence
   of `flag_meanings` and either `flag_values` or `flag_masks`. Wrong lengths,
   empty declarations and float pixels can return True. `Legend` rejects the
   tested wrong lengths; `.cf.flags` also rejects them when evaluation is built.
2. **Three flag representations work lazily.** Enumerations, independent bit
   masks, and combined masks/values produce boolean Dataset variables through
   `.cf.flags`; constructing each valid expression executed zero Dask tasks.
   For independent mask 1, pixels `[0, 1, 2, 3]` yield cloud
   `[False, True, False, True]`.
3. **Unknown names need a caller policy.** `.cf.isin(["typo"])` returned all
   False in the tested valid flag cases rather than raising. Do not assume it
   validates user-supplied class names.
4. **GeoSave categorical semantics are broader.** Palette-only and partial flag
   declarations produced `.cf.is_flag_variable == False`, but
   `.gs.is_categorical == True`. Replacing the latter weakens existing guards.
5. **Legend combined-bitfield inconsistency.** `Legend.classes` returned a
   value→meaning dictionary when both masks and values were supplied, despite
   documenting None for bitfields. This needs review before class statistics,
   colors, or STAC publication are delegated or expanded. STAC `build_classes`
   already has an explicit `flag_masks` guard, so this finding is not a claim
   that current STAC publication emits incorrect combined classes. It is not
   repaired here.
   Additionally, masks `[1, 3]` paired with values `[1, 1]` produce distinct valid
   predicates in cf-xarray, but Legend rejects the repeated values. Enumeration
   uniqueness checks must not be applied indiscriminately to combined bitfields.
6. **Coordinates have attrs.** Latitude values placed in Dataset root attrs were
   not found; `lat(row)` in coords with standard_name latitude was found.
   However `.cf.mean("latitude")` failed on this auxiliary-coordinate layout:
   installed dimension resolution filters matching names to actual dimensions.
7. **Metadata location is key-specific.** Moving axis or standard_name into
   encoding removed discovery. Bounds and cell-measure references in encoding
   still resolved. There is no blanket attrs/encoding equivalence.
8. **Selection carries relationships.** Selecting an air_temperature DataArray
   carried its compatible area coordinate; selecting a Dataset also carried its
   bounds. Single selection with duplicate standard names raised KeyError;
   list selection included both measurements. No pixel tasks executed.
9. **CF Dataset is broader than GeoSave's raster Dataset.** Adding a linked
   `x_bounds(x, nv)` data variable made `.gs.geobox` return None and
   `.gs.validate(georeferenced=True)` fail on the support variable.
10. **Naming precedence needs an explicit policy.** Conventional y/x dimensions
    won even when axis attrs declared them X/Y in reverse. Auxiliary axes
    `north(row)` and `east(col)` were found by `.cf` but excluded by GeoSave's
    dimension-coordinate restriction.
11. **Temporal ingestion is not fully integrated.** Existing unannotated time
    remained absent from `.cf.coordinates` after Item stamping. ODC and newly
    created STAC time were covered by the first-stage tests; that did not prove
    Rasterio, arbitrary files, or caller-created objects had the same contract.
12. **Representation changes are not metadata inspection.** `guess_coord_axis`
    returned a copy with added attrs in the probe; `add_bounds` estimated edges
    around points. Installed `decode_vertical_coords` mutates the Dataset in
    place. These cannot be silently applied as validation.

13. **Related bounds methods differ.** With a bounds reference only in
    `time.encoding`, `.cf.get_bounds("time")` succeeded but
    `.cf.get_bounds_dim_name("time")` raised ValueError in 0.11.3. Installed
    source for the latter searches attrs and asserts an extra dimension of size
    2 or 4. Do not assume every CF helper handles decoded references equally.
14. **Categorical grouping is available.** A local `FlagGrouper` example grouped
    values `[10, 20, 40, 80]` under quality codes `[0, 1, 1, 2]` into labelled
    means `clear=10`, `cloud=30`, `water=80`. This may help class summaries, but
    does not itself provide GeoSave's nodata exclusion or pooled-percent policy.

### Recheck checklist before claiming integration

- [ ] Audit every ingestion source's actual coordinates, attrs and encoding.
- [ ] Decide CF support-variable ownership for each operation; do not globally
      drop every ancillary variable or treat all support variables as coordinates.
- [ ] Preserve per-band metadata across stacking and extraction.
- [ ] Resolve declared-axis ambiguity and conventional-name conflicts explicitly.
- [ ] Establish time discovery and observed-bounds policy for Dataset and DataArray.
- [ ] Retain active CRS links, stale-carrier cleanup and DataTree inheritance.
- [ ] Validate declarations through attrs groups before relying on CF flag predicates.
- [ ] Exercise enum, bitmask and combined flag behavior, missing names and nodata.
- [ ] Verify native-file and STAC publication round trips after representation changes.
- [ ] Verify lazy pixel construction for every new delegated operation.
- [ ] Compare latest documentation claims with the supported installed version.
- [ ] Audit actual changed source as well as behavior; the missed bounds edit shows
      that passing behavioral tests alone cannot prove dependency adoption.

**Current conclusion:** GeoSave partially adopts cf-xarray. The main remaining
work is the raster/metadata representation contract and operation ownership,
not replacing attrs groups or wrapping every `.cf` method.
