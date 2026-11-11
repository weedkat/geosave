# Raster validation blast-radius audit

Status: boundary cleanup implemented. The findings below record the original
reproductions before cleanup.

Spatial transforms, TIFF preflight, STAC Item construction, model restoration and
map placement now call the existing raster validator before interpreting pixel
placement. Operation-specific restrictions remain local. Merge validates grid
consistency without requiring a CRS. Path planning handles selections and
destinations; publication and format writers validate raster suitability.

Regression coverage checks irregular-coordinate rejection, rejection before TIFF
creation through direct/Dataset/DataTree writers, and a useful missing-CRS map
error. Normal raster workflows still use ODC for spatial metadata.

## Scope and evidence

Traced raster accessors, transformations, TIFF persistence, publication path
planning, STAC construction, model restoration and explorer placement. Searched
source, tests, docs, templates and workspace Python/Markdown for removed validation
flags. Notebooks were excluded. No remaining executable callers of the removed
flags were found; documentation describes their removal.

The migration updated explicit calls to `validate`, but missed operations that
still implement partial checks themselves. ODC returning a GeoBox does not
establish coordinate consistency.

The local diagnostic script `/tmp/geosave_validation_blast.py` uses a 3x3 raster
with y=[2,1,0], x=[0,1,5] and EPSG:4326. Its canonical validation raises a
RasterValidationError for irregular spacing. Actual file writes use temporary
directories and are removed afterward. Results are recorded in
`/tmp/geosave-validation-blast-results.txt`. No network access is needed.

## Findings

### 1. High: GeoTIFF validation bypasses raster validity

`io/formats/geotiff.py:192` checks dimensions, time, nodata and placement type,
but never invokes the canonical raster validator. `_write` and batch preflight
both use this incomplete format validator.

Reproduced through both direct `write_geotiff` and public `Dataset.gs.to_cog`:
input x centres [0,1,5] become [0,2.5,5] in the saved file. These are changed
pixel locations without an explicit resampling operation.

The format validator should first enforce the raster guarantees appropriate to
its supported inputs, then enforce TIFF constraints. Do not automatically ban
unreferenced TIFFs: distinguish generic spatial validity from a geographic
reference required by publication. Keep one-date, common nodata, supported
placement and storage encoding checks in the format owner.

### 2. High: publication validation differs by container

`io/paths.py:_expand_paths` validates data-bearing DataTree nodes, but its Dataset
branch only reads a grid and constructs path labels. The public COG path therefore
accepts the malformed Dataset while the same data in a DataTree is rejected.

Reproduced by calling `expand_paths` on both representations, and by writing the
Dataset through `gs.to_cog`. This is a missed migration boundary, not an intended
Dataset/DataTree distinction. Planning should apply equivalent raster guarantees
to both inputs; file validation must also protect direct writer calls.

### 3. High: transforms still use discovered grids as validation

- `transform/warp.py:_read_geobox`: GeoBox type and CRS checks only.
- `transform/warp.py:_read_grids`: GeoBox type check; even its reference
  completeness check differs from `_read_geobox`.
- `transform/extent.py:crop`: checks CRS and region, then calls ODC crop.
- `transform/vector.py:_read_plane_grid`: checks grid type, CRS and plane shape.
- `transform/vector.py:_read_burn_grid`: checks target grid type and CRS.

Reprojection, crop and vectorization all accepted the invalid fixture. Rasterize
has the same bypass by source inspection; it was not separately executed.

Validate native raster inputs at the owning operation boundary before deriving
spatial results. Keep constraints on the requested target, mask, resampling,
plane rank and rasterization dtype with their operations. A GeoBox supplied
directly as a target is not an xarray raster and needs its own input contract.

### 4. Medium: STAC checks combine unrelated concerns and miss validity

`stac/item.py:check_single_date` also checks grid presence, but not CRS or
coordinate consistency. `create_item` independently checks the placement type.
The projection extension later checks CRS.

Reproduced: `check_single_date` accepts a grid without CRS; `create_item` accepts
an irregular georeferenced raster and constructs Item metadata from its inferred
grid. The missing-CRS acceptance of the helper does not establish that full
publication accepts it: later checks can reject it. Publication constructs Items
before writing files, so this audit does not claim a missing-CRS partial-write bug.

Separate raster preflight from the single-observation rule. Keep projection-field
serialization and validation of a supplied grid in the projection extension.

### 5. Medium: explorer presence check dereferences a missing CRS

`viz/explore/layers/raster.py:_check_mappable` checks only whether a geobox exists,
then reads `raster.gs.crs.geographic`.

Reproduced with numeric spatial coordinates and no CRS: AttributeError on None,
instead of a contextual validation error. Require georeferencing first; retain
the longitude-wrap rule as a web-map requirement. No viewer/server was launched.

### 6. Medium: model restoration can propagate invalid reference placement

`model/processing.py:to_raster` checks grid presence and output shape, then writes
that grid onto a new array. It never validates the reference.

Reproduced: the irregular reference is accepted. Its inferred regular placement
is transferred to the prediction. Output rank/shape checks are legitimate and
should remain after validating the reference.

### 7. Lower priority: combine contract and error text disagree

`transform/combine.py:merge` checks for a GeoBox with an error promising a
georeferenced grid, but a CRS-less grid passes. With a CRS, coordinate validation
happens indirectly through `write_crs`; without one, that path is skipped.

Reproduced: a CRS-less Dataset with irregular coordinates passes `merge([ds])`.
The fix needs an explicit operation contract for unreferenced inputs, not an
assumption that all raster combinations require a CRS. Equal coordinates,
variable dimensions and unique names remain combination-specific constraints.

## Checks that should remain local

- Regular GeoBox requirements for chip reconstruction, window reads, CF-axis
  writing and algorithms that cannot use GCP placement.
- Output shape checks, nodata/packing constraints, file path collisions and
  metadata serialization rules.
- Metre-based placement for cloud-shadow distances.
- Validation of new grids/regions supplied by callers.
- `write_grid` structure checks before replacing existing coordinates: full
  validation of the old coordinates would prevent explicit repair.
- Optional metadata discovery and inspection: reading a missing grid is not
  itself an operation failure.

NetCDF/Zarr preserve native coordinates and can represent heterogeneous data.
Their lack of a blanket shared-grid check is not evidence of the TIFF bug.
Do not add global geographic restrictions to all persistence formats.

## Recommended order

1. Fix TIFF validity preflight and Dataset/DataTree planning parity; protect the
   demonstrated changed pixel centres with a round-trip test.
2. Validate raster inputs at warp/crop/vector and model-restoration boundaries.
3. Separate STAC date checks from raster preflight; fix explorer CRS handling.
4. Clarify combine's unreferenced-grid contract and remove indirect dependence
   on write_crs for validation.

Use canonical accessors for common raster guarantees. Keep actual operation
constraints local. Avoid creating another generic grid-discovery helper that
also owns validation, traversal and error policy.

## Checks and limits

Executed the diagnostic cases above, including direct GeoTIFF and public COG
writes. Inspected all listed ownership paths and checked removed-flag callers.
No production code or tests were changed. The full suite was not rerun for this
audit; these are reproduced diagnostic findings, not completed fixes.
