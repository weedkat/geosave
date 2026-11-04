# Anchor formatting and raster metadata ownership

Implement the ownership design agreed in the review conversation.

## Contract

- GeoAnchor formats location/time for users and STAC publication. Optional
  `geobox=` supplies epsg, width, height and res; lon/lat always come from the
  anchor. Without a geobox, existing formatting is unchanged.
- STAC constructs its formatting anchor from its WGS84 union footprint and
  existing time endpoints. It still rejects grid fields for unequal asset grids.
- `core/raster/time.py` owns assigning and reading raster coverage coordinates.
  `utils/datetime.py` keeps date parsing and calendar/row calculations.
- ODC owns band resolution, grouping, fused properties and tree assembly. STAC
  owns interpretation and agreement of published metadata. Attr groups own
  stamping. No generic stamping layer or compatibility aliases.
- Preserve lazy pixels, existing Item IDs, loaded nodata, source metadata and
  unrelated worktree changes. Temporal parser validation is outside this change.

## Steps

- [x] Add public grid-formatting and raster-coordinate contract tests; run them
  before implementation. Establish focused existing-test baseline.
- [x] Consolidate formatting; move bounds functions and update all consumers.
- [x] Move common Item metadata and band agreement into STAC while retaining
  ODC-specific resolution and output adaptation in the loader.
- [x] Refresh API references and architecture documentation. Run affected tests,
  import smoke checks, formatting and lint; review the task-only changes.

## Review focus

- Projected anchors versus WGS84 Item centroids must not change existing IDs.
- Multi-grid Items allow location/time templates but reject grid templates.
- Timeless Items retain existing STAC time handling.
- Bounds preserve scalar/per-time coordinates and lazy pixel data.
- ODC aliases, grouped properties, effective nodata and Collection placement
  survive the metadata ownership change.

## Validation and compatibility

- New grid-formatting test failed on the unsupported keyword before implementation;
  the time-module test failed on the missing module.
- Initial baseline and restricted runs stalled inside Zarr. Outside the sandbox,
  the final affected suite passed: 307 passed, 4 deselected by the default
  slow/integration filter, 38 warnings. Coverage includes STAC publication,
  catalog-to-tree paths, ODC loads, temporal transforms and Zarr round trips.
- Ruff check/format, task-file diff checks, generated API freshness, import smoke
  (no torch or Prefect), and two anchor doctest examples passed.
- Independent task-diff review found no blocking issue.
- Bounds helper imports move to `core.raster.time`; ODC's shared band reader is
  replaced by `stac.band.read_shared_attrs`, taking resolved source band positions.
- Unknown Item ID fields now raise the shared formatter's ValueError, rather than
  the private formatter's KeyError. Publication preflight still leaves assets
  untouched on failure.
- The shared STAC metadata builder now also retains common Item properties for
  native catalog-to-tree paths, matching the ODC loader's metadata convention.
