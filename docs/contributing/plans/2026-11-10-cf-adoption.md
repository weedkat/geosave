# Internal cf-xarray adoption

Current implementation and operation coverage: [CF integration](../reviews/2026-11-10-cf-integration.md). The first-stage notes below are historical and superseded where that report differs.

## Implemented first stage

GeoSave uses cf-xarray 0.11.3 (already a dependency) to interpret CF metadata.
Attrs groups remain responsible for coherent declarations, validation and writing.
Native xarray owns pixel codecs and persistence; ODC owns grids and reprojection.

The first stage changes discovery and known ingestion declarations, not pixel
representation, output packing defaults, variable selection, or observed periods.

- DataArray spatial discovery keeps ODC's conventional dimension names first.
  Otherwise cf-xarray interprets declared axes and geographic coordinates. Only
  unambiguous one-dimensional dimension coordinates qualify. Curvilinear
  latitude/longitude alone do not establish an affine pair.
- Correction from follow-up audit: `from_cf_bounds` still uses manual
  attrs/encoding lookup. The intended replacement did not apply. Passing
  behavioral tests did not establish delegation to cf-xarray.
- ODC-loaded time coordinates receive `standard_name="time"` and `axis="T"`.
  This repairs a demonstrated discovery gap: datetime values alone were not
  recognized by `.cf.coordinates` on the original ODC output.
- New scalar dates supplied by STAC Item stamping carry the same declarations.
  Dates already supplied by a file retain their original metadata.

Removed: the rioxarray axis-discovery fallback. Handwritten bounds-reference
lookup remains and was incorrectly reported as removed. Explicit rioxarray-only spatial-axis configuration is
no longer consulted; custom axes need CF declarations. No compatibility alias
or additional accessor is introduced.

## Ingestion ownership map

| Source | Owner / translation | Native raster location | CF interpretation |
| --- | --- | --- | --- |
| ODC grid and CRS | ODC, then `gs.write_crs` | spatial coordinates, mapping coordinate and variable link | axes / grid mappings |
| ODC effective nodata | `stamp_band_attrs`, `Packing` | variable `_FillValue` attr | native xarray decoding, not `.cf` |
| STAC raster scale/offset | raster extension reader, `Packing` | variable attrs while pixels are packed | native xarray decoding |
| STAC raster unit | raster extension reader | variable `units` attr | semantic units metadata |
| STAC EO band fields | EO extension reader, `Spectral` | band attrs | no automatic CF standard-name equivalence |
| STAC classification | classification extension reader, `Legend` | flag attrs and GeoSave palette | CF flags; palette remains GeoSave-owned |
| ODC observation dates | `load_dataset` | datetime coordinate with CF time declarations | T axis / time coordinate |
| STAC composite periods | `_write_composite_periods`, time-bound accessor | start/end coordinates | GeoSave observed coverage; optional file CF bounds |
| ODC fused Item properties | `_stamp_date_properties` | JSON observation coordinate | STAC provenance, not a CF concept |
| Collection and shared Item properties | `build_stac_metadata`, attrs stamping | root attrs | STAC provenance, not a CF concept |

Source scale/offset agreement before ODC fusion remains necessary. ODC effective
nodata takes precedence over source sentinels. cf-xarray replaces neither rule.

## Next stages and decision gates

1. **Dataset support variables.** Define operation-specific handling of bounds,
   cell measures, ancillary rasters and mappings. Current Dataset methods often
   traverse all data variables. Do not broadly filter these by shape or hide
   them as coordinates: ancillary rasters may need processing, while cell areas
   become invalid after reprojection. Review selection, validation, masking,
   composite, reprojection, tensor conversion and publication together.
2. **Observed-period representation.** Demonstrate Dataset → band extraction →
   DataTree inheritance → publication with native CF bounds before deciding
   whether to retain bounds internally. A DataArray cannot carry a sibling
   bounds variable introducing a new dimension. Do not replace observed periods
   with midpoint bounds from `cf.add_bounds`.
3. **Active mappings versus cleanup.** CF mapping discovery can describe active
   relationships; existing stale CRS-carrier cleanup is a separate concern.
4. **Flag operations.** Use `.cf` where named flag/bitfield evaluation is needed.
   Keep Legend validation, palettes and categorical safeguards, including
   incomplete declarations. Do not replace these guards with is_flag_variable.
5. **Persistence and examples.** After each representation decision, migrate all
   affected readers/writers and examples, then remove superseded conversions.

No automatic guessing, canonical-attribute downloads, pixel decoding, or
metadata reconstruction is introduced as a side effect of inspection.

## Verification

- Raster, conventions and ODC tests: 524 passed, 2 deselected, 3 existing warp
  xfails. Includes NetCDF/Zarr round trips and a real local STAC/ODC load.
- New checks cover CF geographic units, ambiguous axes, curvilinear coordinates,
  missing declared bounds, and metadata discovery without Dask pixel execution.
- Initial ODC contract test failed on missing CF time metadata, then passed after
  the ingestion declaration was added.
- An existing test called an ODC coordinate pair unlabelled despite its geographic
  units; its fixture now removes those attrs, with the labelled case tested separately.

Known pre-existing warning: optional CF bounds serialization can let xarray
choose time and bounds units independently. This stage does not claim complete
CF compliance or resolve temporal serialization policy.
- STAC raster stamping and STAC I/O: 23 passed, 2 deselected. An assertion tied
  to the old phrase "time labels" was updated for the existing generic
  observation-label error; the mismatch still raises ValueError.
- Scoped BasedPyright: 0 errors/warnings. Ruff and API inventory freshness pass.
- The full repository suite was not run for this first stage.


## Follow-up integration audit

The first stage is not complete integration. Read-only probes confirmed:

- A Dataset with a CF-linked `x_bounds(x, nv)` data variable returns no common
  grid and fails raster validation on the bounds variable.
- Existing unannotated time coordinates remain undiscoverable by `.cf` after
  `stamp_item_time`; only newly created coordinates were annotated.
- Rasterio time creation from ACDD, TIFF tags or filenames remains unannotated.
- CF one-dimensional auxiliary axes `north(row)`, `east(col)` are discovered by
  `.cf` but excluded by the GeoSave dimension-coordinate requirement. Supporting
  them requires checking ODC placement, not merely relaxing the check.
- Conventional y/x names still override contradictory CF axis declarations.
  This preserves the old precedence but needs an explicit validation policy.
- Canonical time assumptions persist in accessors, properties, DataTree handling,
  visualization and feature kernels. These are not automatically replaceable
  with semantic time lookup without deciding ambiguity and dimension contracts.

The earlier pass counts verify the selected workflows only. They do not establish
complete CF integration, every ingestion path, or every candidate substitution.


The full documentation inventory, verified local limitations, function-level
adoption matrix and recheck checklist now live in the
[cf-xarray capability review](../reviews/2026-11-10-cf-xarray-capabilities.md).
Use that audit before advancing this implementation plan.

## Completed dependency API audit

The [API audit](../reviews/2026-11-10-cf-xarray-api-audit.md) records all 60 indexed generated pages, installed public members omitted from the index, and bounded runtime probes. Reading coverage is complete for this snapshot; numerical and GeoSave workflow coverage is not. The next gate is the Dataset representation contract across ingestion, transforms and persistence, before further production edits.
