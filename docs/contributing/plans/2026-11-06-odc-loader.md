# ODC loader API

## Accepted contract

- `load_dataset` and `load_datatree` take selected native PySTAC Items.
- `region: GeoRegion` supplies output geometry and optional inclusive temporal
  overlap selection. Timeless Items are excluded only when time is requested.
  Geometry bounds define the grid; no polygon mask or catalog search is implied.
- `crs`, `resolution` and ODC pixel `anchor` control grid construction.
  An exact `geobox` excludes all four construction inputs.
- Default `groupby="id"` retains individual Items. Time/solar-day mosaics are explicit.
- Remove competing raw geometry/grid aliases and redundant extent conversion.
- Preserve lazy reading, metadata policy and relative DataTree resolution families.
  ModelSpec continues forwarding its explicit solar-day default.

## Implementation and verification

1. Add real local COG tests for region coverage, time selection, grouping and
   conflicting grid inputs; observe failures against the former API.
2. Implement typed loader parameters and time selection using shared STAC readers.
3. Update guides and generated API references; run focused loader, metadata and
   model checks. Record limitations in the architecture review.

Automatic footprint union/intersection policies, native opener consolidation,
and ModelSpec extent propagation remain separate work.

## Results

- Initial contract tests: 7 failed, 5 passed against the former loader.
- Final loader/metadata/ModelSpec selection: 41 passed, 1 deselected
  (the previously stalled Zarr preparation workflow), 1 dependency warning.
- STAC instant/composite/coverage selection: 4 passed, 19 deselected,
  1 dependency warning.
- Ruff, targeted basedpyright (0 errors), generated API freshness and
  working-tree whitespace checks passed.
- A broader STAC Item run stalled after 27 completed tests and was interrupted;
  this is not a claim that the full suite passed.
- Requested-region coverage is retained without masking; automatic spatial
  Item filtering is not added. ODC determines source pixel contributions.
- End-to-end local smoke passed: GeoParquet → selected Items, lazy union grid,
  nodata gaps, explicit same-time mosaic and exact intersection grid.
- Independent scoped review found no important defects.
