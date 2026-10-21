# Resolution-based chips

The caller supplies chip size and resolution. Chipping constructs each row's
output grid, records resampling, and opens no raster files. Reading a chip
honours this plan before temporal assembly; unchipped samples keep native grids.

- Infer a common CRS from all eligible source assets in each row. Do not choose
  an asset's resolution or lattice based on iteration order. Reject ambiguous
  or missing CRS; an explicit GeoBox remains available for those cases.
- Use ODC's edge-snapped grid over the footprint at a positive resolution in CRS
  units. An explicit target GeoBox fixes the lattice and cannot be combined with
  resolution. Remove the collection-name shortcut.
- Persist resampling with the chip rows. Default to nearest; allow the existing
  collection/variable mapping. Chip reads align all observations before temporal
  reductions and retain lazy arrays, fixed output shapes, and categorical values.
- Verify mixed source resolutions and pixel offsets, independent locations/CRSs,
  asset-order independence, invalid arguments, metadata-only planning, persisted
  plans, padded read/merge behavior, and unchanged native unchipped reads.
- Update ChipsSpec and notebook 3 to use size/resolution and plain chip reads.

Execution stays in this workspace to preserve existing notebook and source edits.
