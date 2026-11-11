# Explicit Collection selection

Approved direction: `tree.gs.to_catalog(..., collections=["s2", "dem"])`
selects relative subtree paths. Selected node names become Collection IDs;
node attrs supply metadata. Every raster must belong to exactly one selection.
No hierarchy depth or attribute-based boundary discovery is used. The singular
`collection=` assigns the entire input to one ID and is mutually exclusive
with `collections=`. Grid inspection is independent of publication membership.

## Implementation

1. Make `GeoDataTree.grids` inventory all raster nodes, including the root;
   remove `inspect_grids` and Collection-specific inventory columns. Test
   laziness, inherited coordinates, invalid grids, and root data.
2. Add Collection path selection to tree publication and the shared path
   planner. Reject missing, overlapping, duplicate-ID and incomplete selections
   before writes. Preserve selected-root metadata, including root raster data.
3. Test nested Collection roots through COG, Item table, GeoParquet, and reopening;
   check dates, variables, grid metadata and Collection metadata.
4. Update callers, docs and generated API references; run the affected suite,
   lint, documentation checks and review the diff.

## Review focus

- Parent and child rasters are both published.
- Unselected root data cannot disappear.
- Repeated short names in different branches are not guessed.
- Item grouping follows dates across grids, not hierarchy depth.
- Existing unrelated workspace changes remain untouched.
