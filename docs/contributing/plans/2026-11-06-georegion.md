# GeoRegion rename

Approved scope: rename the geographic/time value from GeoAnchor to GeoRegion,
accessors from `.gs.anchor` to `.gs.region`, and `set_anchor` to `set_region`.
Preserve geometry, time, grid separation and all operational behavior. ODC's
pixel-placement `anchor` is unchanged. No compatibility aliases or notebook edits.

Execute inline using the executing-plans workflow in the existing working tree;
preserve unrelated edits. Rename associated flow configuration types, the flow
argument and CLI flag to region. Move the owning modules and mirrored tests to
`region.py` and `test_region.py`. Update active documentation links and generated
API documentation. Loader filtering and multi-resolution design are separate work.

- [x] Migrate existing region/query behavior tests and observe the missing API.
- [x] Rename production definitions, exports, callers and flow configuration.
- [x] Update documentation and regenerate API references.
- [x] Run focused geometry/query/accessor/model/flow checks and inspect remaining
      anchor references to distinguish pixel placement and calendar terminology.
- [x] Record verification and breaking changes in the review.

Verification: 194 tests passed across the completed selections, with 51
deselections and dependency warnings. Two localhost flow fixtures initially
failed under sandbox restrictions; both passed with socket access. The broader
geodata selection was interrupted; an isolated Zarr round trip timed out after
30 seconds. Remaining ODC metadata tests passed separately.

Targeted Ruff lint, formatting of 45 renamed Python files, generated API
freshness and whitespace checks passed. Warp's docstring was clarified without
reformatting unrelated pre-existing code; its existing formatter differences
remain. Two BasedPyright diagnostics reproduce in the pre-rename snapshot:
bounding-box tuple length and optional/GCP grid returned by raster configuration.
They remain outside this behavior-preserving rename. Read-only review found no
blocking implementation issue; its active-documentation corrections were applied.
