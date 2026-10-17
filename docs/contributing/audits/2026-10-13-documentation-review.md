# Documentation review — 2026-10-13

This audit reviewed the current dirty working tree, preserving the starting
content captured in `/tmp/geosave-docs-tests-audit/docs`. Notebook contents were
excluded. The initial inventory contains **85 Markdown documentation files**,
plus `README.md`, `zensical.toml`, and `scripts/api_reference.py`. Authored prose
and examples were read; generated pages were checked against the generator's
complete static source inventory and rendered anchors.

## Changes and evidence

- Repaired the model registration example's invalid dictionary syntax and
  preserved its `Published` constructor-width contract. Executing the guide's
  five local blocks returns `(1, 3, 16, 16)` from the final registered chain.
- Made the raster map example run without an undefined `fields` variable.
  Named vector overlays remain documented separately.
- Corrected training and learning-path text: notebook 2 acquires imagery;
  notebook 3 prepares features and uses task sample dictionaries. The training
  guide's `class_path` model can reload a Lightning checkpoint, but native
  releases require a registered-name construction recipe.
- Described COG writer results as `RasterFile` records with reusable headers,
  and listed the implemented DataTree NetCDF writer. Added `RasterFile` to the
  topic I/O reference. Removed `describe` from its former STAC writer owner
  and `fields_named` from the COG owner's member list; their current owners
  remain covered by the complete API reference.
- Source I/O modules changed concurrently during this audit: `readers.py`
  became `openers.py`, and format readers now use `open_raster`, `open_stack`,
  or `open_vector`. Reconciled the manual I/O reference with the final source.
  Existing generated-reference updates were preserved and freshness checked.
- Marked dated plans, specs, cleanup records, and audits as historical
  evidence. Corrected status summaries for deferred catalog folders, partial
  preparation/cuts plans, the singular STAC builder and RasterFile return
  contract. Kept the original design rationale and experiments; historical
  code blocks are proposals, not runnable current examples.
- Removed stale deferred notebook numbering from open decisions and corrected
  the documentation footer from GeoMoka to GeoSave.

No public API, source behavior, or notebook was changed by the documentation
review. No uncertain documentation was deleted. README content and the source
API generator were retained.

## Validation

- All **110 active Python code blocks** parse; the original registered-chain
  example was the sole syntax failure before correction.
- All **69 navigation entries** refer to existing pages.
- Ten complete offline geodata pages: **43 blocks passed**, including native
  COG, catalog, window, local STAC search/load, metadata and tensor paths.
  Repeated after the concurrent I/O rename with the same result.
- Four additional raster/store/spec guides: **26 blocks passed** with local
  runtime/socket access, including Zarr/NetCDF round trips, Explorer creation,
  lazy reductions and ModelSpec preparation. The restricted Zarr attempt was
  interrupted when synchronization stalled.
- The complete training guide's four blocks passed: CPU fit, validation,
  named-input prediction, explicit checkpoint save and task reload. Its shown
  rounded validation metrics matched the run (`0.3272`, `0.2432`, `0.7155`).
  Altogether **80 local Python blocks across 17 guide pages** were exercised,
  including the composition and native release checks below.
- Model composition: five blocks passed. Native local release save/load:
  output shape and values matched before and after persistence.
- Zensical builds passed in **34.79 s** and **25.63 s**, including reconciled
  format-reader directives and this ledger. Static API freshness and rendered
  coverage passed: **423 definitions + 1,027 fields/methods = 1,450/1,450**.
- Initial rendered-link scan found only three pending audit-ledger links whose
  target records were still being written by this review team. Final integrated
  build/link checks are owned by [the root review](2026-10-13-docs-tests-review.md)
  after all review records exist.
- Scoped `git diff --check` passed.

External service searches, image downloads, Hub publication and notebook
execution were excluded. Local snippets with explicit prerequisites were
checked in their guide context; an importable `my_project.models` is required
for the class-path fragment, and pretrained weights require separate downloads.
Historical test counts and timing measurements are intentionally retained as
snapshots, not advertised as current totals. Source docstring gaps remain
source-owned and visible through `scripts/api_reference.py --audit`.

## Per-file coverage

Each initial documentation file is listed once. Generated directives were
verified by regeneration freshness and rendered API anchors; historical
records were checked for clear status and current-guide links, rather than
executed as present-day API recipes.

| File | Evidence and disposition |
| --- | --- |
| `docs/activities.md` | Read navigation, native-object descriptions and learning-path links; retained current content. |
| `docs/basics/attrs.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/basics/index.md` | Read navigation, native-object descriptions and learning-path links; retained current content. |
| `docs/basics/io.md` | Added the implemented DataTree NetCDF writer to the format table. |
| `docs/basics/rasters.md` | Removed undefined fields from the runnable map call; explained overlays separately. |
| `docs/basics/vectors.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/concepts/catalog-windows.md` | Read object/metadata/training contracts against source and active guide paths. |
| `docs/concepts/data-flow.md` | Read object/metadata/training contracts against source and active guide paths. |
| `docs/concepts/index.md` | Target-oriented training wording matches validity-free task samples. |
| `docs/concepts/lazy-data.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/concepts/metadata.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/concepts/model-inputs.md` | Read object/metadata/training contracts against source and active guide paths. |
| `docs/concepts/raster-grids.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/concepts/training.md` | Read object/metadata/training contracts against source and active guide paths. |
| `docs/contributing/architecture.md` | Read navigation, native-object descriptions and learning-path links; retained current content. |
| `docs/contributing/audits/2026-10-13-cleanup.md` | Read and retained historical audit evidence; added an explicit snapshot notice. |
| `docs/contributing/audits/2026-10-13-cli-flows-cuts.md` | Read and retained historical audit evidence; added an explicit snapshot notice. |
| `docs/contributing/audits/2026-10-13-geodata-tests.md` | Read and retained historical audit evidence; added an explicit snapshot notice. |
| `docs/contributing/audits/2026-10-13-geodata.md` | Read and retained historical audit evidence; added an explicit snapshot notice. |
| `docs/contributing/audits/2026-10-13-io-stac.md` | Read and retained historical audit evidence; added an explicit snapshot notice. |
| `docs/contributing/audits/2026-10-13-ml-task-structure.md` | Read and retained historical audit evidence; added an explicit snapshot notice. |
| `docs/contributing/audits/2026-10-13-model-ml.md` | Read and retained historical audit evidence; added an explicit snapshot notice. |
| `docs/contributing/documentation.md` | Read navigation, native-object descriptions and learning-path links; retained current content. |
| `docs/contributing/maintenance-ledger.md` | Marked the old cleanup ledger as a historical validation and API snapshot. |
| `docs/contributing/open-decisions.md` | Replaced obsolete deferred notebook numbering with the current four-file path. |
| `docs/contributing/plans/2026-10-12-dependency-extras.md` | Read proposal/code and retained rationale; corrected or added explicit historical/current status. |
| `docs/contributing/plans/2026-10-12-file-describes-itself.md` | Read proposal/code and retained rationale; corrected or added explicit historical/current status. |
| `docs/contributing/plans/2026-10-13-ml-task-data.md` | Read proposal/code and retained rationale; corrected or added explicit historical/current status. |
| `docs/contributing/specs/2026-10-08-prepare-design.md` | Read proposal/code and retained rationale; corrected or added explicit historical/current status. |
| `docs/contributing/specs/2026-10-09-catalog-per-collection-design.md` | Read proposal/code and retained rationale; corrected or added explicit historical/current status. |
| `docs/contributing/specs/2026-10-09-examples-series-design.md` | Read proposal/code and retained rationale; corrected or added explicit historical/current status. |
| `docs/contributing/specs/2026-10-10-cuts-windows-design.md` | Read proposal/code and retained rationale; corrected or added explicit historical/current status. |
| `docs/contributing/specs/2026-10-12-stac-structure-design.md` | Read proposal/code and retained rationale; corrected or added explicit historical/current status. |
| `docs/contributing/specs/index.md` | Read navigation, native-object descriptions and learning-path links; retained current content. |
| `docs/examples.md` | Distinguished notebook 0/1 tuple samples from notebook 3 task dictionaries. |
| `docs/faq.md` | Read navigation, native-object descriptions and learning-path links; retained current content. |
| `docs/first-raster.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/guides/benchmarks.md` | Read API/data-flow contracts against source and templates; prerequisite/network examples retained explicitly. |
| `docs/guides/catalogs-and-cuts.md` | Item creation accepts RasterFile records as well as saved paths. |
| `docs/guides/ingest-files.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/guides/large-data.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/guides/model-spec-and-release.md` | Read API/data-flow contracts against source and templates; prerequisite/network examples retained explicitly. |
| `docs/guides/model-spec.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/guides/models.md` | Replaced invalid ellipsis mapping and incomplete registered stage with runnable published-width wiring. |
| `docs/guides/pretrained.md` | Read API/data-flow contracts against source and templates; prerequisite/network examples retained explicitly. |
| `docs/guides/release.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/guides/stac.md` | Read against source contracts and exercised the local Python guide path. |
| `docs/guides/training.md` | Corrected notebook pointers and class_path release restriction; retained checkpoint reuse. |
| `docs/guides/workflows.md` | Read API/data-flow contracts against source and templates; prerequisite/network examples retained explicitly. |
| `docs/guides/workspaces.md` | Read API/data-flow contracts against source and templates; prerequisite/network examples retained explicitly. |
| `docs/index.md` | Read navigation, native-object descriptions and learning-path links; retained current content. |
| `docs/install.md` | Compared Python version, extras and development commands with pyproject.toml. |
| `docs/quickstart.md` | Save/describe table now states RasterFile headers are reused without reopening. |
| `docs/reference/anchor.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/array-accessor.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/attrs.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/benchmarks.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/cli.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/cuts.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/errors.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/features.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/generated/data.md` | Generated inventory matches current static source and all expected rendered anchors. |
| `docs/reference/generated/flows.md` | Generated inventory matches current static source and all expected rendered anchors. |
| `docs/reference/generated/models.md` | Generated inventory matches current static source and all expected rendered anchors. |
| `docs/reference/generated/training.md` | Generated inventory matches current static source and all expected rendered anchors. |
| `docs/reference/geodata.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/index.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/io.md` | Documented RasterFile, removed moved helper members, and followed current format reader names. |
| `docs/reference/metrics.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/ml.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/model-chain.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/model-release.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/model-spec.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/model-stages.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/model.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/public-api.md` | Generated inventory matches current static source and all expected rendered anchors. |
| `docs/reference/raster-accessor.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/rasters.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/stac.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/stack-accessor.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/training-tools.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/training.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/transforms.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/vectors.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `docs/reference/workflow.md` | Reviewed mkdocstrings targets, member ownership, examples and rendered links. |
| `README.md` | Reviewed current install, CLI, native raster example, learning-path links and workspace claims; retained existing content. |
| `zensical.toml` | All navigation paths exist; corrected stale footer branding. |
| `scripts/api_reference.py` | Read static inventory, freshness and rendered-anchor checks; retained existing generator. |
