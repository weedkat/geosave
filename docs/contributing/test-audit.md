# Test use-case audit — 2026-11-12

The starting inventory parsed all **1,201 test functions in 128 test modules**,
including parametrized tests and methods inside test classes. The review groups
coverage by the operation's source owner and the observable workflow it protects.
Passing or failing is not the criterion for retaining a test. Existing source and
test edits were present before this audit; the counts below compare against that
working-tree snapshot, not HEAD.

## Decisions by workflow

| Test group | Functions before | Decision and retained use case |
| --- | ---: | --- |
| `cli/` | 13 | Generate usable workspaces, preserve user files, discover recipes and instantiate generated configs. |
| `examples/` | 8 | Execute documented workflows. Keep the notebook migration failures visible; notebooks were not edited. |
| `flows/` | 10 | Load label tables and define geographic regions for workflows. |
| `geodata/attrs/` | 28 | Interpret CF packing and flags, display colours, ACDD extents and attr-group updates without pixel reads. |
| `geodata/core/raster/utils/` | 18 | Preserve CF file metadata, pool statistics correctly and convert tensor dtypes without accidental reads. |
| `geodata/core/raster/` | 292 | Keep grids, pixels, time cells, band identity, declared extent and lazy execution through accessors, chips and transforms. |
| `geodata/core/vector/` | 92 | Preserve Item rows and positional sample membership; rasterize labels and read viewport windows on explicit grids. |
| `geodata/core/` | 14 | Register native accessors and describe geographic/time regions independently of raster grids. |
| `geodata/features/` | 28 | Compute indices, cloud and shadow masks with correct units, chunk edges, coordinates and missing pixels. |
| `geodata/io/formats/` | 141 | Round-trip pixels and metadata, retain portable assets and reject partial writes or metadata loss. |
| `geodata/io/` | 60 | Dispatch readers, manage resource lifetimes and storage credentials, and format safe output paths. |
| `geodata/stac/` | 109 | Translate native Items, search collections, preserve observation properties, load assets and describe saved files. |
| `geodata/utils/` | 9 | Parse colours, query geographic names and combine property mappings used by callers. |
| `geodata/viz/explore/` | 92 | Draw and identify selected data, bound viewport reads, release subscriptions, serve sessions and export usable snapshots. |
| `geodata/viz/` | 30 | Draw correct classes/RGB, preserve missing pixels and save static figures. |
| `geodata/` | 6 | Use ODC spatial names in memory and CF conventions at file boundaries. |
| `ml/` | 26 | Build training components, compute metrics, log predictions and apply joint image/target augmentation. |
| `model/` | 140 | Route native model values and gradients, execute recipes once, ingest/restore rasters and reload released weights. |
| `templates/` | 75 | Run each generated recipe with its own loss, target semantics, augmentation, checkpoint and prediction behavior. |
| `root` | 10 | Import without unwanted training/UI dependencies and install usable packaged workspaces. |

## What was removed or consolidated

- Retired `raster.gs.to_item`, `to_catalog` and Dataset COG partitioning
  signatures no longer define a supported workflow. Their xfailed old path,
  split, overwrite and publication contracts were removed. File writers,
  native Item publication, collection/property persistence and tree COG writing
  retain their own real round trips. One guard checks that paused publication
  refuses before reading pixels or creating files.
- The deleted `utils/grid.py` suite repeated current netCDF/Zarr profile,
  foreign-coordinate and DataTree grid/metadata tests. Its four cases were removed.
- A supposed root array-constructor test only tested a fixture. Another tested
  native `DataArray.to_dataset` without a GeoSave operation. Both were removed.
- Repeated empty-input validation, renamed-grid lookup and CF decode checks were
  consolidated into the retained validation, native-axis and packing workflows.
- General `GDALBand` colour-interpretation preservation and independent
  packing/nodata clearing described removed abstractions. Retained tests now
  exercise explicit `Display.rgb`, `Display.color_map`, `CFFlags` and the whole
  `CFPacking` group. External RGB reading and rewriting share one round trip.
- Arbitrary time-dimension selection in `expand_stac` no longer exists; the
  raster contract uses `time`. A second duplicate-time rejection test duplicated
  the existing formatted-ID collision test.
- Explorer's mixed-grid display is a real use case. Its test now creates native
  Items from saved files instead of failing in the paused raster publisher.

The stacking rejection matrix now uses five complete, meaningful pairs of band
metadata. Different `cell_methods` can describe different feature bands; a
`calendar` attr on pixel variables is not a supported temporal-axis contract;
and `_Unsigned=true` on the existing uint16 fixture did not change its pixel
interpretation. Signed-to-unsigned CF decoding remains covered separately.
The former policy-by-key Cartesian product also duplicated the separate policy
preservation test. Error assertions retain the rejected operation without fixing
native dependency error wording or requiring names to stay absent.

No numerical warp regression, training recipe, resource-ownership regression,
external-service test or browser test was removed merely because it is expensive,
excluded from the default run, or currently failing. Examples remain consumer
smoke tests; their notebooks were not changed.

## Source ownership

`tests/geodata`, `tests/model`, `tests/ml` and `tests/cli` mirror the corresponding
packages below `src/geosave`. `tests/flows` mirrors `src/geosave_flows`.
`tests/templates/tasks/<task>/<method>` follows the generated recipe owner in
`src/templates/tasks`, without repeating its packaging-only `modules` path.
Root import/packaging tests and example smoke tests are intentional cross-package
checks, not extra production modules.

| Previous test location | Current owner |
| --- | --- |
| `core/raster/test_crop.py` | `core/raster/transform/test_extent.py` |
| `core/raster/test_mask.py` | `core/raster/transform/test_nodata.py` |
| `core/raster/test_reproject.py` | `core/raster/transform/test_warp.py` |
| `core/raster/test_temporal.py` | `core/raster/transform/test_temporal.py` |
| `core/raster/test_vectorize.py` | `core/raster/transform/test_vector.py` |
| `core/raster/test_colorize.py` | `core/raster/test_dataarray.py` |
| `core/raster/test_cf_support.py` | CF utilities, mask/warp and netCDF owners |
| `core/raster/test_packing.py`, `test_extent.py`, `test_persistence.py` | Concrete accessor and file-reader owners |
| `core/vector/test_rasterize.py` | `core/vector/test_catalog.py` |
| `io/formats/test__stores.py` | `io/formats/utils/test_stores.py` |
| `io/formats/test_geotiff_files.py` | `io/formats/test_geotiff.py` |
| `io/test_packing.py` | Format writers, openers and vector Item window reading |
| `stac/test_load_metadata.py` | `stac/test_load.py` |
| `viz/explore/test_snapshot.py` | `test_explorer.py` and `page/test_export.py` |

## File inventory

Every original test module is accounted for below. “Keep” retains the workflow
group above, including its meaningful rejection cases. “Rehome” moves existing
coverage; it does not claim the behavior belongs to a removed source module.

| Original test file (below `tests/`) | Functions | Disposition |
| --- | ---: | --- |
| `cli/commands/test_create.py` | 8 | Keep 8 |
| `cli/core/test_templates.py` | 1 | Keep 1 |
| `cli/core/test_workspace.py` | 4 | Keep 4 |
| `examples/test_notebooks.py` | 8 | Keep 8 |
| `flows/configs/test_region.py` | 3 | Keep 3 |
| `flows/tasks/test_labels.py` | 7 | Keep 7 |
| `geodata/attrs/groups/test_acdd.py` | 1 | Keep 1 |
| `geodata/attrs/groups/test_base.py` | 1 | Keep 1 |
| `geodata/attrs/groups/test_cf.py` | 12 | Keep 12 |
| `geodata/attrs/groups/test_display.py` | 4 | Keep 4 |
| `geodata/attrs/test_combine.py` | 7 | Keep 7 |
| `geodata/attrs/test_stamp.py` | 3 | Keep 3 |
| `geodata/core/raster/test_base.py` | 32 | Keep 24, Remove 8 |
| `geodata/core/raster/test_cf_support.py` | 5 | Rehome 5 |
| `geodata/core/raster/test_chips.py` | 13 | Keep 13 |
| `geodata/core/raster/test_colorize.py` | 4 | Rehome 4 |
| `geodata/core/raster/test_crop.py` | 17 | Rehome 17 |
| `geodata/core/raster/test_dataarray.py` | 24 | Keep 21, Remove 3 |
| `geodata/core/raster/test_dataset.py` | 64 | Keep 56, Remove 7; concurrent removal 1 |
| `geodata/core/raster/test_datatree.py` | 49 | Keep 48, Remove 1 |
| `geodata/core/raster/test_extent.py` | 4 | Rehome 4 |
| `geodata/core/raster/test_mask.py` | 16 | Rehome 16 |
| `geodata/core/raster/test_packing.py` | 9 | Rehome 8, Remove 1 |
| `geodata/core/raster/test_persistence.py` | 5 | Rehome 5 |
| `geodata/core/raster/test_reproject.py` | 13 | Rehome 13 |
| `geodata/core/raster/test_temporal.py` | 27 | Rehome 27 |
| `geodata/core/raster/test_vectorize.py` | 10 | Rehome 10 |
| `geodata/core/raster/utils/test_grid.py` | 4 | Remove 4 |
| `geodata/core/raster/utils/test_statistics.py` | 12 | Keep 12 |
| `geodata/core/raster/utils/test_tensor.py` | 2 | Keep 2 |
| `geodata/core/test_region.py` | 13 | Keep 13 |
| `geodata/core/test_register.py` | 1 | Keep 1 |
| `geodata/core/vector/test_catalog.py` | 38 | Keep 36, Remove 2 |
| `geodata/core/vector/test_item.py` | 11 | Keep 11 |
| `geodata/core/vector/test_rasterize.py` | 10 | Rehome 10 |
| `geodata/core/vector/test_samples.py` | 10 | Keep 10 |
| `geodata/core/vector/utils/test_sampling.py` | 23 | Keep 23 |
| `geodata/features/test_cloud_mask.py` | 11 | Keep 11 |
| `geodata/features/test_dask_mapping.py` | 5 | Keep 5 |
| `geodata/features/test_shadow_mask.py` | 7 | Keep 7 |
| `geodata/features/test_spectral_indices.py` | 5 | Keep 5 |
| `geodata/io/formats/test__stores.py` | 8 | Rehome 8 |
| `geodata/io/formats/test_geojson.py` | 6 | Keep 6 |
| `geodata/io/formats/test_geopackage.py` | 6 | Keep 6 |
| `geodata/io/formats/test_geoparquet.py` | 17 | Keep 17 |
| `geodata/io/formats/test_geotiff.py` | 37 | Keep 36, Remove 1 |
| `geodata/io/formats/test_geotiff_files.py` | 7 | Rehome 6, Remove 1 |
| `geodata/io/formats/test_netcdf.py` | 10 | Keep 10 |
| `geodata/io/formats/test_stac_geoparquet.py` | 27 | Keep 27 |
| `geodata/io/formats/test_zarr.py` | 23 | Keep 23 |
| `geodata/io/test_backend.py` | 5 | Keep 5 |
| `geodata/io/test_openers.py` | 14 | Keep 12, Remove 2 |
| `geodata/io/test_packing.py` | 5 | Rehome 5 |
| `geodata/io/test_paths.py` | 7 | Keep 5, Remove 2 |
| `geodata/io/test_publication.py` | 8 | Remove 8 |
| `geodata/io/test_rasterio.py` | 7 | Keep 7 |
| `geodata/io/test_storage.py` | 14 | Keep 14 |
| `geodata/stac/extensions/test_classification.py` | 3 | Keep 3 |
| `geodata/stac/extensions/test_eo.py` | 2 | Keep 2 |
| `geodata/stac/extensions/test_projection.py` | 3 | Keep 3 |
| `geodata/stac/extensions/test_raster.py` | 6 | Keep 6 |
| `geodata/stac/search/test_api.py` | 7 | Keep 7 |
| `geodata/stac/search/test_query.py` | 2 | Keep 2 |
| `geodata/stac/search/test_table.py` | 10 | Keep 10 |
| `geodata/stac/test_band.py` | 1 | Keep 1 |
| `geodata/stac/test_catalog.py` | 5 | Keep 5 |
| `geodata/stac/test_item.py` | 11 | Keep 9, Remove 2 |
| `geodata/stac/test_load.py` | 21 | Keep 21 |
| `geodata/stac/test_load_metadata.py` | 18 | Rehome 17, Remove 1 |
| `geodata/stac/test_publish.py` | 15 | Keep 12, Remove 3 |
| `geodata/stac/test_raster.py` | 5 | Keep 5 |
| `geodata/test_conventions.py` | 6 | Keep 6 |
| `geodata/utils/test_color.py` | 2 | Keep 2 |
| `geodata/utils/test_geocode.py` | 2 | Keep 2 |
| `geodata/utils/test_mapping.py` | 5 | Keep 5 |
| `geodata/viz/explore/layers/test_raster.py` | 23 | Keep 23 |
| `geodata/viz/explore/layers/test_vector.py` | 12 | Keep 12 |
| `geodata/viz/explore/test_explorer.py` | 38 | Keep 38 |
| `geodata/viz/explore/test_mercator.py` | 6 | Keep 6 |
| `geodata/viz/explore/test_server.py` | 9 | Keep 9 |
| `geodata/viz/explore/test_snapshot.py` | 4 | Rehome 4 |
| `geodata/viz/plot/test_raster.py` | 18 | Keep 18 |
| `geodata/viz/plot/test_vector.py` | 4 | Keep 4 |
| `geodata/viz/test_colors.py` | 5 | Keep 5 |
| `geodata/viz/test_shapes.py` | 3 | Keep 3 |
| `ml/builders/test_criterion.py` | 1 | Keep 1 |
| `ml/builders/test_optimizer.py` | 3 | Keep 3 |
| `ml/builders/test_scheduler.py` | 1 | Keep 1 |
| `ml/callbacks/test_calibrate.py` | 1 | Keep 1 |
| `ml/callbacks/test_segmentation.py` | 3 | Keep 3 |
| `ml/metrics/test_multiclass.py` | 3 | Keep 3 |
| `ml/metrics/test_regression.py` | 1 | Keep 1 |
| `ml/test_cli.py` | 2 | Keep 2 |
| `ml/transforms/test_augmenter.py` | 8 | Keep 8 |
| `ml/transforms/test_mixing.py` | 3 | Keep 3 |
| `model/chain/test_chain.py` | 20 | Keep 20 |
| `model/chain/test_published.py` | 7 | Keep 7 |
| `model/chain/test_step.py` | 18 | Keep 18 |
| `model/release/test_huggingface.py` | 11 | Keep 11 |
| `model/spec/test_model.py` | 34 | Keep 34 |
| `model/stages/decoder/test_dpt.py` | 1 | Keep 1 |
| `model/stages/decoder/test_unet.py` | 1 | Keep 1 |
| `model/stages/encoder/test_clay.py` | 5 | Keep 5 |
| `model/stages/encoder/test_prithvi.py` | 7 | Keep 7 |
| `model/stages/encoder/test_small_cnn.py` | 1 | Keep 1 |
| `model/stages/head/test_classification.py` | 1 | Keep 1 |
| `model/stages/head/test_regression.py` | 3 | Keep 3 |
| `model/stages/head/test_segmentation.py` | 1 | Keep 1 |
| `model/stages/monolith/test_ibm_granite_biomass.py` | 1 | Keep 1 |
| `model/test_ingestion.py` | 4 | Keep 4 |
| `model/test_processing.py` | 9 | Keep 9 |
| `model/test_registry.py` | 16 | Keep 16 |
| `templates/tasks/classification/supervised/test_data.py` | 1 | Keep 1 |
| `templates/tasks/classification/supervised/test_module.py` | 3 | Keep 3 |
| `templates/tasks/pretraining/mim/test_data.py` | 2 | Keep 2 |
| `templates/tasks/pretraining/mim/test_module.py` | 7 | Keep 7 |
| `templates/tasks/regression/mean_teacher/test_data.py` | 3 | Keep 3 |
| `templates/tasks/regression/mean_teacher/test_module.py` | 7 | Keep 7 |
| `templates/tasks/regression/supervised/test_data.py` | 2 | Keep 2 |
| `templates/tasks/regression/supervised/test_module.py` | 8 | Keep 8 |
| `templates/tasks/segmentation/fixmatch/test_data.py` | 5 | Keep 5 |
| `templates/tasks/segmentation/fixmatch/test_module.py` | 5 | Keep 5 |
| `templates/tasks/segmentation/supervised/test_data.py` | 11 | Keep 11 |
| `templates/tasks/segmentation/supervised/test_datasets.py` | 3 | Keep 3 |
| `templates/tasks/segmentation/supervised/test_module.py` | 18 | Keep 18 |
| `test_init.py` | 3 | Keep 2, Remove 1 |
| `test_layering.py` | 5 | Keep 5 |
| `test_packaging.py` | 2 | Keep 2 |

## Verification

Final inventory: **118 test modules and 1,153 functions**, down from 128 and
1,201. This audit removed **47 functions**, consolidated 19 redundant or invalid
parameter combinations in the stacking matrix, and rehomed retained tests. One
additional bounds-reference test was removed by concurrent work on `dataset.py`;
that removal is not attributed to this audit. Test modules contain 21,219 lines,
compared with 22,197 in the starting snapshot.

| Check | Result |
| --- | --- |
| Default suite excluding `test_explorer.py` | 1,456 passed, 2 failed, 4 xfailed, 42 deselected; 130 warnings |
| Isolated `test_explorer.py` | 50 passed; 7 warnings |
| `test_server.py -m slow` | 6 passed, 3 skipped because Playwright is not installed |
| Final accessor/Item check | 148 passed, 2 deselected; 1 warning |
| `ruff check src tests` | Passed |
| Ruff formatting of changed/new Python test files | 27 files passed |
| `git diff --check -- tests` | Passed |

The two failures are both
`test_prediction_uses_image_grid_without_reference_labels`, for
`00_basics.ipynb` and `01_model.ipynb`: their prediction cells still call removed
`GeoDataArray.write_grid`. These are valid consumer workflows with stale
notebooks, so the tests remain. No notebooks were changed.

The four expected failures are the catalog notebook smoke test blocked by paused
raster publication, and three lazy cross-CRS interpolation discrepancies against
the full-grid Rasterio reference. The latter remain numerical regression coverage.
External-service and other slow tests remain excluded by the default markers;
the full integration and slow suites were not run.

The initial unrestricted baseline was interrupted during slow threaded Explorer
reprojection after 49 failures, 1,147 passes and 43 expected failures. Running
Explorer separately subsequently completed in 288 seconds; the broad final run
took 132 seconds. These timings are verification observations, not a performance
claim. Concurrent source/test edits were preserved, so before/after failure counts
must not all be attributed to this cleanup.

Whole-tree formatting also identifies four pre-existing files outside this
cleanup: `examples/test_notebooks.py`, `geodata/attrs/groups/test_cf.py`,
`geodata/attrs/test_stamp.py` and `geodata/viz/plot/test_raster.py`. Their formatting
was left intact.

## Second pass: fixture behavior and coverage map

The maintained [coverage map](../../tests/README.md) now describes suite hierarchy,
caller workflows and protected outcomes without listing individual tests. The
inventory above remains a record of the first pass, not a live coverage index.

This pass traced the suspicious persistence and Item cases through test helpers
and their actual production consumers. Across the six changed test modules,
194 test functions became 181; 3,101 lines became 2,863. This is a net reduction
of 13 functions and 238 lines relative to the start of this pass.

- Dataset COG naming/splitting cases called `tests.conftest.write_cogs`, which
  implements its own date/variable partitioning. They did not exercise
  `Dataset.gs.to_cog`, which is paused. Removed fixture path-format assertions
  and redundant units/overwrite cases; real path formatting, TIFF metadata,
  batch overwrite and tree COG coverage remain with their owners.
- Preserved rejection of an extra nonspatial axis as a direct GeoTIFF writer
  case, including verification that no file is created. Removed two unused
  Dataset test helpers and their obsolete constant/imports.
- Removed a tree Item-indexing case whose IDs and collections were manufactured
  by `write_collection_cogs` and `item_table`. Removed a shape-only crop duplicate;
  the transform suite checks cropped pixels, grid and root metadata together.
- Consolidated deferred tree writing across COG, netCDF and Zarr. Dataset and tree
  checks now observe Dask execution before compute and reopen saved pixels,
  instead of spying on private `_write` calls and backend chunk containers.
- Removed Item cases where fixture code selected composite intervals, converted
  equal boundaries into an instant, or omitted arbitrary coordinate properties.
  These assertions could pass without a production implementation of their
  purported contracts. Direct time normalization/property selection and actual
  file-period round trips remain. Moved band-attribute translation to `test_band`.
- Removed the STAC GeoParquet footer-only duplicate. The format-selection round
  trip already checks STAC encoding, codec, collection metadata and Item identity.

No production code or notebooks were changed in this pass. The broad first-pass
results above are historical; this pass uses focused validation of edited suites.

Second-pass verification:

| Check | Result |
| --- | --- |
| Six edited suites: Dataset, DataTree, STAC Item/band, GeoTIFF and STAC GeoParquet | 230 passed, 4 integration cases deselected, 23 warnings (17.48 s) |
| `ruff check src tests` | Passed |
| Formatting of six edited test modules | Passed |
| `git diff --check -- tests docs/contributing/test-audit.md` | Passed |

The sandboxed focused run stalled during storage checks and was interrupted;
rerunning the same scope outside the sandbox completed with the result above.
The full suite, slow checks and external-service integration were not rerun.
