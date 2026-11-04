# Executable ModelSpec processing implementation plan

> Execute inline with the executing-plans workflow. The user authorized implementation and iteration against real end-to-end flows.

**Goal:** A YAML recipe executes the same raster, tensor, and output processing in training and inference, retaining per-invocation values without a processor wrapper.

**Architecture:** Keep ordered CallSpec/StageSpec execution and native values. ModelSpec v3 adds explicit encoding, batch transforms, chip decoding, and merged-output postprocessing. Catalog/sample/chip orchestration remains explicit; existing acquisition helpers are not redesigned in this iteration. The first input boundary is a selected sample/chip with adequate spatial and temporal support.

**Stack:** Existing xarray, Dask, PySTAC, GeoPandas, PyTorch, Lightning, Pydantic, YAML.

## Decisions and limits

- Keep `to_catalog` and native persistence.
- No project-specific code is required by the demonstrator recipe; calls use installed functions.
- YAML parsing remains inert; execution invokes trusted installed code, not a sandbox.
- v3 replaces context/TransformSpec with named encoding results and ordinary calls. No compatibility aliases.
- Intermediate values are plain dictionaries. Tensor layouts and dtype requirements are checked at the model boundary.
- Training augmentation remains owned by each generated method. Explicit chip size belongs to its DataModule.
- No automatic movement of calls across sampling/chipping, halo inference, or arbitrary remote code delivery.
- Source acquisition helpers remain available while the sample-processing contract is iterated.

## Tasks

- [x] Add failing YAML workflow tests: temporal layout and lazy selection; fitted normalization and runtime inverse; logits merged before class selection; invalid references/layouts.
- [x] Implement v3 processing stages, tensor requirements, and narrowly needed native processing functions. Keep model independent of ml.
- [x] Migrate existing spec consumers, generated method configs, and normalization consumers. Repair hierarchy selection in existing catalog preparation without implicit resampling.
- [x] Run a bounded CPU Lightning training/save/reload/prediction workflow through real COG and GeoParquet persistence, plus temporal and merged-output cases.
- [x] Update user documentation with executable YAML/Python and clear scope boundaries.
- [x] Run focused checks and default suite; compare failures with the pre-change run and record remaining limitations.

## Review focus

Wrong channel/time ordering; stale per-chip inverse state; implicit eager reads; inconsistent normalization between training/inference; categorical selection before merging; extra model kwargs from intermediate results; changed working-tree content outside this task.

## Results

- Schema v3 implemented with explicit encoding, shared transforms, decoding, and postprocessing; generated methods migrated without compatibility aliases.
- End-to-end workflow: two-date packed rasters -> COGs -> GeoParquet -> samples -> chips -> generated Lightning DataModule/Module -> CPU training -> checkpoint/YAML reload -> per-chip inverse -> merged prediction -> COG/GeoParquet readback. Numerical values, grid, and interval metadata checked.
- Additional flows: explicit mixed-resolution alignment, temporal-context dimensions, per-sample inverse isolation, prepared-catalog dependencies, and merged-logit class selection.
- Review fixes: implicit variable broadcasting and omitted decoding raster dependencies both reproduced with failing tests, then fixed and verified.
- Final default suite: 1,460 passed, 41 deselected, 3 xfailed, 144 warnings. The xfails remain the existing lazy-warp discrepancies; no claim of unqualified all-tests-green.
- Import layering: 5 passed. All six shipped YAML recipes executed through normalization against independent numerical expectations.
- Ruff source/tests check, focused formatting, API-reference freshness, and scoped diff whitespace checks passed.
- Zensical build completed, with docstring warnings in existing modules. An initial attempt to relocate the site using an unsupported CLI flag/absolute config was discarded; the normal repository build passed.
- The restricted-sandbox pre-change full-suite attempt was interrupted after stalling around local-service tests. Final default verification ran with local-service permissions; no regression-count claim is based on that incomplete baseline.

## Remaining design boundaries

The caller supplies the selected sample and sufficient neighborhood/time support.
No stage movement, automatic halo calculation, generalized intermediate artifact
serialization, or augmentation/context correction is inferred. Fitted parameters
can be declared in YAML; the application must snapshot the effective recipe with
its training run and supply versioned implementation dependencies. Catalog
materialization remains raster-only. Lower-resolution predictions require an
explicit output grid and compatible merge policy. Release removal/redesign and
other review repairs are separate work.
