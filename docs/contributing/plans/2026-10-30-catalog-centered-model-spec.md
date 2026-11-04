# Catalog-centered ModelSpec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every ModelSpec step takes a catalog and gives one: `ingest` (anchor -> raw catalog), `prepare` (raw -> prepared catalog), `sample` (prepared catalog -> GeoSamples).

**Architecture:** `ModelSpec` gains `prepare` and `sample`, composing existing geodata calls (`GeoCatalog.sample`, `GeoSamples.frame`, `GeoSamples.chip`, `DataTree.gs.to_cog`, `stac.create_items`, `stac.create_catalog`). `FramesSpec.cut` / `ChipsSpec.cut` fold into `sample`. The Prefect ingest flow writes the raw catalog.

**Tech Stack:** Python, xarray DataTree, pystac / stac-geoparquet, pydantic, Prefect, pytest.

**Spec:** `docs/contributing/specs/2026-10-30-catalog-centered-model-spec.md`

## Global Constraints

- `geodata <- model <- ml`; `model/spec` imports no torch and no `geosave.ml`; `geosave` never imports `geosave_flows`.
- No compatibility aliases; callers of removed names are migrated.
- Raster work stays lazy until written; persistence needs round-trip tests, lazy operations laziness tests.
- Do not touch notebooks or the unrelated working-tree changes (`viz/`, `utils/geocode.py`, `examples/`, `open-decisions.md`).
- Checks: `uv run ruff check .`, `uv run basedpyright <changed files>`, `uv run pytest`.

## Review Focus

- `prepare` with no path: the catalog's assets exist in a temporary folder and read back — Task 1.
- `prepare` on a raw sample missing a declared raster or a `keep` collection raises `KeyError` naming it — Task 1.
- `prepare` into an existing path raises `FileExistsError` before reading pixels — Task 1.
- `sample` on a spec with neither `frames` nor `chips` returns the plain sample table — Task 2.
- The flow's catalog writes and reads back with its Collection definitions — Task 3.

---

### Task 1: `ModelSpec.prepare`

**Files:** Modify `src/geosave/model/spec/model.py`; Test `tests/model/spec/test_model.py`

**Interfaces:**
- Produces: `ModelSpec.prepare(self, raw: GeoDataFrame[GeoCatalog], path: Location | None = None, *, target: Target | None = None, keep: Sequence[str] = ()) -> GeoDataFrame[GeoCatalog]`. For each `raw.gs.sample()` row: `rasters = row.gs.to_xarray(target=target).gs.rasters`, `results = self.preprocess(rasters)`, save `{name: results[name] for name in self.input_rasters} | {name: rasters[name] for name in keep}` with `tree(...).gs.to_cog(templates)` (one file per band, per instant when dated) under `<path>/<sample id>/`; Items id `f"{sample id}_{{start:%Y%m%dT%H%M%S}}"`, `datetime=row.gs.timespan`. Collections: raw's definitions where named, else `create_collection(name, description=f"Prepared raster {name}")`. Writes `model_spec.yaml` beside. `path=None` -> `tempfile.mkdtemp(prefix="geosave-prepared-")`.

- [ ] Tests (restored from `ff07d7e4^`, returning a table): `test_prepared_pixels_are_the_preprocessed_raw_pixels`, `test_prepare_keeps_pixels_lazy_until_written` (`written == [True, True]`), `test_prepare_refuses_an_existing_folder`, `test_prepare_refuses_keep_given_as_text`, new `test_prepare_without_a_path_writes_a_temporary_catalog` (assets exist, pixels equal `2 * nir`), new `test_prepare_names_a_missing_keep_collection` (`KeyError`, match `label`).
- [ ] Run `uv run pytest tests/model/spec/test_model.py -k prepare -v` — FAIL, then implement, then PASS.
- [ ] Commit `feat: ModelSpec.prepare turns a raw catalog into a prepared one`

### Task 2: `ModelSpec.sample`; `cut` folds in

**Files:** Modify `src/geosave/model/spec/model.py`, `src/geosave/model/spec/windows.py`; Tests `tests/model/spec/test_model.py`, `tests/model/spec/test_windows.py`

**Interfaces:**
- Produces: `ModelSpec.sample(self, catalog: GeoDataFrame[GeoCatalog], *, target: GeoBox | None = None) -> GeoDataFrame[GeoSamples]` — `catalog.gs.sample()`, then `.gs.frame(windows=, reference=, stride=)` when `frames`, then `.gs.chip(shape, target=, resolution=, overlap=, pad_mode=)` when `chips`. `FramesSpec.cut` and `ChipsSpec.cut` removed.

- [ ] Move `test_chip_spec_cuts_at_its_declared_resolution`, `test_frames_cut_as_the_spec_declares`, `test_chips_cut_as_the_spec_declares` to `test_model.py` as `test_sample_*`, calling `ModelSpec.sample` and comparing with direct `frame` / `chip` calls; add `test_sample_without_frames_or_chips_is_the_catalog_sample`.
- [ ] Run `uv run pytest tests/model/spec -v` — FAIL, implement, PASS.
- [ ] Commit `refactor: ModelSpec.sample frames and chips a catalog`

### Task 3: The ingest flow writes one catalog

**Files:** Modify `src/geosave_flows/ingest.py`, `src/geosave_flows/cli.py`; Delete `src/geosave_flows/tasks/sample.py`, `tests/flows/tasks/test_samples.py`; Test `tests/flows/test_ingest.py`

**Interfaces:**
- Produces: `ingest(anchor: AnchorConfig, *, output: str, spec: str) -> str` — refuses an existing `output`; `model.ingest(anchor.open(), f"{output}/{{collection}}/{{time:%Y%m%dT%H%M%S}}.tif")`, written with `.gs.to_geoparquet(f"{output}/catalog.parquet", collections=[create_collection(name, description=f"Raster {name} as ingested") for name in model.rasters])`; returns that path. CLI drops `--format` / `--write-options`.

- [ ] Tests: monkeypatch `ModelSpec.load_rasters` to return `{"optical": <dated raster>}`; assert the result ends in `catalog.parquet` and reads back with collection `optical` and its definition; keep the no-`stac` refusal and the slow JSON-anchor run.
- [ ] Run `uv run pytest tests/flows -v` and `-m slow` — FAIL, implement, PASS.
- [ ] Commit `feat: ingest flow writes one catalog per run`

### Task 4: Docs and callers

**Files:** `docs/guides/model-spec.md`, `docs/guides/workflows.md`, `docs/concepts/model-inputs.md`, `docs/contributing/specs/index.md`, `src/geosave/model/README.md` as hits require.

- [ ] `grep -rn "\.cut(\|write_sample\|--format\|--write-options" src docs tests` — rewrite each hit to `spec.sample(...)` / the catalog flow; add the spec to `specs/index.md`.
- [ ] `uv run ruff check .`, `uv run basedpyright src/geosave/model/spec src/geosave_flows`, `uv run pytest -q` — PASS.
- [ ] Commit `docs: catalog-centered ModelSpec flow`
