# STAC Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** One verb per meaning in the STAC code, before the nested-tree work builds on it.

**Architecture:** Mechanical renames and one move; behaviour unchanged. The existing tests are the safety net.

**Spec:** this plan; agreed in discussion on 2026-11-03.

## Global Constraints

- `build_<thing>`: construct a new STAC object or metadata value from parts; returns it.
- `stamp_<thing>`: attach information onto an xarray object; returns a new one (as `attrs.stamp_attrs`).
- `read_<thing>` / `write_<thing>`: get fields from, or set fields on, a STAC object.
- No aliases for old names. "describe" leaves every docstring and comment of the files touched.
- Edits stay uncommitted.

## Renames

| Module | Today | Becomes |
|---|---|---|
| `stac/source.py` | `StacSource.load` | `StacSource.load_dataset` |
| `stac/source.py` | `_describe_bands`, `_read_band_attrs` | `stac/translate/band.py`: `stamp_band_attrs(items, loaded, cfg=None)`, `read_shared_band_attrs` |
| `stac/source.py` | `_describe_whole_load` | `_build_load_metadata` |
| `stac/source.py` | `_record_date_properties` | `_stamp_date_properties` |
| `stac/translate/item.py` | `read_stac_metadata` | `build_stac_metadata` |
| `stac/translate/item.py` | `write_properties_coordinate` | `stamp_properties_coordinate` |
| `stac/translate/item.py` | `assign_time` | `stamp_item_time` |
| `stac/translate/item.py` | `group_observations` | `group_files_by_date` |
| `stac/translate/item.py` | `check_observation` | `check_single_date` |
| `stac/translate/item.py` | `select_observation_properties` | `select_item_properties` |
| `stac/extensions/{eo,raster}.py` | `from_dataarray` | `build_band` |
| `stac/extensions/classification.py` | `from_dataarray` | `build_classes` |
| `stac/extensions/{eo,raster,classification}.py` | `read_fields` (stamps a variable from band fields) | `stamp_fields` |
| `stac/translate/asset.py` | `from_raster` | `build_asset` |
| `stac/extensions/projection.py` | `from_geobox`, `to_geobox` | `build_fields`, `read_geobox` |
| `stac/translate/band.py` | `write_metadata` | `write_band_fields` |

## Steps

- [ ] **Step 1:** Baseline: `uv run pytest tests/geodata tests/model/spec/test_model.py tests/flows/test_ingest.py -q` (1157 passed).
- [ ] **Step 2:** Apply the renames to definitions and every caller in `src`, `tests`, `scripts`, `docs` (not `docs/contributing`); move the band-attr functions.
- [ ] **Step 3:** Rewrite "describe" wording in the touched files by what happens: build, read, or attach.
- [ ] **Step 4:** Same test command → same count; `uv run ruff check .`; `uv run basedpyright` on the touched files; `uv run python scripts/api_reference.py`.
