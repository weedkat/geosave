# Attrs Groups Implementation Plan

> **For agentic workers:** executed inline with superpowers:executing-plans; ledger at `.superpowers/sdd/2026-10-10-attrs-groups/progress.md`.

**Goal:** Replace the attrs model/header/namespace/registry/rebase machinery with a few validated attr groups read through `gs` properties and written through `gs.write_*`, joins through xarray's `combine_attrs`, and CF attrs as the only in-memory metadata.

**Design (settled in conversation 2026-10-10):**
- `attrs/base.py`: `AttrsGroup` Protocol (`KEYS`, `JOIN`, pydantic `model_validate`/`model_dump`), `Join`, `CONFIG`, `Decoded[T]`, `parse(band, group)`, `stamp(band, group, value)`.
- Groups, one pydantic class per file, aliases via `AliasChoices`/`serialization_alias`, band checks via validation `context={"dtype": ...}`: `Nodata` (`_FillValue`, reads `nodata` too, writes `_FillValue` only), `Packing`, `Legend` (classes ↔ `flag_values`/`flag_meanings`, `color_map`), `Spectral`, `GDALBand` (`variable_name`, `colorinterp`; plotting RGB and GeoTIFF band identity).
- `attrs/combine.py`: `combine(attrs_list, context=None)`, the `combine_attrs` callable: each group in `GROUPS` parsed per object and joined by its `JOIN` (`agree` raises `ValueError`, `drop` leaves it out); plain keys as xarray's `drop_conflicts`.
- `gs` accessors: one line per property / `write_*`; Dataset versions take `target=`.
- Unmanaged keys stay plain attrs: CF descriptors, ACDD. GeoTIFF dataset tags move private to `io/raster/geotiff`. `StackedAttrs` removed (to_array keeps root attrs only). `TimeSpec` replaced by a CF `time_bnds` bounds coordinate written by `resample`.
- Deleted: `AttrsModel`, `field_keys`, `MUST_AGREE`, scope registry, `AttrsNamespace`, `AttrsHeader`, `AttrsEdit`, `create_header`, `rebase`, `attrs.merge`, `common_attrs`, `gs.attrs`, `gs.rebase`, header modules (moved to their boundaries).

**Constraints:** no commits (working tree shared); don't edit notebooks; workflow (Prefect) code may only change where it imports removed names; `geodata` imports no torch.

## Tasks

1. New `attrs` core: base + five groups + combine, with tests (`tests/geodata/attrs/`), beside the old package.
2. Accessors: `core/array.py`, `core/raster.py`, `core/base.py`, `core/stack.py` on the new groups; remove `gs.attrs` / `gs.rebase`.
3. Transforms: nodata, packing, warp, color, concat, merge, composite (joins via `combine`).
4. IO: GDAL reader returns plain attrs; GeoTIFF writer/tags private; Zarr.
5. STAC: extensions and source read/write groups; STAC header code moves to `stac/source.py`.
6. Remaining consumers: `model/spec/rasters.py`, `utils/statistics.py`, viz.
7. `TimeSpec` → `time_bnds`: `resample`, `fill_gaps`, `gs.timespan`, `stac._item` time cover.
8. Delete the old package and its tests; docs; full suite and ruff green.
