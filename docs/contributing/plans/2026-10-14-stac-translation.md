# STAC Translation Implementation Plan

> Implementation authorized. User decisions: automatic export of recognized property coordinates; observation/composite scope; catalog-owned Collection definitions. Progress and rulings: [implementation ledger](2026-10-14-stac-progress.md).
> For agentic workers: use `superpowers:executing-plans` when implementation is requested. Steps use checkboxes for tracking.

**Goal:** Translate xarray metadata into native PySTAC objects through explicit schema ownership, preserving selected observation properties through scene writing.

**Architecture:** Coordinates hold observation values under their exact STAC property names. Scene carries those coordinates independently of RasterFile's actual stored header. Level translators compose native STAC objects; extension modules own schema operations; loading owns metadata precedence.

**Tech Stack:** xarray, PySTAC, odc-stac, existing raster writers, pytest.

**Spec:** The code-first design below records the proposal developed in the current conversation. Naming and new interfaces are proposed, not current APIs.

## Code-first design

### Native Collection and the two caller workflows

```python
collection: pystac.Collection = source.collection  # existing public property

# Catalog existing source assets without reading pixels.
catalog = source.search(anchor)
catalog.gs.to_geoparquet("source.parquet")

# Publish newly written rasters.
raster = source.load(anchor)
scenes = raster.gs.to_cog("prepared/{time:%Y%m%d}.tif")
prepared = gs.stac.create_collection("prepared", description="Prepared observations")
catalog = gs.stac.create_catalog(
    gs.stac.create_items(scenes, collection=prepared.id), collections=[prepared],
)
catalog.gs.to_geoparquet("prepared/catalog.parquet")
```

Keep `create_catalog`, `GeoVector`, `GeoCatalog` and `GeoSample`. A native GeoDataFrame gets a GeoCatalog accessor; one Series gets a GeoSample accessor. Item tables and sampling tables are distinct data contracts, not new wrapper classes. Only Item tables serialize as STAC GeoParquet.

`StacSource.collection` must remain a native PySTAC Collection for both API and table sources. Do not reconstruct it from the loaded Dataset's ACDD attrs. Publishing may copy it and update its extent to the written Items; preserve its descriptive fields, providers, license, summaries, extensions and extra fields without mutating the source object. Persistence owns structural link changes. Keep the current Collection-or-ID builder argument; document that Item creation uses identity and full Collection metadata is attached when creating the catalog, or explicitly supplied to persistence.

### Observation property workflow

```python
source = client.source("sentinel-2-l2a").set_config(
    groupby="time",
    with_properties=[
        {"key": "eo:cloud_cover", "units": "%"},
    ],
)
ds = source.load(anchor)

ds["eo:cloud_cover"]
# <xarray.DataArray 'eo:cloud_cover' (time: 2)>
# array([12., 65.])
# Attributes:
#     long_name:      Cloud cover over the source scene
#     units:          %

clear = ds.where(ds["eo:cloud_cover"] < 20, drop=True)
scenes = clear.gs.to_cog("out/{time:%Y%m%d}.tif")
items = list(gs.stac.create_items(scenes, collection="observations"))
assert items[0].properties["eo:cloud_cover"] == 12.0
```

`groupby="time"` does not guarantee one source Item per output observation: Items can share a timestamp. Property fusion must remain visible in the contract.

### Names and meaning

| Name | Meaning |
| --- | --- |
| `eo:cloud_cover` | Percentage of the source scene covered by clouds; a coordinate indexed by time, not a pixel mask. |
| `cloud_mask` | A spatial array identifying cloudy pixels. Do not use this name for a scene percentage. |
| `time` | Observation dimension/coordinate, never a raster band. |
| `long_name` | Human-readable description of the quantity. |
| `units` | Units of the actual coordinate values; `%` here, not a fraction. |
| `standard_name` | A registered CF standard name only when its definition matches. Do not put STAC keys here. |
| `Scene.coords` | Nonspatial xarray coordinates preserved by the write result; Item publication accepts one observation/composite. |
| `RasterFile.header` | Metadata read from the stored file, with its real grid, dtype and variable layout. |

Keep exact STAC property names, including their namespace: `ds["eo:cloud_cover"]`. Colons are valid xarray names; bracket access is ordinary xarray usage. Do not rename properties on loading, introduce alias mappings, or add a `stac_property` attribute. Reject a `with_properties` request that supplies a different output `name`; the remaining native options still apply.

Extension modules recognize the fields they own by exact coordinate name. Ordinary coordinates are ignored. A user rename changes that identity; there is no hidden alias to preserve. Do not invent CF `standard_name` values or duplicate coordinate values in Dataset attrs.

### Preserve coordinates independently of file headers

Proposed addition to the existing Scene record:

```python
@dataclass(frozen=True)
class Scene:
    collection: str | None
    files: Mapping[AssetKey, RasterFile] | None
    pending: Delayed | None = None
    coords: xr.Coordinates | None = None
```

Layout captures scalar and time-indexed nonspatial coordinates after selecting the scene's time and before splitting its variables into files. Exclude spatial coordinates and grid mappings. Preserve coordinate names, attrs, dtype and time alignment. Carry the same coordinates through immediate and deferred writing. Do not import STAC into raster layout to interpret them.

RasterFile remains a description of the stored file. GeoTIFF need not grow a private serialization scheme for arbitrary coordinates. `Scene.open` can recover only what files actually store; a TIFF alone cannot recover observation metadata discarded by its format. A persisted STAC Item is the metadata record for that case.

### Automatic Item-property translation

Decision: automatically export recognized property coordinates under their exact STAC names. Do not add an export-selection argument, mapping attribute, alias table or attrs group.

ODC already reads and groups Item properties. Do not add `translate/properties.py` or a second property reader. Item assembly calls each supported extension's `write_item(item, coords)` directly. Core property handling stays in `translate/item.py`. Extract a shared scalar helper only if the written implementations repeat it.

A recognized property must be scalar or have one element on `time`. Convert NumPy scalars to JSON scalars. Omit missing values (None, NaN, NaT); reject numeric infinity and invalid schema values. Ordinary coordinates and unsupported property names are ignored for export, while remaining available for processing. Do not guess schema URIs.

Initial Item-property export scope: EO cloud/snow cover and scalar core `platform`, `constellation`, `gsd`. Keep existing Projection, Raster, EO-band and Classification support. View and other Item extension schemas can be added later through their own modules; arbitrary Item JSON is not promised a lossless xarray round trip.

A source property still describes its source unless the user recomputes it. Automatic export does not recalculate cloud cover after cropping or mosaicking. Users can update or drop a coordinate using xarray; no hidden lineage framework is added. Add a regression asserting that a dropped coordinate is not exported.

### Temporal scope

Decision: implement observations and composites, not full multidate cube publication/loading.

- One Item row represents one observation or composite. `datetime` is scalar or null, never a list.
- Preserve nominal time and temporal bounds independently: STAC allows both. A composite's interval must survive even when it has one nominal time label.
- An input retaining more than one timestamp is still a time series. `create_items` refuses to publish it as one observation; tell the caller to split with a time template or explicitly produce a composite. Zarr/NetCDF persistence itself remains able to store the series.
- Do not fabricate one Item per timestamp pointing to a whole cube without a supported slice-selection contract.
- Keep existing timeless behavior in this change; do not redesign it as part of the refactor.
- A sampling-table row can contain many Items and dates. `GeoSample` retains temporal sample loading; that row is not itself a STAC Item.

`Scene` remains the writer's grouping of files for a cataloged product, including pending writes. Document that storage can represent more than the observation publisher supports. It is not a new pixel container or a generic inverse of every STAC Item.

### Schema ownership

```python
# extensions/eo.py — proposed interface; reads exact coordinate keys
def write_item(item: pystac.Item, coords: xr.Coordinates) -> None: ...

# translate/asset.py — composition sketch
def write(item: pystac.Item, key: str, file: RasterFile) -> None:
    asset = pystac.Asset(
        absolute_location(file.path),
        media_type=media_type(file.path),
        roles=["data"],
    )
    item.add_asset(key, asset)
    projection.write(asset, file.header.gs.geobox)
    band.write(asset, file.header)
```

These are sketches of ownership, not complete implementations. EO owns both its supported Item fields (cloud and snow cover) and spectral band metadata. Raster owns storage facts; Classification owns legend fields and their attachment; Projection owns grid fields. Use scope-qualified names where a schema supports multiple levels. Do not force all scopes into one base class.

Create the Item, attach each Asset, then use PySTAC extension methods so schema declarations are added at the write site. Remove `extensions.schemas` and its inference pass. Setter checks do not replace full Item/schema validation.

### Loading and reverse translation

Keep values from ODC's property loading and configured fuser. After promoting them to coordinates, keep each request's exact key as its coordinate name; do not accept a different output name. Use correct units and descriptions for recognized fields; reject incompatible units rather than relabeling values without conversion.

Do not promise exact source-Item reconstruction after grouping, mosaicking, cropping, resampling or reduction. For example, source cloud cover still describes the original footprint after a crop; exporting it as the cloud cover of a derived footprint requires a deliberate choice. By user decision, recognized coordinates export automatically. This is value propagation, not a claim that derived-product semantics have been validated.

Move `load_attrs` from `translate/__init__.py` to `load.py`. Read metadata directly from Assets and their band indices, then apply it to the real loaded Dataset. Preserve loader grid, dtype and nodata. Remove fake scalar RasterFile headers from this path. For chip loading, inspect the native Items and actual opened files; maintain the existing lazy windowing behavior while replacing the same fake-header dependency.

## Module ownership

| File | Responsibility |
| --- | --- |
| `stac/create.py` | Public builders and table uniqueness. |
| `stac/translate/item.py` | Item identity, time, footprint, assets and property composition. |
| `stac/translate/asset.py` | Native Asset construction and file identity. |
| `stac/translate/band.py` | Ordered band layout and schema composition. |
| `stac/translate/table.py` | Pure Item-table conversion shared by search and persistence. |
| `stac/translate/collection.py` | Selected ACDD translation and Item-derived extents. |
| `stac/extensions/*.py` | Schema-specific fields, native PySTAC operations and declarations. |
| `stac/load.py` | Loading integration, band resolution and metadata precedence. |
| `stac/source.py` | Collection/query/load configuration and public load entrypoint. |
| `io/raster/layout.py`, `file.py` | Scene coordinate preservation through file layout and writes. |

## Global constraints

- Preserve unrelated working-tree changes. No commits or staging as part of planning.
- Ignore notebooks. Keep native xarray and PySTAC objects; add no dependencies.
- Do not compute raster pixels to translate metadata. Reading small property coordinates is allowed.
- No compatibility aliases, extension discovery framework or duplicate value storage.
- Keep search transport restructuring outside this change.
- Preserve supported external STAC band layouts in one translation owner; do not silently change the output schema/version.

## Review focus

- Colon-containing coordinate names survive selection and supported store round trips (Task 1).
- Fused properties describe a grouped observation, not an exact input Item (Task 1).
- Deferred and immediate writes carry the same coordinates without retaining pixel arrays (Task 2).
- Multidate stores and derived spatial footprints must not silently inherit invalid scalar claims (Task 3).
- Source metadata must never overwrite the loaded grid, dtype or nodata (Task 5).

## Implementation sequence

### Task 0: Pin the native API and temporal contracts

Files: `stac/source.py`, `stac/translate/item.py`, `stac/create.py`, `stac/translate/collection.py`, `core/sample.py`, corresponding tests.

- [x] Test that API and table sources return a native Collection with summaries, providers, extensions and extra fields intact, and that `source.collection` does not search Items or load pixels.
- [x] Test both caller workflows above; verify the source Collection remains unchanged after publishing a subset.
- [x] Add tests for a single instant, a composite with nominal time plus bounds, an interval-only composite, a refused multidate store, and successful explicit per-date store splitting.
- [x] Preserve a composite's bounds instead of collapsing them to its nominal timestamp; reject unsupported multidate publication with a useful error. Treat the latter as a documented breaking change.
- [x] Add an asset-reordering regression: Item identity and footprint must be stable. Derive spatial ID fields from the Item footprint; require a common grid for grid-specific ID-template fields.
- [x] Decode `GeoSample` source Items once per load call, passing that local result to grid selection and loading. Do not cache across caller mutations.

### Task 1: Preserve exact property coordinate names

Files: `stac/source.py`, `stac/extensions/eo.py`, `tests/geodata/stac/test_source.py`.

- [x] Add behavioral tests for EO cloud cover under its exact STAC key, scalar time selection, missing values and two Items fused into one observation.
- [x] Run those tests and confirm a request for a different output name is currently accepted.
- [x] Use `PropertyLoadRequest.key` as the coordinate name and reject differing `name` values; keep the fuser's actual output. Add schema-owned quantity descriptions for supported EO properties.
- [x] Verify colon names survive NetCDF round trips, slicing selects the correct value, and reduction removes the time-varying coordinate.

### Task 2: Carry observation coordinates through Scene

Files: `io/raster/file.py`, `io/raster/layout.py`, `tests/geodata/io/raster/test_file.py`, `test_layout.py`, `tests/geodata/core/test_datatree.py`.

- [x] Add round-trip tests for two dates, per-band files, DataTree layers and deferred writes; assert Scene coordinates equal the selected input coordinates.
- [x] Run tests to establish current coordinate loss.
- [x] Add `Scene.coords`; capture coordinates in planned scenes and pass them through `save`, `defer` and `_write`, without changing RasterFile headers.
- [x] Verify a Dask callback observes no pixel computation during planning or Item construction from written scenes.

### Task 3: Translate recognized Item properties

Files: `stac/translate/item.py`, `stac/extensions/eo.py`, `tests/geodata/stac/translate/test_item.py`, `tests/geodata/stac/extensions/test_eo.py`, `tests/geodata/stac/test_create.py`.

- [x] Add tests asserting cloud cover 12.0 and EO declaration on the generated Item; test out-of-range values, missing values, unrelated coordinates and unresolved time axes.
- [x] Implement direct schema-owned Item writes from exact coordinate names; preserve native PySTAC setter checks. Do not add another property reader.
- [x] Test that arbitrary coordinates and spatial coordinates never become properties.
- [x] Document deliberate metadata reassessment for derived footprints; test that removing a source property coordinate prevents its publication.
- [x] Verify Items survive JSON/GeoParquet round trips with their selected property values and schema declarations.

### Task 4: Restore schema and level ownership

Files: `stac/extensions/*.py`, `stac/translate/asset.py`, new `stac/translate/band.py`, `stac/translate/item.py`, matching tests.

- [x] Add regression tests for classified assets, ordered multiband metadata, partial band descriptions and supported core `bands` input.
- [x] Move composition from the extension initializer into level translators; move field attachment and declaration into schema modules.
- [x] Construct Item owners before writing extensions and remove schema inference.
- [x] Use public dependency band-resolution capabilities where they match the supported layout; smoke-test installed PySTAC/ODC behavior before choosing emitted versions.
- [x] Verify schema validation separately from setter validation; do not claim unsupported schema coverage.

### Task 5: Remove metadata-only files from loading

Files: new `stac/load.py`, `stac/source.py`, `stac/translate/__init__.py`, `stac/translate/asset.py`, `stac/translate/item.py`, `core/sample.py`, `io/raster/assemble.py`, matching tests.

- [x] Add tests through both source and chip loading for aliases, packing, legends, source disagreements, nodata override and lazy window selection.
- [x] Read metadata by resolved native Asset/band identity and apply it to real loaded variables; keep consensus and precedence in the loading owner.
- [x] Remove fabricated RasterFile construction and its now-unused inverse APIs once all callers are migrated.
- [x] Verify actual file headers retain their grid/dtype invariant and pure STAC metadata inspection opens no files.

### Task 6: Separate Item-table translation from format I/O

Files: new `stac/translate/table.py`, `io/vector/stac_geoparquet.py`, `io/vector/stac_json.py`, `io/openers.py`, `stac/create.py`, `stac/source.py`, `stac/search/api.py`, `stac/search/table.py`, `core/catalog.py`, `core/sample.py`, matching tests.

Proposed in-memory conversion interface:

```python
# stac/translate/table.py
def from_items(items: Iterable[pystac.Item]) -> gpd.GeoDataFrame: ...
def to_items(rows: gpd.GeoDataFrame) -> list[pystac.Item]: ...
```

Move the current `from_item`, `to_item` and their Arrow conversion helper here. These functions translate representations without reading or writing a Parquet file. Both format writers, search results and accessors consume this one implementation. Use plural names because they convert collections of Items. Update callers without compatibility aliases; keep public `create_catalog` unchanged for this focused refactor.

Keep format-specific work in I/O:

| Function/class | Decision |
| --- | --- |
| `stac_geoparquet.write` | Keep: file encoding, relative hrefs and embedded Collection metadata. |
| `get_collections` | Rename to `read_collections`: it performs file I/O. Update its public export and callers. |
| `parse_collections` | Keep: decoding the format's JSON metadata is its responsibility. StacTable needs it for its existing DuckDB path. |
| `resolve` | Rename to `resolve_asset_hrefs`: resolve against the Parquet location; keep in the format module. |
| `stac_json.write` | Accept native Items; keep static hierarchy, JSON locations, relative links and persistence together. |
| `UPathStacIO` | Keep beside STAC JSON: it implements PySTAC's native I/O interface over project storage. |

`write` already matches the other format modules. Do not add generic wrappers just to manufacture symmetry. GeoParquet already has a reader; static Catalog JSON can be read with native PySTAC. `open_vector("catalog.json")` currently dispatches to GeoJSON, not static STAC traversal; document this distinction instead of treating every JSON file as a catalog.

- [x] Move in-memory conversion tests to `tests/geodata/stac/translate/test_table.py`; preserve native fields, input immutability, empty frames, row ordering and geometry-derived bbox behavior.
- [x] Move conversion ownership and update all callers, including lazy imports; verify ordinary GeoParquet reads still avoid importing STAC dependencies.
- [x] Remove the JSON module's import from the GeoParquet module and fix examples referring to nonexistent `stac.from_item` or old `create_item` APIs.
- [x] Change `stac_json.write` to accept `Iterable[pystac.Item]`; the catalog accessor converts rows once. Direct native-Item publishing must not pass through a GeoDataFrame.
- [x] Add one shared `validate_items(items: Sequence[pystac.Item]) -> None` in `stac/create.py` for duplicate `(collection_id, id)` checks. Both writers and `create_catalog` call it; pure table conversion remains permissive. Move the helper to a separate owning module only if actual imports require it, not as speculative infrastructure.
- [x] Preserve JSON/GeoParquet relative-href and Collection round-trip tests.
- [x] Add an overwrite regression: unrelated files and referenced assets inside the JSON output folder must survive. Replace the current recursive folder deletion with replacement of catalog-owned JSON only; refuse ambiguous ownership rather than deleting unrelated content.

### Task 7: Integrate and document

Files: `docs/guides/stac.md`, `docs/concepts/metadata.md`, affected reference pages and source docstrings.

- [x] Run `uv run pytest tests/geodata/stac tests/geodata/io/raster tests/geodata/core tests/geodata/cuts` with local socket access where required.
- [x] Run scoped Ruff, the repository's configured type checker on changed modules, and `git diff --check`; report actual scope and any pre-existing findings.
- [x] Run selected integration schema-validation tests separately where network/schema access is available.
- [x] Document exact coordinate naming, scope-specific extension operations, Scene/file distinction and non-lossless derived-product semantics.
- [x] Review breaking internal APIs and public Scene changes before implementation handoff is marked complete.

## Evidence before implementation

Earlier in this conversation: 128 STAC tests passed, 2 deselected with local socket access. Probes showed fabricated headers lose grid/dtype; core `bands` were ignored; iterable schema inference omitted Classification; GeoTIFF writing lost observation coordinates.

Latest decision: no property renaming and no extra identity attribute. Smoke-tested `eo:cloud_cover` through xarray selection and a NetCDF write/read: values [12.0, 65.0] survived. No production files were edited.

I/O review: 43 focused tests passed initially. The remaining schema-validation test failed on restricted DNS, then passed when rerun with network access: all 44 selected tests passed across those runs.

## Decisions sealed in the latest review

- `source.collection` is the native Collection API; no extra getter or metadata wrapper.
- Keep `create_catalog` and the three established pandas/GeoPandas accessor classes.
- Exact STAC names become coordinates. Recognized properties export automatically.
- Full multidate cubes are outside this implementation; observations and composites are supported.
- Remove the planned extra property reader. Use ODC for extraction and extension modules for schema writes.
- Preserve one explicit distinction: Item tables describe products; sampling tables describe model inputs.

The initial schema list and temporal scope are deliberately bounded. Expand only through a concrete schema or cube use case, not a generic plugin protocol.

## Collection ownership and stacks

Native definitions are serialized once in `DataFrame.attrs["stac:collections"]`.
`create_catalog(items, collections=[...])` attaches them; `source.search` does so
automatically. `catalog.gs.collections` returns native copies, and a GeoParquet
read restores the definitions. Writers require referenced definitions, copy them
with Item-derived extents, and preserve descriptive metadata. Explicit writer
definitions override attached ones by id. Unused definitions are not emitted.

`gs.GeoCatalog.concat([optical, labels])` combines rows and referenced definitions,
rejecting conflicting definitions by id. The result feeds `.gs.stacks(on="label")`.
Keep one table input for stacks. Multi-Collection publication stays supported;
one Collection per file is a convention, not a restriction.

Implementation and verification completed; see the linked ledger for evidence, review fixes, and existing adjacent check failures.
