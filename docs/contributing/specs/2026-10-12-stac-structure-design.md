# STAC package structure

> Historical proposal. The extension/translation/search structure is implemented,
> with later API changes: `create_collection(id, description=...)` builds a
> Collection; singular `create_item` takes paths or `RasterFile` records;
> `StacClient.open` accepts providers, API endpoints, and table paths. ACDD
> combines fields separately. `StacTable` retains its DuckDB credential path.
> The signatures and breaking changes below describe the original proposal,
> not current APIs. See [STAC](../../guides/stac.md) and the
> [reference](../../reference/stac.md).

## Goal

`geodata/stac` is organized on two axes that match STAC itself: one module per
extension schema, and one module per STAC level (Collection > Item > Asset >
Band) translating that level to and from its xarray counterpart. Each fact has
one translation, in both directions. Every location is a UPath with typed
`storage_options`, so local and remote catalogs take the same code path.

```python
import geosave as gs

scene = gs.open_raster("s3://bucket/s2.zarr", storage_options={"anon": True})
forest = gs.stac.create_collection(scene.gs.acdd)            # ACDD -> Collection
rows = gs.stac.create_items(scene.gs.to_cog("s3://bucket/cogs/{time:%Y%m%d}.tif"), collection=forest)
rows.gs.to_stac_json("s3://bucket/stac", collections=[forest])     # remote static catalog
rows.gs.to_geoparquet("s3://bucket/stac.parquet", collections=[forest])

table = gs.stac.StacTable("s3://bucket/stac.parquet")
ds = table.source("forest").load(anchor)
ds.gs.acdd.title == forest.title                              # round trip
```

## Problems this replaces

- Band translation exists twice: `stac/_bands.py` (Raster, EO, Classification;
  STAC 1.0 and 1.1) and `stac/item.py::band_attrs` (EO only, 1.0 only). Chips
  read by `cuts/sample.py` use the second and lose packing, units and legends
  that `StacSource.load` keeps.
- `item._band_names` counts EO bands only, so `cog_files` misnames a STAC 1.1
  multi-band asset.
- ACDD has no owner: an ad-hoc dict in `_bands.load_attrs`, a raw `"summary"`
  lookup in `io/raster/tags.py`, nothing in the other direction.
- `stac/item.py` mixes item building, item reading and an unused `default_key`.
- `io/vector/stac_geoparquet.py` builds Collections (`describe`); `stac` imports
  that and the private `parse_collections` back.
- `StacClient.open` returns a private `_StacTableClient`.
- `StorageOptions = Mapping[str, object]` makes `UPath(path, **options)` fail
  type checking (no `__new__` overload accepts `protocol: object`) and accepts
  misspelled keys. `create_item`, `CogFile.href` and `stac_json.write` work on
  `str` or local `Path`.

## Layout

```text
geodata/
├── attrs/groups/acdd.py       # ACDD(AttrsGroup)
├── io/storage.py              # StorageOptions: union of upath's TypedDicts
├── io/vector/
│   ├── stac_geoparquet.py     # rows <-> one parquet file; rows <-> pystac Items
│   └── stac_json.py           # rows <-> static catalog folder, through UPath
└── stac/
    ├── __init__.py            # public surface
    ├── extensions/            # schema axis; the only importer of pystac.extensions
    │   ├── projection.py
    │   ├── raster.py
    │   ├── eo.py
    │   └── classification.py
    ├── translate/             # level axis; composes extensions
    │   ├── __init__.py        # load_attrs(items, collection, ds)
    │   ├── collection.py
    │   ├── item.py
    │   ├── asset.py
    │   └── band.py
    ├── search/
    │   ├── query.py           # StacQuery
    │   ├── api.py             # StacClient
    │   └── table.py           # StacTable
    ├── source.py              # StacSource
    └── create.py              # create_collection, create_item, create_items
```

Dependencies inside `stac` point one way:
`extensions <- translate <- {create, source}`; `search/api.py` and
`search/table.py` import `source` to build sources, and `source` imports only
`search/query.py`.
`io/vector/stac_*` imports `translate.collection.describe` lazily.
`cuts` imports `translate`, never `extensions`.

## Extensions: one module per schema

Stays at `stac/extensions/`, a sibling of `translate/`, because a schema is
not owned by one level: EO and View also state Item properties
(`eo:cloud_cover`, `view:sun_azimuth`) that `StacSource` loads as
`with_properties`. Content is unchanged. Each module keeps `write(asset, raster)` and `read(obj) -> dict` on PySTAC's
extension class.

| Schema | Written on | Read into |
|---|---|---|
| projection | Asset | grid (written only; the loader states the grid) |
| raster | Asset bands | `scale_factor`, `add_offset`, `units` |
| eo | Asset bands | `Spectral` attrs, `long_name` |
| classification | Raster bands | `Legend` attrs |

`extensions.ASSET` keeps its write order (projection, raster, eo,
classification).

## Translate: one module per STAC level

Every level module exposes `read` (STAC -> xarray) and `write`
(xarray -> STAC), the same names the extension modules use.

### `translate/collection.py`

```python
def read(collection: pystac.Collection) -> attrs.ACDD: ...
def write(acdd: attrs.ACDD) -> pystac.Collection: ...
def describe(
    items: list[pystac.Item], collections: Iterable[pystac.Collection] | None
) -> dict[str, pystac.Collection]: ...    # moved from io/vector/stac_geoparquet
```

Mapping:

| ACDD | Collection |
|---|---|
| `id` | `id` |
| `title` | `title` |
| `summary` | `description` |
| `keywords` (comma-separated) | `keywords` |
| `institution` | `providers[0].name` |
| `license` | `license` |

`write` raises `ValueError` when `id` or `summary` is missing (both required
by STAC). `license` defaults to `"other"`. The extent is global and unbounded
until `describe` sets it from the Items.

### `translate/item.py`

```python
def instant(item: Mapping[str, Any]) -> pd.Timestamp | None: ...   # moved from stac/item.py
def covered(rasters: Sequence[xr.Dataset]) -> tuple[DateTime, DateTime]: ...  # was _covered
def write(named: Mapping[str, tuple[str, xr.Dataset]], *, id, collection, when) -> pystac.Item: ...  # was _item
```

### `translate/asset.py`

```python
def bands(asset: pystac.Asset) -> list[tuple[RasterBand, Band]]: ...   # was _bands.read_bands; 1.0 + 1.1
def names(key: str, asset: pystac.Asset) -> tuple[str, ...]: ...       # was _band_names; counts bands(), not EO only
def files(item: pystac.Item) -> list[CogFile]: ...                     # was cog_files
def fields(asset: pystac.Asset, *, band: int = 1) -> dict[str, object]: ...  # was read_asset_fields
def write(item: pystac.Item, key: str, href: str, raster: xr.Dataset) -> None: ...  # add asset + extensions.ASSET
```

### `translate/band.py`

```python
def read(stored: RasterBand, spectral: Band) -> dict[str, Any]:
    return {**raster.read(stored), **eo.read(spectral), **classification.read(stored)}

def item_attrs(item: pystac.Item) -> dict[str, dict[str, Any]]: ...
    # replaces stac/item.py::band_attrs; variable name -> read(...) for every band
```

### `translate/__init__.py`

`load_attrs(items, collection, loaded, *, stac_cfg)` moves here from
`_bands.py` unchanged in behavior, except that the root attrs come from
`attrs.stamp_attrs(..., collection.read(collection))` instead of a literal dict.
`_bands.py` is deleted.

## ACDD attrs group

```python
class ACDD(AttrsGroup):
    """Describe a whole dataset as ACDD discovery attributes."""

    # joins by dropping a conflict: REFUSE_CONFLICT stays False

    id: str | None = None
    title: str | None = None
    summary: str | None = None
    keywords: str | None = None
    institution: str | None = None
    license: str | None = None
```

`gs.acdd` reads it from a Dataset's root attrs, `gs.write_acdd` stamps it, as
for the other groups. `io/raster/tags.py` reads `summary` through it.

## Creating items

`create.py` holds the public builders and the scene grouping
(`_records_items`); everything that touches a STAC object goes through
`translate`.

```python
def create_collection(acdd: attrs.ACDD) -> pystac.Collection: ...
def create_item(assets: Mapping[str, Location], *, id, collection=None, datetime=None,
                storage_options: StorageOptions | None = None, **options) -> GeoDataFrame: ...
def create_items(files: Iterable[CogFile], *, id=None, collection=None, datetime=None,
                 storage_options: StorageOptions | None = None, **options) -> GeoDataFrame: ...
```

`create_collection` takes one input, an `ACDD`. A caller without a raster
writes `gs.ACDD(id="forest", summary="Forest samples")`.

`default_key` is deleted (unused).

## Search

- `search/query.py`: `StacQuery`, unchanged.
- `search/api.py`: `StacClient` with `cdse`, `planetary_computer`,
  `element84` and `open(url | provider)`. `open` no longer accepts a Parquet
  path.
- `search/table.py`: `StacTable(path: Location, *, duckdb=None)`, public, the
  former `_StacTableClient`. Collections from file metadata come through
  `stac_geoparquet.get_collections`'s public parser, not `parse_collections`.

Both satisfy `SearchClient` (`search`, `search_items`, `collection`,
`collections`, `source`).

## Storage

```python
from upath.types.storage_options import (
    AzureStorageOptions, GCSStorageOptions, HfStorageOptions,
    HTTPStorageOptions, S3StorageOptions,
)

type StorageOptions = (
    S3StorageOptions | GCSStorageOptions | AzureStorageOptions
    | HTTPStorageOptions | HfStorageOptions
)
```

Smoke-tested with ty: `UPath(path, **(storage_options or {}))` type-checks,
and a misspelled key such as `{"anonn": True}` is an error. `Unpack` is not
used: it takes one TypedDict (one backend) and would mix storage keys such as
`timeout` with the function's own keywords.

Every STAC entry point that takes a location takes `Location` and
`storage_options`:

- `create_item` assets and `create_items` files are opened through
  `storage.location`.
- `stac_json.write(rows, path: Location, *, storage_options=None)` writes
  through a `pystac.StacIO` subclass backed by UPath, the extension point
  PySTAC provides, instead of `Path` and `shutil.rmtree`. It returns a UPath.
- `StacTable` keeps rustac's DuckDB session for credentials; see Open points.

## Public surface

```python
# geosave.geodata.stac
StacClient, StacTable, StacQuery, StacSource, StacSourceConfig, StacProvider
create_collection, create_item, create_items
from_item, get_collections      # re-exported from io.vector.stac_geoparquet
```

`cog_files`, `instant` and `band_attrs` leave the public surface; `cuts`
imports them from `translate`.

## Breaking changes

- `create_collection(id, *, description, ...)` becomes `create_collection(acdd)`.
- `StacClient.open("x.parquet")` becomes `StacTable("x.parquet")`.
- Module paths: `stac.client` -> `stac.search.api`, `stac.query` ->
  `stac.search.query`, `stac.item` -> `stac.create` / `stac.translate.*`,
  `stac._bands` -> `stac.translate`.
- `stac.cog_files`, `stac.instant` are no longer exported.
- `to_stac_json` returns a UPath.
- Callers to update: `cuts/sample.py`, `cuts/frames.py`, `cuts/stacks.py`,
  `model/spec/stac.py`, `geosave_flows/tasks/labels.py`, `workspace/modules/*`,
  templates' READMEs, `docs/guides/stac.md`, `docs/reference/stac.md`,
  `docs/quickstart.md`, `docs/guides/catalogs-and-cuts.md`,
  `docs/guides/model-spec.md`, `docs/basics/rasters.md`.

## Tests

Tests mirror the new tree: `tests/geodata/stac/{extensions,translate,search}/`,
`test_create.py`, `test_source.py`, `tests/geodata/attrs/groups/test_acdd.py`.
Existing tests move with their code; new ones:

- `translate/test_collection.py`: ACDD -> Collection -> ACDD round trip;
  `write` refuses a missing `id` or `summary`; `describe` sets the extent.
- `translate/test_band.py`: `item_attrs` on a STAC 1.1 `bands` asset returns
  packing, units, spectral and legend attrs (the bug `cuts/sample.py` had).
- `translate/test_asset.py`: `names` and `files` name `key.1..n` for a STAC
  1.1 multi-band asset.
- `cuts/test_sample.py`: a chip from a catalog with packing and a legend keeps
  both.
- `test_source.py`: `load(...).gs.acdd` equals `collection.read(source.collection)`.
- `io/raster/test_tags.py`: `summary` is still the GeoTIFF image description.
- `io/vector/test_stac_json.py`: round trip on a `memory://` UPath.
- `io/test_storage.py`: `location` keeps typed options on the UPath it returns.

## Open points

- **Remote `StacTable` credentials.** rustac reads through DuckDB, which uses
  DuckDB secrets, not fsspec options, so a remote table has its own
  credential path (`duckdb=`). Translating `S3StorageOptions` into a DuckDB
  secret is not part of this change; it is recorded in `open-decisions.md`.
