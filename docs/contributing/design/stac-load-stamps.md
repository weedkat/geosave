# STAC metadata on loaded rasters: stamps and property coordinates

Status: approved 2026-11-13. Not implemented. Replaces the
`stac_properties` coordinate recorded in
[architecture](../architecture.md#raster-structure).

## Why

`stac.load_dataset` writes STAC metadata onto the loaded Dataset through a fixed
row of private helpers. The Dynamic World ingest found three problems:

| Problem | Today |
| --- | --- |
| A provider leaves metadata out | Planetary Computer states no scale/offset, only `s2:processing_baseline`. The caller writes packing after loading, past the mixed-scale check, so a mosaic of two baselines loads silently |
| The reader misses a STAC layout | CDSE states `raster:scale`/`raster:offset` on the asset (STAC 1.1). `band._read_fields` reads only per-band lists, so CDSE loads with no packing |
| Item properties are opaque | One dict per time step in `stac_properties`: not selectable (`ds.sel(time=cloud_cover < 20)` is impossible), no dtype, JSON text in files, and inconsistent with `start_datetime`/`end_datetime`, which already are coordinates |

There is no place to add "read this metadata this way": each helper does one
job, and `band_attrs` alone mixes three STAC schemas.

## Where STAC metadata lives

```text
<xarray.Dataset>
Dimensions:                  (time: 3, y: 510, x: 510)
Coordinates:
  * time                     (time) datetime64[ns]  2019-02-03T07:41 ...      the Item's datetime
    start_datetime           (time) datetime64[ns]  ...                       time cells
    end_datetime             (time) datetime64[ns]  ...
    eo:cloud_cover           (time) float64  0.4 12.1 3.0                     property, per step
    s2:processing_baseline   (time) <U5      '02.12' '02.12' '05.00'          property, per step
    platform                 <U11            'Sentinel-2B'                    property, shared
    month                    (time) int64    2 2 2                            not a property
    spatial_ref              int32           32737                            grid
Data variables:
    B04                      (time, y, x) uint16                              pixels only
Attributes:
    stac_collection: {...}   stac_extensions: [...]                           StacCollection
```

| What | Where |
| --- | --- |
| pixels | data variables, nothing else |
| band metadata (packing, spectrum, legend) | the band variable's attrs, as today |
| an Item property | a coordinate named by its exact key: along `time` when it differs per step, scalar when the whole load shares it |
| the Item's time and period | `time`, `start_datetime`, `end_datetime`, as today |
| the Collection and declared extensions | root attrs, `StacCollection`, as today |

### Marking a coordinate as a property

A coordinate is an Item property exactly when its attrs hold `stac_property`,
the property's name. A coordinate without it (`month`) is a working coordinate
and is never published.

```text
>>> ds["eo:cloud_cover"]
<xarray.DataArray 'eo:cloud_cover' (time: 3)>
array([ 0.4, 12.1,  3. ])
Attributes:
    stac_property:  eo:cloud_cover
```

The mark lives on the coordinate, not in a list on the Dataset, so it follows
the coordinate and goes when the coordinate goes. Checked with xarray:

| Operation | per-step property | shared property |
| --- | --- | --- |
| `isel`, `sel`, `ds["B04"]`, arithmetic, `where`, spatial `mean` | kept | kept |
| `concat` along time, `merge` | kept | kept |
| `mean("time")`, `resample(...).mean()` | dropped: one scene's value no longer describes the result | kept |
| netCDF and Zarr round trips | kept | kept |

`stac_property` is an attrs group, `attrs.StacProperty(name: str)`, beside
`StacCollection` in `attrs/groups/stac.py`.

### Accessor

```python
ds = ds.gs.write_stac_property("dw:annotation_source", "experts")         # scalar, shared
ds = ds.gs.write_stac_property("eo:cloud_cover", [0.4, 12.1, 3.0])        # along time
ds = ds.gs.write_stac_property("dw:month", ds["month"].values)            # publish a working coordinate
ds.gs.stac_properties
{'eo:cloud_cover': 'eo:cloud_cover', 'platform': 'platform', 'dw:month': 'dw:month'}
```

```python
def write_stac_property(
    self, name: str, values: Any, *, item_dim: str = TIME_COORDINATE
) -> Dataset:
    """Write an Item property as a coordinate named `name`, marked as a STAC property.

    A scalar applies to the whole raster; a sequence holds one value per Item
    along `item_dim`, the dimension `expand(item_dim)` makes Items of.

    Raises:
        KeyError: A sequence is given and the raster has no `item_dim`.
        ValueError: A sequence's length differs from `item_dim`, or a value
            is not a scalar (str, int, float, bool) or None.
    """

@property
def stac_properties(self) -> dict[str, Hashable]:
    """Return each STAC property name with the coordinate that holds it."""
```

`stac_properties` returns a mapping because STAC properties are open-ended
key-value content. It replaces the coordinate of the same name.

Only scalar property values become coordinates. List- or dict-valued
properties (`instruments`, `proj:centroid`, `processing:software`) stay in the
source Items. ODC cannot load them either.

## Loading: a pipeline of stamps

A **stamp** writes onto the loaded Dataset what the STAC documents state.
Stamps share one signature; `load_dataset` runs the named stamps in order. They
are chosen by name from a registry, as in `model/stages/registry.py`.

```python
# caller
scene = stac.load_dataset(items, grid, stamps=[*stac.DEFAULT_STAMPS, "sentinel_2"])
```

```python
# geodata/stac/stamps.py
@dataclass(frozen=True)
class LoadSource:
    """What a load read: its Items, their Collection and how they were grouped in time."""

    items: Sequence[pystac.Item]
    collection: pystac.Collection | None
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any]


type Stamp = Callable[[xr.Dataset, LoadSource], xr.Dataset]

STAMPS: dict[str, Stamp] = {}
DEFAULT_STAMPS = ("raster", "eo", "classification", "rgb", "title", "properties", "collection")


def register_stamp(name: str) -> Callable[[Stamp], Stamp]:
    """Make a stamp selectable by name in `load_dataset(stamps=...)`."""


def run_stamps(loaded: xr.Dataset, source: LoadSource, names: Sequence[str]) -> xr.Dataset:
    """Apply the named stamps in order; a later stamp overrides attrs an earlier one wrote."""
```

`LoadSource` is a typed value: a stamp needing more context gets a new field.

### One stamp per schema

| Stamp | Module | Writes | Refuses |
| --- | --- | --- | --- |
| `raster` | `stac/extensions/raster.py` | packing, nodata, unit | Items packing a band differently |
| `eo` | `stac/extensions/eo.py` | spectral attrs | |
| `classification` | `stac/extensions/classification.py` | flags and colours | |
| `rgb` | `stac/extensions/eo.py` | `Display.rgb` from red/green/blue common names | |
| `title` | `stac/stamps.py` | `long_name` from a one-band asset's title | |
| `properties` | `stac/stamps.py` | a scalar coordinate, via `write_stac_property`, for each scalar property every Item shares | |
| `collection` | `stac/stamps.py` | `StacCollection` attrs | |
| `sentinel_2` (opt-in) | `stac/extensions/sentinel_2.py` | packing from `s2:processing_baseline`, ESA: (DN + BOA_ADD_OFFSET) / 10000, offset -1000 from baseline 04.00 | Items of different baselines; an Item with no baseline |

Rules:

1. **A stamp reads only STAC, never another stamp's output.** `rgb` reads
   `eo:common_name` from the source bands, not the attrs `eo` wrote. Removing a
   stamp never breaks another; order matters only where two write the same attr.
2. **Shared reading is a function, not a precomputed table.** A band stamp asks
   for each Item's band fields:

```python
# stac/band.py
def read_band(item: pystac.Item, name: str) -> dict[str, Any]:
    """Return the STAC fields of the band ODC loads as `name` from this Item.

    Asset-level fields (STAC 1.1, e.g. CDSE's raster:scale) apply to every band
    of the asset; the band's own entry overrides them. Names resolve through
    odc.stac's parser, so `red` finds asset B04 as the loader did.
    """
```

### `load_dataset` after the change

```text
load_dataset(items, grid, groupby, stamps=DEFAULT_STAMPS, with_properties=None, mask_and_scale=False)
├── check items                one collection, at least one Item              fixed
├── odc.stac.load              pixels on the grid; requested properties       fixed
├── time axis                  drop for timeless Items; cells: the Item's     fixed
│                              period, else the solar day for solar_day
├── requested properties       ODC's variable → coordinate named by the exact fixed, only with
│                              key, via write_stac_property                    with_properties
├── stamps, in order           each reads LoadSource only
└── mask_and_scale             native CF decoding, after the stamps           fixed, optional
```

What comes from a load option is a fixed step; what the STAC documents state
is a stamp.

`with_properties` stays: ODC loads and fuses a requested property per time
step. Measured on two Planetary Computer tiles fused into one solar day:

```text
groupby="id"                         groupby="solar_day"
eo:cloud_cover  24.001  0.712        eo:cloud_cover  12.36          mean of the finite numbers
s2:mgrs_tile    '47NMC' '47NMB'      s2:mgrs_tile    '47NMC,47NMB'  strings joined
instruments (a list)                 ValueError in ODC
```

Requests default to float64 for numbers (ODC's float32 turns 24.001125 into
24.0) and object for strings; a request may still state its own `dtype` and
`fuser`.

## Publication

`StacItem.from_datatree` and `to_catalog` build an Item's properties from the
coordinates in `ds.gs.stac_properties`, taking each one's value at that Item's
time step, and its time from `time`, `start_datetime` and `end_datetime`.
Nothing else becomes a property.

## Today's code → after

| Today | After |
| --- | --- |
| `stamp_band_attrs`, `band.read_attrs`, `band._read_fields`, `band.read_shared_attrs` | stamps `raster`, `eo`, `classification`, `rgb`, `title`; `band.read_band` |
| `_check_source_scales` (before loading) | inside `raster`, after loading |
| `_stamp_date_properties` | stamp `properties` (shared) + fixed step (requested) |
| `build_stac_metadata` + `attrs.stamp_attrs` | stamp `collection` |
| — | stamp `sentinel_2` |
| `_read_item_cells`, `_write_solar_day_cells`, timeless drop | fixed time-axis step |
| `_check_items` | fixed input check |
| `_prepare_options` | unsupported-option check, chunk default, property dtypes |
| `stac_cfg` option | removed: unused; filling missing metadata is a stamp's job |
| `STAC_PROPERTIES` coordinate (`conventions.py`) | removed: per-key coordinates |
| `stac/raster.py` JSON encode/decode of `stac_properties` | removed: scalar coordinates are native in netCDF and Zarr |
| `_place_asset` (in `load_datatree`) | Item node holds that Item's property coordinates |
| `.gs.composite()` rebuilding `stac_properties` | keeps, per bin, the properties every member scene shares, via `write_stac_property` |
| `publish.StacItem.from_datatree` | reads `stac_properties` mapping |

## Goals, non-goals

Goals: one mechanism for metadata from STAC; provider gaps fixed by an opt-in
stamp with its own consistency check; CDSE packing read without a stamp; Item
properties as typed, selectable coordinates, with one explicit rule for what
gets published.

Non-goals: stamps that change pixels or the grid (those are operations); a
stamp per provider for what the provider already states; callables passed
directly to `load_dataset` (names only: `list[str | Stamp]` would be a union
serving two cases, and names work from YAML); list- or dict-valued properties
on rasters.

## Decided

- Stamps take `(loaded, LoadSource)`; `LoadSource` grows by fields.
- Names only; `DEFAULT_STAMPS` is a tuple callers extend.
- Later stamps override earlier ones' attrs.
- Each stamp checks its own consistency.
- Built-in stamps register when `geosave.geodata.stac.stamps` is imported,
  which imports the extension modules defining them; a project stamp registers
  in its own module, imported before the load.
- An unknown stamp name raises `KeyError` listing the registered names, before loading.
- `load_datatree` takes `stamps` and forwards it.
- `geosave.geodata.stac` exports `DEFAULT_STAMPS`, `LoadSource` and `register_stamp`.
- A coordinate is a STAC property exactly when it carries `stac_property`.
- Only scalar property values become coordinates.

## Left to the builder

- `properties` writes only what every Item of the load shares. Whether it
  should also write, per time step, what that step's Items share (solar-day
  mosaics with differing scenes) needs the step → Items mapping; carry it on
  `LoadSource` if wanted, verified against `odc.stac.load(groupby="solar_day")`.
- `write_stac_property` on DataArray as well as Dataset, if publication from a
  DataArray needs it.

## Work order

1. Cleanup: move the load helpers into stamps and fixed steps; delete
   `stac_cfg`, `STAC_PROPERTIES` and its JSON encoding, the old helpers; no
   compatibility aliases.
2. `attrs.StacProperty`, `write_stac_property`, `stac_properties`;
   publication and `composite` on them.
3. `band.read_band` with asset-level `raster:*` inheritance.
4. `sentinel_2` stamp.
5. Tests per stamp and per accessor, rewriting `tests/geodata/stac/test_load.py`
   (27 cases) and the `stac_properties` tests (14 files in tests and docs).
   New cases: CDSE-shaped asset-level packing; ESA mixed-baseline refusal;
   unknown stamp name; the coordinate-survival table above; netCDF/Zarr round
   trip of property coordinates; publication of marked coordinates only.
6. Docs: `architecture.md` raster structure, `concepts/metadata.md`,
   `guides/stac.md`, `conventions.py` diagrams.
7. Dynamic World ingest drops `read_esa_packing` and passes
   `stamps=[*DEFAULT_STAMPS, "sentinel_2"]`.
