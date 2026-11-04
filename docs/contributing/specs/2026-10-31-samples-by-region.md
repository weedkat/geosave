# Samples are item-frame rows grouped by region

Status: approved 2026-10-31; table shape superseded 2026-11-02 by
[sample rows as lists](../plans/2026-11-02-sample-rows-as-lists.md): one row
per sample, Item columns as lists, no `region` key.

## Intent

A sample table is the item table with one column added. Rows stay individual
STAC Items; a sample is the group of rows sharing one region. Grouping is
spatial only and uses pandas `groupby`, so per-sample questions are ordinary
pandas calls. This replaces the one-row-per-sample table whose columns held
Items, frame recipes and chip grids as JSON.

```python
items = gs.open_vector("catalog.parquet")      # GeoCatalog: one Item per row

samples = items.gs.sample()                     # region = each Item's normalized footprint
samples = items.gs.sample(regions)              # region = the polygon of each `regions` row an Item shares area with

samples.groupby("region").size()                # rows per sample, plain pandas
for region, rows in samples.groupby("region"):
    tree = rows.gs.to_xarray()                  # DataTree, one group per collection

samples.gs[0]                                   # rows of the first sample
len(samples.gs)                                 # number of samples
samples.gs.to_geoparquet("samples.parquet")     # stac-geoparquet; rows are still Items
```

```text
   id        collection  geometry (footprint)  region
0  s2_A_jan  s2          (0,0,10,10)           (0,0,10,10)
1  s2_A_feb  s2          (0,0,10,10)           (0,0,10,10)
2  s2_B_jan  s2          (10,0,20,10)          (10,0,20,10)
3  s1_swath  s1          (5,2,15,8)            (0,0,10,10)
3  s1_swath  s1          (5,2,15,8)            (10,0,20,10)    a row repeats per region it covers
4  label     label       (1,1,4,4)             (0,0,10,10)
```

```python
class GeoCatalog(GeoVector):
    COLUMNS = ("type", "stac_version", "id", "assets")

    def sample(self, regions: gpd.GeoDataFrame | None = None) -> GeoDataFrame[GeoSamples]:
        """Add each Item's region; an Item repeats once per region it shares area with."""


class GeoSamples(GeoCatalog):
    COLUMNS = (*GeoCatalog.COLUMNS, "region")

    def __getitem__(self, position: int) -> GeoDataFrame[GeoSamples]:
        """Return the rows of the sample at `position`, in order of first appearance."""

    def __len__(self) -> int:
        """Return the number of samples."""

    def to_xarray(self, *, chunks: ChunkSpec = {}) -> DataTree:
        """Open one sample's Items lazily, one group per collection."""
```

## Vector accessors

Tables take domain names, rows take pandas' word for them (`iterrows`), and
the base is named for the kind of data, as `GeoRaster` is.

```text
raster  GeoRaster (base)  ->  GeoDataArray   GeoDataset   GeoDataTree
vector  GeoVector (base)  ->  GeoFrame       GeoCatalog   GeoSamples
row                                          GeoCatalogRow  GeoSamplesRow
```

```python
plots.gs            # GeoFrame        any other GeoDataFrame: to_geojson / to_geopackage / to_geoparquet(**options)
items.gs            # GeoCatalog      stac-geoparquet, one Item per row: collections, to_items, check_unique, sample
samples.gs          # GeoSamples      a catalog plus `region`: len, [i], to_xarray
items.iloc[0].gs    # GeoCatalogRow   one Item: to_item, to_xarray, anchor
samples.iloc[0].gs  # GeoSamplesRow   the same, reading the row without `region`
```

- `GeoVector` holds what every vector shares: `crs`, `footprint`, `query`,
  `plot`, `explore`, `rasterize`, `concat`. It has no writers, so a catalog
  inherits none it must refuse; `to_geoparquet` takes no `**options`.
- `GeoCatalog` follows the stac-geoparquet spec
  (radiantearth/stac-geoparquet-spec). A frame is one when it has the spec's
  required columns `id`, `links`, `assets`, `stac_extensions` (`geometry` is
  the active geometry; `bbox` is the GeoParquet covering, dropped on read).
  `type` and `stac_version` are not required: the spec makes `type` optional
  and does not list `stac_version`; rustac fills both when building Items, so
  the reader no longer invents them. `collection` is optional too.
- Collection definitions live in `attrs["stac:collections"]` and in the
  file's `stac-geoparquet` metadata under `collections`.
- Ownership of Item <-> row conversion (`stac/translate/table.py` deleted):
  `create_catalog` builds rows from Items; `GeoCatalog.to_items` builds Items
  through rustac (stac-geoparquet's `stac_table_to_items` does not fill
  `type`, which pystac then rejects; verified); uniqueness is
  `GeoCatalog.check_unique` (collection, id), overridden by
  `GeoSamples.check_unique` (collection, id, region), so writers call one
  method. `GeoSamples.to_items` and `GeoSamplesRow` drop `region`, which
  places a row in a sample and is no Item field.
- A row's values carry no CRS, so `GeoSamplesRow` exposes no `region`; read
  regions through the table's `region` column.
- The reader makes asset hrefs absolute in the `assets` column itself, with
  no rows -> Items -> rows round trip.
- `GeoVector.query` holds the query rule; `core/vector/query.py` is deleted
  and the row time-span readers live in `utils/datetime.py`.
- Public `GeoItem` -> `GeoCatalogRow` and public `GeoVector` -> `GeoFrame`,
  with no aliases.

## Decisions

- `GeoCatalog` stays plain stac-geoparquet, one Item per row.
- Grouping keys on geometry only. The `region` geometry column is both the
  group key and the sample's extent; no separate id column.
- `regions=None` groups Items whose footprints are exactly equal.
- `regions` is any GeoDataFrame of polygons. An Item joins every region it
  shares area with (`sjoin(predicate="intersects")`, then drop pairs whose
  intersection area is 0, so a shared edge alone does not join).
- `sample()` normalizes each region polygon once. Grouping relies on exact
  equality afterwards; values are copied, never recomputed.
- `to_xarray` returns a DataTree with one group per collection, each stacked
  along `time` as `open_raster` stacks several Items. A collection of timeless
  Items has no `time` axis.
- `.gs` dispatches to `GeoSamples` when a `region` geometry column is present.
- `__getitem__` and `__len__` use `groupby("region", sort=False)`.

Builder's calls (mine, not yet confirmed):

- Group order is order of first appearance.
- `to_xarray` raises when the rows hold more than one region.
- `regions` is reprojected to the item table's WGS84 before joining; `region`
  is stored in WGS84 like the footprints.
- `StacMetadata` is folded into `GeoCatalog` (see Vector accessors).

- `to_xarray` cuts each raster to the sample's region. With `regions=None` the
  region is the shared footprint, so that is the whole raster.

- A sample-table row is identified by `(collection, id, region)`. Item ids stay
  the Items' own; only the uniqueness check widens. Persistence is
  stac-geoparquet with `region` as a second GeoParquet geometry column (WKB,
  listed in the `geo` metadata), appended after `stac_geoparquet.arrow.to_parquet`
  writes the Items. Verified: stac-geoparquet accepts repeated ids, and
  `gpd.read_parquet` restores both geometry columns with `geometry` active.

## Open before implementation

- Whether `attrs["stac:collections"]` survives `groupby` iteration (verified
  for pandas 2/geopandas 1.1: it does) and boolean selection.

## Non-goals

Frame, chip and merge are removed, not ported. They return later as operations
designed on this base. No compatibility path for old sample tables.

## Removal scope

- `geodata/sampling/` (sample, frame, chip, merge) and its tests.
- `core/vector/schema.py`; `GeoSample` row accessor; `Window`,
  `WindowFunction`; `GeoSamples.frame/chip/merge`;
  `io/vector/sample_geoparquet.py`.
- `model/spec/windows.py` (`FramesSpec`, `ChipsSpec`) and the `frames`/`chips`
  fields of `ModelSpec`; `ModelSpec.sample` calls `catalog.gs.sample()` only.
- The retired specs: frame-input-sequences, frame-records-and-derived-padding,
  sample-read-contract, sampling-windows. Their lessons are below.

Breaking: saved sample tables and model specs declaring `frames`/`chips` no
longer load.

## Retired mechanisms: what to carry over

The last implementation is in the local stash `40624b45` ("snapshot before
samples-by-region rewrite"), in
`src/geosave/geodata/sampling/` and `core/vector/samples.py`. The rules below
were learned from it and should hold in any new design.

### Frame (temporal selection)

- A slot such as `-2W` is a half-open period relative to a frame time:
  position and width from the unit (`s, min, h, D, W, M, Y`). Months and years
  use calendar arithmetic (`pd.DateOffset`), without snapping to month ends.
- Each slot is one output time step. Collections keep independent lengths and
  cadences in one frame; there is no global frequency.
- An observation matches a slot only when its whole coverage fits inside it
  (`start <= span.start` and `span.end < end`), so composites stay intact.
- Several observations in one slot require an explicit reducer. `first`/`last`
  pick a whole scene before opening files and keep its nodata; numeric
  reducers combine pixels when read.
- Read equal-timestamp Items separately: distinct acquisitions must each
  contribute to a composite.
- One Item can hold several raster times, so the reducer check must also run on
  the opened raster, not only on metadata.
- `fill="interpolate"` fills only wholly absent interior slots. Missing first or
  last slots make the frame incomplete; nodata in observed slots is never filled.
- Frame times come from a `reference` collection's acquisitions, else the union
  of requested collections' times, newest first. `stride` steps through
  candidates before completeness is checked; default is the largest slot count.
- An undeclared reference keeps its one matching observation (several raise);
  timeless collections stay attached to every frame; other unrequested dated
  collections are dropped.
- Incomplete frames are dropped with a `DroppedRowsWarning`, not raised.
- Output time labels are slot starts; bounds are inclusive, so the end is
  `slot_end - 1µs`.

### Chip (spatial windows)

- A grid comes from `target` (its lattice, not its extent) or `resolution`
  (`GeoBox.from_geopolygon(region, crs=..., anchor="edge", tol=1e-7)`); all
  sources in a sample must share one CRS when planning from a resolution.
- Snap the region to whole pixels on that lattice with
  `odc.geo.math.snap_grid(..., tol=1e-7)`; the tolerance absorbs WGS84
  round-off.
- Tiler lays chips; cache layouts by region shape. With overlap, add a halo
  (`tiler.calculate_padding()`) so every region pixel sits away from a chip
  edge at least once and a tapered merge leaves no seam at the border.
- Padding is derived, not stored: `grid.overlap_roi(layout, tol=1e-7)`. Pad on
  the opened raster's actual grid, since an explicit read target may differ
  from the planned one.
- Align observations to the chip grid before temporal composition, then crop to
  the read region and pad.
- Default resampling is nearest so class labels are never blended.

### Merge (chip outputs back to a raster)

- Each pixel is the weighted mean of valid chip outputs covering it; pixels no
  chip covers with positive weight are NaN. Accumulate in float64.
- Weights are the outer product of
  `scipy.signal.windows.get_window(name, n, fftbins=False)` per axis; None is
  all ones. Edge weights can be zero, so tapered windows need overlap.
- Overlaps average as logits do: take classes after merging.
- A chip matches its parent by spatial identity and time span, not by row order,
  so matching survives GeoParquet round trips and subsetting.
- Require the chip and layout resolutions to agree to `rtol=1e-7`; clip each chip
  to the layout with `overlap_roi`, and skip empty ROIs (their stops can be
  negative) after validating the chip's output.

### Mistakes not to repeat

- Persisting operation state as JSON strings in columns, decoded per property.
- Inferring a row's stage from which optional columns are filled.
- One parameter accepting several meanings (`target` as name, grid, raster or
  CRS) and duplicating `align` inside `to_xarray`.
