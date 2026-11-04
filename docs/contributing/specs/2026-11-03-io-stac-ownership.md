# I/O, STAC and GeoAnchor ownership

Status: implementation authorized. This document replaces the earlier iterative
sketches, including the rejected optional-GeoBox anchor and to_geobox method.
Execution and checks: [implementation plan](../plans/2026-11-03-io-stac-ownership.md).

## Public workflow

```python
from geosave.geodata import GeoAnchor, io
from geosave.geodata.stac import StacClient, StacQuery, from_items
from odc.geo.geobox import GeoBox

anchor = GeoAnchor.from_bbox((112.10, -8.10, 112.20, -8.00), timespan="2025-06")
client = StacClient.element84()
query = StacQuery(collections=["sentinel-2-l2a"])
query = query.set_anchor(anchor).set_filter("eo:cloud_cover <= 10")
items = client.search_items(query)
collection = client.collection("sentinel-2-l2a")

dataset = io.load_dataset(
    items, geopolygon=anchor.geometry, crs="EPSG:32749", resolution=10,
    bands=["red", "nir"], collection=collection, patch_url=client.patch_url,
    with_properties=["eo:cloud_cover"], chunks={"x": 1024, "y": 1024},
)

# A shared native grid makes repeated loads align exactly.
grid = GeoBox.from_geopolygon(anchor.geometry, crs="EPSG:32749", resolution=10)
aligned = io.load_dataset(items, geobox=grid, collection=collection)
tree = io.load_datatree(items, geobox=grid, collection=collection)
catalog = from_items(items, collections=[collection])
```

Native `pystac_client.Client.search(...).items()` works as loader input too.
Search and loading remain separate. Loading does not repeat a search, infer
an endpoint, fetch a Collection, or reapply an anchor's dates to selected Items.

## Ownership

```text
geodata/
  core/anchor.py         CRS-aware geometry and optional time
  io/
    formats/            geotiff, netcdf, zarr, geojson, geopackage, geoparquet,
                        stac_geoparquet, _geopandas, _stores
    rasterio.py         Rasterio reader and environment configuration
    paths.py            path expansion and lazy selections
    storage.py          UPath conversion, credentials, staging and transfer
    stac.py             native Item asset reads and write_assets
    odc.py              load_dataset/load_datatree and ODC metadata composition
    openers.py          suffix dispatch and resource ownership
    backend.py          xarray integration
  stac/
    catalog.py          native Item/table conversion
    item.py             Item identity, geometry, time and properties
    asset.py            one file's Asset metadata
    band.py             one variable's band metadata
    extensions/         PySTAC extension integration, one module per schema
    search/             query, HTTP client and independent Parquet table client
```

There are no compatibility namespaces for io.raster, io.vector or stac.translate.
Canonical low-level format imports use io.formats; modules are not also exposed
as io.geotiff/io.zarr. Typed read/write options remain exported through io.

`_geopandas` prepares GeoJSON/GeoPackage engine keywords. Rasterio's GDAL
environment does not configure the separate Pyogrio/Fiona binding. `_stores`
preflights NetCDF/Zarr suffixes, destination existence and encoding names without
computing pixels. No generic Store wrapper or backend hierarchy is introduced.

`io.stac.write_assets` plans destinations and builds/validates Item metadata
before writing COGs. `raster.gs.to_catalog` delegates to it and returns a native
GeoDataFrame. `catalog.gs.to_geoparquet` separately persists the table. Preserve
preflight checks without promising transactionality once actual writes begin.

STAC GeoParquet owns footer serialization, relative href persistence and
Collection extent preparation. Catalog conversion is shared by search, accessors,
writers and flows without importing a format implementation from core.

## GeoAnchor

```python
@dataclass(frozen=True)
class GeoAnchor:
    geometry: Geometry
    timespan: DateRange | None = None
```

Geometry is nonempty, valid, and carries a CRS. Points retain point identity;
polygons retain their shape. Constructors accept a geometry/frame, WGS84 bounds,
or latitude/longitude. They do not choose pixel resolution, dimensions or CRS
for an output grid. A point needs an area or explicit grid before raster loading.

- Dataset/DataArray anchor: grid footprint and temporal envelope, no pixel read.
- DataTree anchor: union of leaf footprints and envelope of dated leaves. No
  alignment or selection of an arbitrary finest/first grid.
- Item row anchor: WGS84 Item geometry and instant/interval, even without projection.
- Frame anchor: union in its active geometry CRS and envelope of dated rows.
  Timeless members, including GeoSave's STAC sentinel interval, do not widen it.
- Samples continue returning native frames; samples[i].gs.anchor follows exactly
  the same rules as selecting those rows directly. Existing sample footprint
  intersection policy remains separate.

A standalone Series loses its parent frame's CRS. Item row access requires
WGS84 geometry; use projected.iloc[[i]].gs.anchor for projected rows. Empty or
invalid geometry, unknown frame CRS, and incomplete date intervals raise errors.
Extracted anchors are snapshots, not cached state on mutable xarray/GeoPandas.

Exact grids stay native GeoBoxes on raster accessors or as explicit load inputs.
There is no GeoAnchor.geobox, resolution, set_grid or to_geobox. Its format method
retains geographic/time fields; resolution must be read from a separate grid.
Rasterization accepts a native GeoBox or raster reference.

## Discovery and loading

`StacQuery.set_anchor` returns a new query, replaces datetime and spatial
selectors with the anchor's WGS84 intersects, and clears bbox. A timeless anchor
clears datetime. Collections, IDs, filters, ordering and pagination remain.

StacClient retains provider factories, API sessions/retries, Collection
inspection, native Item searches and table searches. Its patch_url property
exposes the provider signer explicitly. Persisted Items keep durable URLs.
StacTable searches Parquet with rustac and does not inherit HTTP factories.
StacClient.open continues dispatching Parquet endpoints for model recipes.

`StacLoadOptions` is a TypedDict used through Unpack in both standalone loaders.
Native ODC options are forwarded; defaults remain solar_day and 1024-pixel spatial
chunks. Explicit chunks=None requests eager loading. GeoBox is mutually exclusive
with alternate grid selectors, as ODC validates. Geometry/CRS/resolution can yield
exactly the same grid as explicit GeoBox construction, but resolution alone does
not guarantee matching transform or shape. Geometry loading covers a rectangular
extent, not a polygon mask. ODC's anchor keyword controls pixel snapping.

Loaders consume Item iterables once and reject empty/mixed-collection selections.
An explicit Collection must agree with Item identities. Tree loading needs a
collection ID. No implicit metadata lookup occurs.

ODC owns reads, reprojection, mosaicking, grouping, property fusion and pixels.
GeoSave preserves CF axes and agreed source band metadata, keeping loaded nodata
and other pixel metadata authoritative. Published digital numbers are not decoded.
Requested properties retain exact STAC names and fused output-time values on
stac_properties(time); common properties stay in stac_common_properties attrs.
Existing duplication of requested constant properties is preserved.

DataTree leaves contain Datasets under collection/grid paths. The finest selected
family uses the requested grid; other families use grids scaled by source
resolution ratios. This preserves relative resolution, not guaranteed native
absolute resolution. Family discovery still uses the first Item; changing that
policy is outside this behavior-preserving extraction. Native Item opening keeps
original asset grids and remains a distinct operation.

## Consumer migration and excluded changes

StacSource is removed. ModelSpec's serializable load configuration now belongs
to model/spec/stac.py as StacLoadConfig. Model recipes retain endpoint fallback,
fill-missing selectors, explicit-ID bypass, default load settings and provider
signing. Recipe/model loading and ingest take geobox explicitly alongside the
anchor. Flow configuration schemas remain; open() returns (anchor, geobox).

No new model preprocessing/publication trust model, named-place boundary registry,
format support, STAC extension policy, or property placement policy is introduced.
GeoTIFF creation-option behavior is not silently changed by this ownership move.
The obsolete first-asset read_grid/has-grid anchor lookup is removed.

## Verification

Test native values and grids, laziness, fused properties, metadata round trips,
point/polygon identity, multi-grid and mixed-CRS extraction, real timeless Items,
provider signing and model search defaults. Compare broad failures against the
saved pre-refactor checkout. Never report timed-out or deselected checks as passes.
