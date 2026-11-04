# Architecture

This page is for contributors. Start with the [practical guides](../index.md#find-your-task)
when learning to use GeoSave.

## Native objects and package boundaries

| Concern | Object or library |
| --- | --- |
| Raster data | xarray Dataset, DataArray, and DataTree |
| Vector data | GeoDataFrame |
| Model graph | PyTorch Module |
| Training | Lightning Module, DataModule, Trainer, and callbacks |
| Model publishing | Transformers and Hugging Face Hub |
| STAC | PySTAC, stac-geoparquet, rustac, and odc-stac |

GeoSave adds accessors and contracts around these objects. A separate wrapper
should own an invariant that the native object cannot represent cleanly.

```mermaid
flowchart LR
    ml --> model
    model --> geodata
    ml --> geodata
    geosave_flows --> geosave
```

| Package | Responsibility |
| --- | --- |
| `geodata` | Accessors, metadata, I/O, transforms, STAC, sampling, and visualization. |
| `model` | Model chains, registry, stages, specifications, and native releases. |
| `ml` | Reusable metrics, builders, losses, transforms, CLI, and callbacks. |
| `cli` | Workspace creation and task templates. |
| `templates` | Project-owned Lightning Modules, DataModules, Datasets, configs, and scripts copied into consumer workspaces. |

The model package does not import GeoSave's training package. Importing
`geosave` or `geosave.model.spec` does not load torch. Operations that need it,
such as `.gs.to_tensor()`, import it when called. Encoder libraries may load
additional dependencies when their modules are imported.

Prefect jobs live in `geosave_flows`, a second import package in the same
distribution, installed with `pip install "geosave[flows]"`. Those jobs compose
public GeoSave APIs; `geosave` imports neither `geosave_flows` nor Prefect.

## Data flow

```text
raster files -> item table -> samples -> trees -> chips -> Dataset samples -> Lightning -> release
```

Raster writers save pixels and return paths. STAC describes
those files in an item table. `sample` groups its Items into samples, the
row positions of each, without reading pixels; a sample's rows `to_datatree` open
it as a lazy tree, and `Chips` cuts
trees to the model's input size. A source-specific torch Dataset reads chips,
converts pixels to named tensors, and returns the task's sample contract. A DataModule batches and
prepares those samples; the Module owns training behavior.

`ModelSpec` owns raster requirements, preprocessing, input names, context, and
tensor transforms. Model stages own the network. A native release contains the
chain and optional spec; a Lightning checkpoint additionally holds training state.
ModelSpec v3 is being iterated through executable YAML processing recipes and
end-to-end workflow tests. Automatic scope inference and custom-code publication
remain open decisions. Training recipes live in templates, not `ml/tasks`.

## Implementation rules

- `core/raster/grid.py` owns spatial coordinates and grid metadata;
  `core/raster/time.py` owns reading and assigning temporal coverage coordinates.
  These describe existing observations without changing their pixels. Date
  parsing and calendar calculations stay in `utils/datetime.py`; transforms
  change observations or their sampling.
- `GeoAnchor.format` is the public location/time formatter used by STAC Item
  publication. An explicit `geobox=` supplies grid fields. STAC owns the Item's
  WGS84 union footprint, date selection and matching-grid validation.
- `io/odc.py` owns ODC band/alias resolution, grouping, fused property output and
  DataTree assembly. STAC modules interpret published metadata and determine
  agreement between sources; `attrs/stamp.py` applies attribute groups. The
  loader places Collection metadata on the Dataset root or collection node,
  and per-observation properties follow ODC's actual grouped time axis.
- Rasters are built with xarray's own constructors and placed with `write_grid`;
  `to_datatree` makes one a collection's grid node and `xr.merge` combines trees.
  There is no `core/ctor`: no GeoSave constructor builds an xarray or pandas
  object. Item tables come from `to_catalog`, `StacClient.search`, a stac-geoparquet
  file or `stac.from_items`, and plain vector tables from native
  GeoDataFrame construction or `open_vector`.
- `core/register.py` registers all xarray and pandas `.gs` accessors. Raster
  accessors extend their corresponding xarray types. Every GeoDataFrame uses
  GeoCatalog and every row GeoCatalogItem. Each accessor validates its own container. Registration does
  not load torch or visualization.

- `core/vector/catalog.py` owns GeoCatalog: place, drawing, query, writers,
  Item operations, and `sample` with its steps (distinct footprints, the Items
  of each footprint, temporal pairing and windows); an item table follows
  stac-geoparquet, one Item per row, and Item operations check their required
  columns when called. `item.py` owns GeoCatalogItem. `samples.py` owns
  GeoSamples, the item table plus each sample's row positions; samples may
  share rows, which a pandas GroupBy cannot hold. STAC encoding is
  selected at write time by `io/formats/geoparquet.write_geoparquet` from the frame's
  `stac-geoparquet` metadata. The same `to_geoparquet` method writes features;
  damaged catalogs raise instead of falling back to feature encoding.
- `viz` makes two products and shares what both need: `colors.py` (how
  values become colours) and `shapes.py` (drawable geometry by type).
  `viz/plot` draws static figures for `.gs.plot`, one module per data kind;
  `viz/explore` is the interactive map for `.gs.explore`. The two never import
  each other. Inside `explore`, Explorer composes the page; page components
  consume layers and comparison state, with Explorer imports used only for typing.
  `layers/` is the model: each layer is a Parameterized holding its
  data and live display settings, draws its viewport, and returns its legend
  as plain values; a `VectorLayer` of catalog rows owns the rasters its
  raster column opens. Pixels come from geodata's `.gs.read_pixels`.
  `explorer.py` owns the sources, the derived shown layers, the selection and
  the basemap. `page/` is the view, one module per region of the screen: it
  binds widgets to that state by reference, and keeps a class (`Map`,
  `AttributeTables`, `LayerDetails`) only where a part holds state of its own.
  Each table and custom color selector owns and releases its subscriptions;
  removed layers leave the page's table and style caches. An Explorer retains
  one page per session. Identify shares the raster eligibility used for drawing,
  including comparison and date selection, and uses unique layer labels.
  Catalog load generations prevent obsolete results from changing current
  selections; removal unchecks pending rows while retaining opened rasters.
  `mercator.py` holds the web-map extents and `server.py` serving.
- `core/anchor.py` describes geometry and time independently of grids. Raster
  accessors expose exact grids through `.gs.geobox`; `.gs.anchor` exposes coverage.
  Frames and trees union their footprints and envelope dated members. A point
  names a location, not a raster window.
- `stac/catalog.py` owns native Item/table conversion, including `from_items`.
  `stac/item.py`, `asset.py` and `band.py` own metadata for those STAC levels;
  extension wrappers are the only importers of `pystac.extensions`.
- `io/formats/` owns encoding-specific readers/writers. `io/rasterio.py` owns
  Rasterio reading and environment configuration. `_geopandas.py` prepares
  GeoPandas engine options; `_stores.py` checks NetCDF/Zarr destinations.
- `io/paths.py` expands time/band/group fields into destinations and lazy raster
  selections. `io/storage.py` owns UPath conversion, credentials and staging.
  GeoTIFF writing validates all destinations before executing writes.
- `io/stac.py` reads native Item assets and writes catalog assets. `write_assets`
  builds and validates Item metadata before writing pixels and returns an
  in-memory GeoDataFrame. `io/formats/stac_geoparquet.py` separately persists it,
  including Collection definitions and portable hrefs.
- `stac/search/` owns API/table discovery. `set_anchor` replaces spatial and time
  selectors. Selected Items feed `io/odc.load_dataset` or `load_datatree` without
  another search. ODC owns raster reading, grouping, mosaicking and resampling;
  GeoSave preserves CF axes, band attrs and fused per-time properties.
- DataTree loading preserves relative resolution families: the finest family
  uses the requested grid and coarser families use scaled grids. Family discovery
  currently reads the first Item. Native asset opening preserves original grids.
- Model recipes retain their serialized load settings in `model/spec/stac.py`,
  fill missing search selectors, bypass anchor filters for explicit IDs, and
  pass the client signer to the standalone loader. Their exact grid is explicit.
- `core/raster/chips.py` owns cutting and merging raster windows. Collection
  sampling and geographic anchors do not choose a pixel grid implicitly.
- Loaders own open resources and transfer cleanup to native xarray objects.
  `transform/combine.py` combines open rasters without taking their resources.
- Keep raster work lazy where the operation supports it. Pixel reads and writes
  must be clear at the public interface.
- Preserve applicable CRS, coordinates, transform, nodata, dtype, and variable
  identity. Decoding, casting, resampling, and coordinate changes are explicit.
  See [metadata behavior](../concepts/metadata.md) for current encoding limits.
- Constructors and raster accessors use spatial dimensions `y, x`. CF coordinate
  metadata distinguishes projected coordinates from latitude and longitude.
- Keep internal helpers with the package that uses them. Share an implementation
  only when written callers need the same behavior.
- Generated workspaces compose library APIs. Edit `src/templates/` to
  change future starters; a generated workspace belongs to its user.

## Errors, warnings, and maintenance

Use TypeError for unsupported types, ValueError for invalid values or
incompatible data, and KeyError for missing mapping entries. Preserve dependency
exceptions; attach useful context and re-raise. Prefect and Lightning own retry
and run presentation.

Warnings describe an operation that continues with an assumption, lost metadata,
or incomplete coverage. Categories live in `geodata.warnings` and derive from
GeoSaveWarning. The [error reference](../reference/errors.md) lists them.

Tests mirror `src/geosave/`. Persistence tests check round trips; lazy operations
check deferred execution as well as computed values. Open decisions and active design
proposals live beside this page: [open decisions](open-decisions.md) and
[design specs](specs/index.md).

### Explorer component ownership

`explore/explorer.py` owns layer membership, active selection, and temporal
state. Its public `period` and `freq` parameters stay on Explorer. The page
composes persistent child Viewers inside `pn.ui.Page`; `__panel__` returns their
existing components. LayerDetails uses named `@param.depends` display methods
and native Param references, detached on release without updating destroyed UI.

Each layer declares the parameters used by rendering, its legend, and its
summary. Map owns dynamic rendering subscriptions; LayerList owns navigation
subscriptions; LayerDetails owns each cached legend's subscriptions. There is
no global layer-change event. Map retains one DynamicMap and uses a native
HoloViews Counter to invalidate nested plot caches; coordinated date changes
batch into one redraw. AttributeTables reconciles membership separately from
selection and retains its table models.

`explore/comparison.py` contains comparison state; `page/comparison.py` builds
its controls. Explorer's `select_drawn_rasters()` remains the common eligibility
policy for drawing and Identify. `explore/server.py` owns the port registry and
server start/stop mechanics; Explorer owns independent session copies.

RasterLayer owns a byte-bounded LRU of viewport pixel windows, with a separate
active-window fast path for over-budget images. The 128 MiB default limits
retained pixel windows, not total process memory or rendering buffers. Pixel
reading remains in geodata I/O, including `read_scene_pixels`; no tile server,
global raster cache, or alternate reader was introduced.
