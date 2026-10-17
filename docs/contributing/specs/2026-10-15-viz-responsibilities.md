# Visualization responsibilities

The accessor workflow stays native:

```python
scene.gs.plot("ndvi")                         # HoloViews figure, matplotlib
explorer = scene.gs.explore(overlay=fields)    # Panel viewer, Bokeh
catalog.gs.explore()                          # footprints, attributes, lazy opening
```

## Module ownership

| Module | Responsibility |
| --- | --- |
| `viz/plot.py` | Static raster figures and vector outlines |
| `viz/shapes.py` | Validate, group and summarize geometry for both renderers |
| `viz/map.py` | Web Mercator viewport, screen grid and basemaps |
| `viz/raster.py` | Raster overview caching, window reads and screen sampling |
| `viz/style.py` | Colour palettes, stretches and shared presentation defaults |
| `explorer/layers/` | Data-specific rendering, metadata, identification and selection |
| `explorer/docks/` | Widgets for one user task per module |
| `explorer/app.py` | Layer membership, selection, time, comparison and app layout |

`docks/style.py` owns controls and legends; `docks/inspection.py` owns metadata and click results. Their shared location in a tab strip does not define their responsibility. Search, tables, time and comparison remain separate because each owns a distinct interaction. Layer construction performs actual type dispatch; it is not an alternative data representation.

Functions name actions (`select_drawable`, `group_by_element`, `draw_raster`, `create_layer`, `create_element`, `get_info`). Properties name values (`bbox`, `summary`, `dates`). A GeoDataFrame is named `drawable_df` or `projected_df`; a HoloViews composition is a `plot` or a list of `elements`.

Callbacks name their effect: `_sync_layers`, `_sync_time`, `_draw_map`,
`_identify_layers`, `_select_layer`, and `_set_visibility`. Local names describe
their values, such as `selected_rasters`, `band_records`, and `dates_by_label`.
Properties use `draw_order`, `band_names`, and `opened_rasters`; overviews expose
`full_raster` and `coarse_raster`. Framework hooks and keyword names remain native.

Time selection and metadata bounds use chronological dates even when the source
axis is unsorted. Pixel-size labels use the source CRS units; the map's Web
Mercator display does not change the source pixel size into metres.

## Native property filters

Ordinary properties have no STAC-specific visualization rules. Panel Tabulator owns column filter parsing, comparisons, sorting and pagination. The table projects `current_view.index` into the layer's `filtered_rows`; both use original positional row indices. Drawing, highlighting and identification respect those rows. Selection remains separate, so filtering does not silently untick an opened raster.

Free-text search belongs to VectorLayer. CatalogLayer adds only STAC date-interval overlap and opening selected sources. Text/date predicates are attached using Tabulator's documented `add_filter(pn.bind(...))` interface. There is no cloud-cover constant, dedicated parameter or widget, and no second property-filter language.

## HoloViz documentation and runtime evidence

Reviewed official documentation on 2026-10-15:

- [HoloViz advanced dashboards](https://holoviz.org/tutorial/Advanced_Dashboards.html): `Viewer`, Param, HoloViews pipelines, `DynamicMap`, `.apply.opts()` and Panel layout are intended composition mechanisms.
- [hvPlot geographic data](https://hvplot.holoviz.org/en/docs/latest/user_guide/Geographic_Data.html): tile maps use Web Mercator; GeoPandas geometry types can be plotted directly, but mixed geometry requires separate handling. Spatialpandas is supported for large geometry rendering.
- [hvPlot resampling options](https://hvplot.holoviz.org/en/docs/latest/ref/plotting_options/resampling.html): numeric rasterization retains values for frontend colour mapping and hover; datashading returns RGB. Native rasterization manages viewport streams.
- [HoloViews events](https://holoviews.org/user_guide/Responding_to_Events.html): streams connect viewport and interaction events to callbacks.
- [Panel Tabulator](https://panel.holoviz.org/reference/widgets/Tabulator.html#filtering): column filters and reactive function filters are supported public interfaces.

Smoke tests confirmed that `hvplot.polygons(rasterize=True)` supplies numeric Image output, hover, a colourbar, PlotSize and RangeXY streams. Changing RangeXY changes the rendered extent. Tabulator header filters preserve original indices through remote pagination and selection.

The current explorer retains the lower-level HoloViews/Datashader rendering path, rather than claiming to be a plain hvPlot explorer. A live layer can switch between scalar quantities, classes and RGB composites. A smoke switching one DynamicMap between Image and RGB raises HoloViews' fixed-element-type assertion. Keeping RGB output preserves the live layer and its streams; the tradeoff is that numeric inspection and legends remain explicit panels rather than native image hover/colourbars. A regression exercises switching all three representations on one rendered DynamicMap.

Raster sampling remains GeoSave-owned: it preserves the existing lazy reprojection/window path, nodata handling and categorical resampling. A future replacement with HoloViz regrid must demonstrate those same contracts, including reads at small viewport scale. An overview computes its coarse representation on first use; it is not a guarantee of zero full-source I/O.

## Interface changes

The `.gs.plot()` and `.gs.explore()` entry points retain their signatures. Internal visualization function/method names now use verbs; direct callers and custom Layer subclasses must update. `Overview` lives in `viz.raster`; presentation constants live in `viz.style`. `FilterPanel` replaces the catalog-only search panel. `max_cloud_cover` is removed: use the table's ordinary `eo:cloud_cover` column filter.

Direct visualization callers must replace `Explorer.act` with `run_action`,
`Explorer.stacked` with `draw_order`, `RasterLayer.names` with `band_names`,
`CatalogLayer.opened` with `opened_rasters`, and `Overview.full`/`coarse` with
`full_raster`/`coarse_raster`. The `RasterLayer` input parameter is named `raster`.

## Remaining constraints

Dynamic map interaction requires a live Python process. RGB output does not expose numeric values through native image hover. Table row visibility belongs to its attached table; constructing multiple independent attribute-table docks for the same layer would introduce multiple visibility owners and is not a supported layout. No browser automation has verified the final client-side layout in this change.
