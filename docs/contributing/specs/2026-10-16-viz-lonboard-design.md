# Explorer map on Lonboard

Status: built 2026-10-16; compare flips between sides on one map (see Decisions). Supersedes the map half of
[viz on HoloViz's intended layers](2026-10-14-viz-holoviz-design.md); static
figures (`plot.py`) are unchanged.

## Goal

The explorer's UI and API stay: `.gs.explore()`, `*` composition, `Explorer`,
its docks and page. Its map engine is replaced, because the HoloViews
`DynamicMap` stack re-renders the whole screen on the server per pan
(~100 ms with a catalog, ~600 ms with a raster) and needs slot, stream and
plot-holding workarounds to stay alive in a notebook.

The map must stay smooth in local Jupyter, `panel serve`, remote JupyterHub,
VS Code and Colab, so nothing the map draws may depend on an HTTP port of our
own.

## Decisions

| Decision | Choice | Evidence (smoke tests, 2026-10-16) |
|---|---|---|
| Map engine | `lonboard.Map` (deck.gl through anywidget) | 100k polygons + raster pan at 60 fps; Bokeh `patches` drops to ~15 fps at 100k |
| Raster pixels to the browser | Lonboard `RasterLayer`: the browser asks for tiles, Python answers over the widget comm with encoded images | No server port, so every environment that shows a widget shows tiles; stale requests are cancelled by Lonboard |
| Raster in a COG | `RasterLayer.from_geotiff` on the variable's source file (async-geotiff, obstore) | Remote Sentinel-2: pans settle in 0.5–1 s, 15–40 ms render per tile |
| Raster not in a file (computed, in memory) | Web Mercator XYZ tiles read as xarray windows | 35–45 ms per tile in memory. Remote files read this way take 14–55 s per tile, so files never take this path |
| What a raster shows | Unchanged: a band with a `Legend` draws its classes, `.gs.rgb` draws a composite, else the first band; Style picks another | The rule already lives in attrs that `row.gs.to_xarray()` fills |
| Catalog Items | Ticking a row opens `row.gs.to_xarray()` (headers only) and draws it as any raster | One entry point; attrs come from STAC and the file |
| Vectors | `PolygonLayer` / `PathLayer` / `ScatterplotLayer` per geometry kind, from the frame's shown columns only | 200k Overture buildings open in 9.5 s and pan at 60 fps; the whole table would otherwise go to the browser |
| Large vectors | Above `VIEW_LIMIT` (250k) features, a layer sends only the features in view, re-queried through the frame's spatial index when the view settles | 798k buildings take ~55 s to open and 844 MB in the browser |
| Colour by column | `lonboard.colormap.apply_categorical_cmap` / `apply_continuous_cmap` | Lonboard's own API; colours update in the browser |
| Selection | A click identifies through the frame's spatial index (several rows may be hit); the selected rows draw as a small outlined layer on top | `selected_index` holds one row; clicks reached Python in every environment tested |
| Identify | `Map.on_click` gives a coordinate; each visible layer answers `identify` | Lonboard's own click hook |
| Picking | Tile layers (basemap, rasters) are `pickable=False` | deck.gl picks from only 255 sub-layers and every visible tile is one; with tiles pickable, clicks on 798k buildings reached nothing |
| Basemaps | Carto light and dark through `MaplibreBasemap`; OSM and Esri imagery as a bottom `BitmapTileLayer` the browser fetches | Both drawn in the smoke tests |
| Page, served | `pmui.Page` with the map in `pn.pane.IPyWidget`; the map gets its `view_state` at construction | Panel reads a widget's state while it is built; Lonboard sets `view_state` only after |
| Page, notebook | `ipywidgets.HBox([pn.ipywidget(sidebar), map])`, shown through `Explorer._repr_mimebundle_` | A Lonboard map nested in a Panel layout fails under Panel's `ipywidgets` and `vscode` comms ("model not found"); side by side works under all three |
| Compare | One map shows the left or the right side in place of the rasters; Left/Right flips them | `jslink` of `view_state` does not cross `ipywidgets_bokeh`, and a drawn lonboard map moves only through `fly_to`, so two linked maps would need a hand-made relay that lags each pan |

## Structure

```text
viz/
├── map.py        # WEB_MERCATOR, BASEMAPS, View (bbox ↔ view state)
├── style.py      # Stretch, ramps, category colours; to_image(values) -> EncodedImage
├── plot.py       # static figures, unchanged
└── explorer/
    ├── app.py    # Explorer: the lonboard Map, the stack order, clicks, the page
    ├── layers/
    │   ├── base.py     # Layer: settings as parameters, `deck` kept in step with them
    │   ├── raster.py   # RasterLayer: COG variables through from_geotiff, others as xarray windows
    │   ├── vector.py   # VectorLayer: one lonboard layer per geometry kind
    │   └── catalog.py  # CatalogLayer: footprints; ticked rows open as rasters
    └── docks/    # unchanged; they edit layer and Explorer parameters
```

`viz/raster.py` (`Overview`) and `map.create_tiles` are deleted; `viz/shapes.py`
stays for splitting frames by geometry kind, which `plot.py` shares. HoloViews,
hvPlot and datashader stay only where `plot.py` uses them.

### Layer

A layer keeps its settings as parameters, as today, and owns the Lonboard
layer the map draws. A setting changes the Lonboard layer's traits; the
browser redraws, the server renders nothing.

```python
class Layer(param.Parameterized):
    visible = param.Boolean(default=True)
    alpha = param.Number(default=0.8, bounds=(0.0, 1.0))

    @property
    @abstractmethod
    def deck(self) -> BaseLayer:
        """Return the lonboard layer the map draws."""

    @param.depends("visible", "alpha", watch=True)
    def _show(self) -> None:
        self.deck.visible = self.visible
        self.deck.opacity = self.alpha
```

`get_info`, `get_attrs`, `get_key`, `identify`, `summary` and `bbox` keep
their meaning. `render(view)` is gone.

### RasterLayer

```python
def _build_deck(self) -> lonboard.RasterLayer:
    names = self.band_names                       # the selection the rule picked
    sources = {self._raster[name].encoding.get("source") for name in names}
    if len(sources) == 1 and None not in sources:
        geotiff = open_geotiff(next(iter(sources)))
        return lonboard.RasterLayer.from_geotiff(geotiff, render_tile=self._render_file_tile,
                                                 pickable=False)
    return lonboard.RasterLayer(_tile_matrix_set=None, _crs=WEB_MERCATOR,
                                _fetch_tile=self._read_tile, _render_tile=to_image,
                                pickable=False, ...)
```

- `_render_file_tile` picks the selection's bands from the file tile, scales
  them as the variable's attrs say, and colours them with the layer's rule.
- `_read_tile(x, y, z)` reads the window under a Web Mercator tile from the
  raster in memory, in a worker thread, and colours it the same way.
- A different variable, colormap or date builds a new deck layer; the
  Explorer swaps it into `map.layers`. Opacity and visibility never rebuild.
- `open_geotiff(href)` lives in `io/storage.py`, which owns turning a
  location into a store: `obstore.store.from_url` with the location's
  `StorageOptions`, so signed and unsigned buckets read as `to_xarray` does.

Composites whose three bands sit in three files (Sentinel-2 bands on Earth
Search) take the xarray path in this version: correct, slower remotely.

### VectorLayer and CatalogLayer

- The frame is split by geometry kind once; each part becomes one Lonboard
  layer from `frame[[*shown_columns, geometry]]`. Lonboard reprojects its
  own copy to EPSG:4326; the frame stays as given.
- `column` colours through Lonboard's colormap helpers; `color` sets one
  fill. The search and table filters hide rows on the GPU through
  `DataFilterExtension` (`get_filter_value` is 1 for kept rows), so a filter
  sends one number per row, not the geometry again.
- A click identifies through the frame's spatial index, as today, and the
  selected rows draw as a small outlined layer on top. Hovering shows the
  shown columns through Lonboard's tooltip.
- Above `VIEW_LIMIT` features the layer holds only the rows in view: the
  Explorer passes each settled `view_state` and the layer replaces its
  table with `frame.sindex.query(view_box)` rows. A view still holding more
  than `VIEW_LIMIT` asks to zoom in.
- `CatalogLayer` draws its footprints the same way; ticking rows opens each
  as `RasterLayer(row.gs.to_xarray())`, unticking closes it, as today.

### Explorer

```python
self.map = lonboard.Map([self._basemap_layer, *(layer.deck for layer in self.draw_order)],
                        basemap=..., view_state=View.frame(bbox).view_state)
self.map.on_click(self._identify_layers)
self.map.observe(self._follow_view, "view_state")   # large vectors re-query
```

Adding, removing and reordering set `self.map.layers`; `zoom` calls
`self.map.fly_to`. Theme switches the Carto style and the category palette.

## Dependencies

Added to the core dependencies beside Panel: `lonboard[geotiff]` (with
async-geotiff, morecantile, arro3, geoarrow-rust), `obstore` and
`ipywidgets_bokeh`.

## Testing

- `tests/geodata/viz/explorer/test_layers.py`: each layer's `deck` is the
  Lonboard type expected; changing `alpha`/`visible` changes the deck's
  traits without rebuilding it; the raster rule picks classes, composite and
  first band; a COG-backed selection builds `from_geotiff`, an in-memory one
  the tile reader; `_read_tile` returns an image for a tile inside the
  raster and None outside; a large frame's table holds only rows in view.
- `test_app.py`: adding, removing and reordering change `map.layers` in
  order; a click identifies visible layers; ticking a catalog row adds a
  raster layer, unticking removes it.
- `tests/geodata/io/test_storage.py`: `open_geotiff` reads a local COG and
  a COG written to an in-memory store.
- Browser smoke (Playwright + Brave), served and in JupyterLab under default
  and `vscode` comms: a remote COG Item, a computed raster, 200k polygons,
  OSM and Esri basemaps; pan frame times, tile settle times, selection
  reaching Python, zero console errors, screenshots light and dark.

## Risks

- `RasterLayer`'s general constructor takes underscored arguments. Pin
  `lonboard>=0.17,<0.18` and ask upstream for a public one.
- VS Code and Colab were tested only through Panel's comm modes in
  JupyterLab.
- A raster computed lazily from remote files (an NDVI over a remote Item)
  takes the xarray path and reads remotely per tile, which is slow; it draws
  smoothly once computed or persisted.
- Remote reads through `to_xarray` need the COG GDAL settings
  (`configure_gdal`) to open in seconds rather than tens of seconds; that
  default belongs to the I/O package, not the map.
