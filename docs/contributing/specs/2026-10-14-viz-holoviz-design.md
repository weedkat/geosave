# viz on HoloViz's intended layers

The current module ownership, property filtering and interface names are recorded in [Visualization responsibilities](2026-10-15-viz-responsibilities.md).

Status: implemented 2026-10-14; the explorer map half is superseded by [Explorer map on Lonboard](2026-10-16-viz-lonboard-design.md). Static figures still follow this spec.

## Goal

`geodata/viz` should read as geosave composing HoloViz, not as a parallel
plotting stack. Every drawing step uses the HoloViz layer meant for it
(hvPlot, then HoloViews operations, then datashader), and every data step uses
geosave's own accessors (`.gs.to_nan`, `.gs.reproject`, `.gs.colorize`,
`.gs.rgb`, `.gs.legend`). What stays hand-written is what neither owns: the
explorer's layer stack, swipe, time bar, attribute table and identify.

Public calls do not change: `.gs.plot()`, `.gs.explore()`, `Explorer`,
`RasterLayer`, `VectorLayer`, `CatalogLayer`.

## Decisions

| Decision | Choice | Evidence (probes, 2026-10-14) |
|---|---|---|
| High-level API | Add `hvplot` to the core dependencies, beside holoviews and panel | `.hvplot.image`, `.hvplot.rgb(bands="band")` draw rasters on bokeh and matplotlib |
| GeoViews | Not added | hvPlot on a geopandas frame imports geoviews; on a spatialpandas frame it does not. Data reaches Web Mercator through `.gs.reproject` and `to_crs` |
| Vectors | Hand hvPlot/HoloViews a `spatialpandas.GeoDataFrame` | Polygons, lines, points and Multi* draw and rasterize |
| Mixed geometry frames | Split once by `geom_type`, one element per part | hvPlot refuses a mixed frame |
| Text column colouring | `aggregator=ds.count_cat(column)` + `shade(color_key=...)` | `rasterize` refuses non-numeric `c`; `ds.by(col, ds.any())` breaks `shade` |
| Explorer map | Each layer is a `DynamicMap` drawing one RGB per view part, its opacity and visibility linked with `.apply.opts`; the map overlays them on the basemap and is laid out again only when the stack changes, opening on the view it had | Opacity and visibility update in about 30 ms without a redraw; a rebuilt map keeps its ranges through `xlim`/`ylim`, and Panel leaves one plot behind it |
| Compare | `pn.Swipe` of two map panes, axes linked by Panel | Panes of one document link their axes by default; no pixels are cut by hand |
| Catalog search | `CatalogLayer` parameters (text, dates, cloud cover) filter the drawn footprints and, through Tabulator's `add_filter`, the table | One `filter_items` serves the map and the table |
| Class rasters | Drawn as RGB through `.gs.colorize()`, named by a legend of swatches | Classes are colour already; a `{code: colour}` cmap on matplotlib misaligns colours, so a third colormap path buys nothing |
| Dated figures | A `HoloMap` of one hvPlot element per date | `hvplot.rgb(groupby="time")` refuses a `(time, band, y, x)` array |

## Two ways a raster draws

HoloViews draws a raster two ways: `Image`, one band through a colormap, and
`RGB`, three colour channels. geosave keeps exactly those two. A band whose
`Legend` lists classes is colour already, so `.gs.colorize()` turns it into
channels and it draws like a composite; its names come from the `Legend`,
never copied elsewhere.

```python
def as_rgb(raster: xr.Dataset, variables: tuple[str, ...]) -> xr.DataArray | None:
    """Return the colour channels a selection draws as, None for one band a colour ramp draws."""
    if len(variables) == 3:
        return raster[list(variables)].gs.to_array()      # a composite
    legend = raster[variables[0]].gs.legend
    if legend is not None and legend.classes:
        return raster[variables[0]].gs.colorize()         # classes in their legend's colours
    return None
```

Figures and the map both call `as_rgb`, so a selection draws the same way on
both; the map calls it on the window it regridded, so class codes are never
blended before they are coloured.

## Structure

```text
viz/
├── style.py      # Stretch, category_colors
├── shapes.py     # drawable geometry split by the element drawing it: Polygons, Path, Points
├── plot.py       # static figures through hvPlot (matplotlib)
├── map.py        # Web Mercator: View, Overview, basemaps, map colours
└── explorer/
    ├── app.py    # Explorer: the stack, the map or swipe, the page
    ├── layers/   # base (Layer, Key), raster, vector, catalog; one module per kind
    └── docks/    # base (Dock), layers, search, compare, tabs, time, table
```

A layer is state (parameters) plus `draw(view)`, a pure function returning an
RGB, wrapped by `Layer.on_view` in a `DynamicMap` fed by `RangeXY`, `PlotSize`
and its own parameters. A dock is a panel placed by `region` (`"sidebar"`,
`"tabs"`, `"main"`); a tab naming an explorer parameter in `raised_by` comes
forward when it changes, as Identify does on a click.

`elements.py` is deleted. Map colours (`COLOR`, `HIGHLIGHT`, `OUTLINE`) and
`CMAPS` live in `map.py`, the only module that defines how a map looks; the
explorer's `THEME` and `CUT` stay in `app.py`, which alone uses them.

### `style.py`

`Style` and `Paint` are replaced by two small values named as a GIS user
knows them, a function per remaining decision, and nothing that repeats the
`Legend`:

```python
@dataclass(frozen=True)
class ColorRamp:
    """A colormap between two limits: how one band or numeric column draws."""
    cmap: str | Sequence[str] = DEFAULT_CMAP
    clim: tuple[float, float] | None = None
    def fit(self, values: np.ndarray) -> ColorRamp: ...   # limits from the finite range, set ones kept
    @property
    def palette(self) -> list[str]: ...                   # 256 hex colours, for a legend's ramp


@dataclass(frozen=True)
class Stretch:
    """The limits a composite's channels stretch between onto [0, 1]."""
    clim: tuple[float, float] | None = None
    def fit(self, values: np.ndarray) -> Stretch: ...     # uint8 as stored, else 2-98 percentile
    def apply(self, channels): ...                        # ndarray or DataArray, nan kept


def category_colors(values: pd.Series, theme: Theme = "light") -> dict[object, str]:
    """Colour each category of a text column from the theme's glasbey palette, in sorted order."""
```

Nodata leaves colour entirely: a map reads rasters through `.gs.to_nan()`, so
nodata is nan before anything is coloured. A figure keeps drawing values as
stored, nodata included, as it does today.

### `plot.py`

`figure` takes the raster and the variables it draws, so the accessors pass
through what the caller named:

```python
def figure(raster, variables, *, cmap=None, clim=None, cols=4, title=None, xlabel=None):
    rgb = as_rgb(raster, variables)
    if rgb is None:
        ramp = ColorRamp(cmap or DEFAULT_CMAP, clim)
        frames = _by_date(raster[variables[0]], lambda band: band.hvplot.image(
            x="x", y="y", cmap=ramp.cmap, clim=ramp.clim, colorbar=True, data_aspect=1))
    else:
        if legend is None:
            rgb = Stretch(clim).fit(rgb.values).apply(rgb)  # one stretch for every date
        frames = _by_date(rgb, lambda channels: channels.hvplot.rgb(
            x="x", y="y", bands="band", data_aspect=1))
        if legend is not None:
            frames = frames * _swatches(legend)             # named classes beside the map
    return _captioned(frames, raster, cols=cols, title=title, xlabel=xlabel)
```

A colormap is refused for a composite or a class band, and limits for a class
band, as today. `outline` draws `split_by_geometry(frame)` part by part with
`spd.GeoDataFrame(part).hvplot()`. Caption and layout code is unchanged.

### `map.py`

Keeps `View` (ranges and screen size; `View.canvas` gives the keywords that
put a datashader operation on its grid) and `Overview` (zoomed-out copy plus
full-detail windows), because multiscale reading of a lazy raster is ours.
`Overview` reads the selected variables as a Dataset through geosave:

```python
values = raster.gs.to_nan().gs.write_nodata(float("nan"))
self.full = values.gs.reproject(WEB_MERCATOR, resampling=kernels)   # refuses blending classes
self.coarse = values.gs.reproject(self.full.gs.geobox.zoom_to(size), resampling=kernels).compute()
```

A `View` is an odc-geo `BoundingBox` in Web Mercator with a screen size;
`View.framing(bbox)` opens on a box with a margin. No image helpers remain:
`hv.RGB` leaves nan clear itself, and the browser lays images over each other.

### Explorer layers

Each layer draws itself; the explorer only overlays what layers return.

```python
# RasterLayer: one RGB per view, a band through shade, classes colorized, a composite stretched
def element(self, time=None) -> hv.DynamicMap:
    drawn = self.on_view(self.draw, "variable", "time", "cmap")
    return drawn.apply.opts(alpha=self.param.alpha, visible=self.param.visible)

# VectorLayer: one DynamicMap per element (fills, edges and lines, points), then the selection
# Explorer._draw
self._pane.object = self._framed(reduce(operator.mul, [x.element() for x in self.stacked], self._basemap))
```

A text column with one category draws as one colour: datashader colours
categories against the least counted one, so a lone category would vanish.

## Explorer page

Layout, served and in a notebook alike:

```text
┌ sidebar ──────────┬ main ──────────────────────────────────────┐
│ Catalogs          │ basemap · theme toggle                      │
│ Search Items      │                                             │
│ Layers            │       map, or a swipe between two rasters   │
│ Compare           │                                             │
│ ───────────────── │ time player                                 │
│ Style | Info |    │ attribute table: header filters, pager      │
│ Identify          │                                             │
└───────────────────┴─────────────────────────────────────────────┘
```

The selected layer's tabs move from the right contextbar into the sidebar
under the layer list, as Planetary Computer and Copernicus Browser keep them,
so the map takes every column the sidebar does not. Served, it is a
`pmui.Page` with `sidebar` and `main` only; in a notebook, a `pmui.Row` of the
same two columns. The attribute table filters by column headers and pages
through Tabulator's own pager; a catalog's table also follows its search.

### Light and dark

- The page follows panel-material-ui's theme: `Page`'s toggle when served,
  the JupyterLab or VS Code theme in a notebook. The explorer reads it from
  the components' `dark_theme` parameter.
- Map colours that sit on the basemap switch with the theme: the default
  basemap is `CartoLight` or `CartoDark`, and text categories use colorcet's
  `glasbey_dark` on light and `glasbey_light` on dark, so every category
  stays visible against its background. A basemap the user picked is kept.
- Text never carries a hard-coded colour: legends, facts and identify
  results are pmui components (`Typography`, `List`), not inline HTML, and
  take the theme's text colour. A swatch is the only coloured element.
- Tables use Tabulator's `materialize` theme on light and `midnight` on dark,
  switched with the theme.

### Raster Info tab

Modelled on Planetary Computer's item panel and Copernicus Browser's layer
info, built from the attr groups geosave already parses, nothing read twice:

| Section | Source |
|---|---|
| Overview: title, summary, institution, license, keywords | `ACDD` on the raster |
| Grid: CRS, size, pixel size, extent in degrees | `.gs.crs_name`, `.gs.geobox` |
| Time: dates, first and last, step | `.gs.times`, `TimeSpec` |
| Bands: one row per variable with dtype, nodata, units, common name, centre wavelength, FWHM, scale and offset | `Nodata`, `Quantity`, `Spectral`, `Packing` |
| Classes: code, name, swatch for a band with a legend | `Legend` |
| All attrs: the raw attrs, grouped and searchable, collapsed by default | today's table |

A section with nothing to show is left out. Vector and catalog layers keep
today's facts (CRS, rows, geometry, columns, dates).

## Testing

- `tests/geodata/viz/test_style.py`: `ColorRamp.fit`, `Stretch.fit`/`apply`,
  `category_colors` per theme, `as_rgb` for a band, a composite and classes.
- `test_plot.py`: band, classes (colorized, legend swatches), RGB, `time` layout, mixed
  frame outline; existing figure assertions kept.
- `test_map.py`: `Overview` stays lazy until a view reads it (a dask graph
  check); nodata reads nan; classes refuse blending through `.gs.reproject`.
- `test_elements.py` is deleted; its geometry-split cases move to `test_plot.py`.
- `explorer/test_layers.py`: each layer's `image` returns an RGB on the
  view's grid; selection and identify unchanged.
- `explorer/test_docks.py`: the raster Info tab lists each section its attrs
  fill and leaves out empty ones; the table still searches and filters.
- Browser smoke test (Playwright + Brave), once light and once dark: open a
  raster, a vector with a text column and a catalog; pan, zoom, add, remove,
  reorder, swipe, time step, search the table; zero console errors, the zoom
  kept across each, and screenshots checked for legible text and categories
  in both themes.

## Risks

- `hvplot` is a new core dependency.
- datashader's categorical colouring blends overlapping categories by count,
  where today the highest category wins; overlapping polygons may tint.
- `.gs.reproject` warns about a band with no fill value; after `.gs.to_nan`
  the overview passes nan explicitly so a map raises no warning.
- Replacing the map's plot (a reorder, an added or removed layer, compare on
  or off) can leave Bokeh's `ResizeObserver` on the removed figure: when the
  page resizes in the same moment, the browser logs `Cannot read properties
  of undefined (reading 'top_panel')` from `FigureView._measure_layout`. The
  live map is unaffected; pan, zoom and style changes never replace the plot.
- panel-material-ui's `MenuList` sends a row action but leaves its menu open
  until a click outside or Escape.
