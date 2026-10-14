# viz on HoloViz's intended layers

Status: proposed, 2026-10-14.

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
| Explorer map | One persistent plot; each layer renders through `rasterize` / `shade` / `spread`, stacked by HoloViews' `stack` | Swapping an overlay per layer change works without errors but resets the zoom and leaves the old plot in the document; a DynamicMap with a varying overlay is refused |
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
├── style.py      # ColorRamp, Stretch, as_rgb, category_colors
├── plot.py       # static figures through hvPlot (matplotlib)
├── map.py        # Web Mercator: View, Overview, image helpers, basemaps, map colours
└── explorer/     # Panel app; layers render through HoloViews operations
```

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

Image helpers: `blank(view)`, `rgb_element(channels, alpha)`, `rgba(image)`.
`aggregate`, `spread`, `stack`, `composite`, `on_web`, `plain_columns`,
`Paint` and `_hoverable` are deleted. `tiles(name)` stays.

### Explorer layers

Each layer returns a HoloViews `RGB` for a view; the explorer stacks them.

```python
# RasterLayer
def image(self, view) -> hv.RGB | None:
    window = self.overview(...).view(view)                  # GDAL regrid of the selection
    rgb = as_rgb(window, self.variable)
    if rgb is None:
        return shade(hv.Image(window[self.variable[0]]), cmap=self.cmap,
                     clims=self.ramp.clim, cnorm="linear", alpha=..., dynamic=False)
    if len(self.variable) == 3:
        rgb = self.stretch.apply(rgb)
    return webmap.rgb_element(rgb, self.alpha)

# VectorLayer: count_cat with category_colors, mean with a ColorRamp, any() for one colour
# Explorer._draw
return stack(hv.Overlay([x.image(view) for x in shown]), dynamic=False)
```

The vector layer builds its spatialpandas shapes once, per geometry type,
with the row number and size columns it already carries. Swipe keeps cutting
two RGBA arrays at the split column.

## Explorer page

Layout, served and in a notebook alike:

```text
┌ sidebar ──────────┬ main ──────────────────────────────────────┐
│ Catalogs          │ basemap · theme toggle                      │
│ Layers            │                                             │
│ ───────────────── │                 map (fills the width)       │
│ Style | Info |    │                                             │
│ Identify          │ time bar · swipe                            │
│ (selected layer)  │ attribute table: search, header filters,    │
│                   │ rows per page, pager                        │
└───────────────────┴─────────────────────────────────────────────┘
```

The selected layer's tabs move from the right contextbar into the sidebar
under the layer list, as Planetary Computer and Copernicus Browser keep them,
so the map takes every column the sidebar does not. Served, it is a
`pmui.Page` with `sidebar` and `main` only; in a notebook, a `pmui.Row` of the
same two columns. The attribute table keeps today's search, header filters,
rows-per-page and pager.

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
- `stack`, `shade` and `spread` per layer per pan must stay as fast as the
  hand-written path; the browser smoke test times a pan on a large raster.
