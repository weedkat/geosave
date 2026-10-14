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
| High-level API | Add `hvplot` to the viz dependencies | `.hvplot.image`, `.hvplot.rgb(bands="band")` draw rasters on bokeh and matplotlib |
| GeoViews | Not added | hvPlot on a geopandas frame imports geoviews; on a spatialpandas frame it does not. Data reaches Web Mercator through `.gs.reproject` and `to_crs` |
| Vectors | Hand hvPlot/HoloViews a `spatialpandas.GeoDataFrame` | Polygons, lines, points and Multi* draw and rasterize |
| Mixed geometry frames | Split once by `geom_type`, one element per part | hvPlot refuses a mixed frame |
| Text column colouring | `aggregator=ds.count_cat(column)` + `shade(color_key=...)` | `rasterize` refuses non-numeric `c`; `ds.by(col, ds.any())` breaks `shade` |
| Explorer map | One persistent plot; each layer renders through `rasterize` / `shade` / `spread`, stacked by HoloViews' `stack` | Swapping an overlay per layer change works without errors but resets the zoom and leaves the old plot in the document; a DynamicMap with a varying overlay is refused |
| Class figures | Keep class positions + named `cbar_ticks` | A `{code: colour}` cmap on matplotlib misaligns colours and labels the colorbar numerically |

## Structure

```text
viz/
├── style.py      # Style: how values become colour; also how a vector column does
├── plot.py       # static figures through hvPlot (matplotlib)
├── map.py        # Web Mercator: View, Overview, basemaps, map colours
└── explorer/     # Panel app; layers render through HoloViews operations
```

`elements.py` is deleted. Map colours (`COLOR`, `HIGHLIGHT`, `OUTLINE`) and
`CMAPS` move into `map.py`, the only module that defines how a map looks; the
explorer's `THEME` and `CUT` stay in `app.py`, which alone uses them.

### `style.py`

`Style` stays the one rule from values to colour, shared by figures and maps.
It loses the pixel work HoloViz does (`rgba`, `stretch`) and gains what
`Paint` did, so a vector column and a raster band use one model:

```python
@dataclass(frozen=True)
class Style:
    kind: Kind                       # "continuous" | "classes" | "rgb"
    cmap: str | Sequence[str] = DEFAULT_CMAP
    clim: tuple[float, float] | None = None
    classes: Mapping[object, str] | None = None   # code or category -> name
    colors: Mapping[object, str] | None = None    # code or category -> #rrggbb

    @classmethod
    def of(cls, array, *, cmap=None, clim=None, legend=None) -> Style: ...
    @classmethod
    def of_column(cls, values: pd.Series, cmap=DEFAULT_CMAP) -> Style:
        """Categories by name with glasbey colours, quantities on a ramp."""
    def resolve(self, values) -> Style: ...        # limits from pixels, unchanged
    def positions(self, values) -> np.ndarray: ...  # class figures only
```

`nodata` leaves `Style`: rasters reach a figure or map through `.gs.to_nan()`,
so nodata is already nan. `Paint`, `Paint.numbers` and `_plain` are deleted.

### `plot.py`

```python
def figure(array, style, *, cols=4, title=None, xlabel=None):
    array = array.gs.to_nan()
    if style.kind == "rgb":
        drawn = _stretched(array, style).hvplot.rgb(x="x", y="y", bands="band", groupby=TIME, ...)
    else:
        drawn = array.squeeze("band", drop=True).hvplot.image(x="x", y="y", groupby=TIME, **_options(style))
    return _captioned(drawn, array, cols=cols, title=title, xlabel=xlabel)

def outline(frame, *, color, alpha):
    return hv.Overlay([
        spd.GeoDataFrame(part).hvplot(color=color, alpha=alpha, data_aspect=1)
        for _, part in _by_geometry(frame)
    ])
```

`_options` maps a Style onto hvPlot keywords (`cmap`, `clim`, `colorbar`;
classes add `clim=(-0.5, n-0.5)`, `color_levels`, `cbar_ticks`). Caption and
layout code is unchanged. `_by_geometry` is the one place geometry types are
split, a `groupby` on `geom_type` with the `Multi` prefix dropped, and is
shared with the explorer.

### `map.py`

Keeps `View` (ranges and screen size) and `Overview` (zoomed-out copy plus
full-detail windows), because multiscale reading of a lazy raster is ours.
`Overview` is built on geosave rather than odc directly:

```python
values = array.gs.to_nan()
self.full = values.gs.reproject(WEB_MERCATOR, resampling=kernel)   # refuses blending classes
self.coarse = values.gs.reproject(self.full.gs.geobox.zoom_to(size), resampling=kernel).compute()
```

`View.canvas`, `aggregate`, `spread`, `stack`, `composite`, `on_web`,
`plain_columns` and `_hoverable` are deleted. `tiles(name)` stays.

### Explorer layers

Each layer returns a HoloViews `RGB` for a view; the explorer stacks them.

```python
# RasterLayer
def image(self, view) -> hv.RGB | None:
    window = self.overview(...).view(view)                  # GDAL regrid, unchanged
    if style.kind == "classes":
        return hv.RGB(window.gs.colorize(), ...)            # the band's own legend
    if style.kind == "rgb":
        return hv.RGB(stretched(window, style), ...)
    return shade(hv.Image(window), cmap=self.cmap, clims=style.clim,
                 cnorm="linear", alpha=round(self.alpha * 255), dynamic=False)

# VectorLayer
def image(self, view) -> hv.RGB | None:
    agg = rasterize(self._elements, aggregator=self._aggregator, dynamic=False, **view.ranges)
    shaded = shade(agg, color_key=style.colors) if categorical else shade(agg, cmap=..., clims=...)
    ...                                                     # boundaries, dots: spread(..., px=2)

# Explorer._draw
return stack(hv.Overlay([x.image(view) for x in shown]), dynamic=False)
```

The vector layer builds its spatialpandas elements once, per geometry type,
with the row number and size columns it already carries. Swipe keeps cutting
two RGBA arrays at the split column.

## Testing

- `tests/geodata/viz/test_style.py`: `Style.of_column` for categories and
  quantities; `positions`; `resolve`. `rgba`/`stretch` tests go.
- `test_plot.py`: band, classes (named ticks), RGB, `time` layout, mixed
  frame outline; existing figure assertions kept.
- `test_map.py`: `Overview` stays lazy until a view reads it (a dask graph
  check); nodata reads nan; classes refuse blending through `.gs.reproject`.
- `test_elements.py` is deleted; its geometry-split cases move to `test_plot.py`.
- `explorer/test_layers.py`: each layer's `image` returns an RGB on the
  view's grid; selection and identify unchanged.
- Browser smoke test (Playwright + Brave): open a raster, a vector and a
  catalog; pan, zoom, add, remove, reorder, swipe, time step; zero console
  errors and the zoom kept across each.

## Risks

- `hvplot` is a new dependency of the viz extra.
- datashader's categorical colouring blends overlapping categories by count,
  where today the highest category wins; overlapping polygons may tint.
- `.gs.colorize` on a map window needs the band's `Legend` to survive
  `.gs.reproject` and `.gs.to_nan`; the first plan step checks it, and a
  band without one falls back to the colours `Style` already holds.
- `.gs.reproject` warns about a band with no fill value; after `.gs.to_nan`
  the overview passes nan explicitly so a map raises no warning.
- `stack`, `shade` and `spread` per layer per pan must stay as fast as the
  hand-written path; the browser smoke test times a pan on a large raster.
