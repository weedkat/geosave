# viz on HoloViz's Intended Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `geodata/viz` on hvPlot, HoloViews' datashader operations and geosave's own accessors, then give the explorer a left sidebar, light and dark themes, and a raster Info tab.

**Architecture:** `Style` stays the one colour rule; figures draw through hvPlot on matplotlib; each explorer layer renders a HoloViews `RGB` for the view on screen with `rasterize`/`shade`/`spread`, and the explorer stacks them with `stack` into one persistent bokeh plot. Rasters reach Web Mercator through `.gs.to_nan()` and `.gs.reproject`; class rasters colour through `.gs.colorize()`.

**Tech Stack:** hvPlot 0.12, HoloViews 1.23, datashader 0.19, spatialpandas 0.6, Panel 1.9 + panel-material-ui 0.16, xarray, geopandas, pytest, Playwright with Brave for browser checks.

**Spec:** `docs/contributing/specs/2026-10-14-viz-holoviz-design.md`

## Global Constraints

- No GeoViews or cartopy: vectors reach hvPlot/HoloViews as `spatialpandas.GeoDataFrame`, rasters through `.gs.reproject`.
- `hvplot` joins the core `dependencies` in `pyproject.toml`, beside `holoviews`.
- No vague constructor names: `Style.from_band`, `Style.from_column`; nothing named `of`.
- Public calls keep their signatures except `basemap`, whose default becomes `None` (follow the theme): `.gs.plot()`, `.gs.explore()`, `Explorer`, `RasterLayer`, `VectorLayer`, `CatalogLayer`.
- Figures draw values as stored, nodata included; only the map reads through `.gs.to_nan()`.
- A lazy raster stays lazy: the explorer reads only a coarse copy and the windows a view shows.
- The explorer keeps one bokeh plot for its life (`explorer._pane.object` is never replaced).
- Text carries no hard-coded colour; a swatch is the only coloured element.
- Comments say what a chunk does, not how; docstrings Google style, concise.
- Do not touch unrelated working-tree changes; commit only the files each task lists.

## Review Focus

- A text column holding missing values: features with no category draw nothing, the rest draw in their category's colour (Task 3, `test_missing_categories_draw_nothing`).
- A lazy raster opened in the explorer is not computed whole (Task 3 keeps `test_reads_lazily_beyond_its_coarse_copy`).
- Switching theme after the user picked a basemap keeps their basemap (Task 4, `test_theme_keeps_a_picked_basemap`).
- A raster carrying no ACDD, spectral or legend attrs shows only the sections it fills (Task 6, `test_leaves_out_empty_sections`).
- A swipe that splits at 1.0 shows the left side whole, not flipped (Task 3, `test_swipe_keeps_orientation`).

---

### Task 1: `Style.from_band`, `Style.from_column`, nodata off `Style`

**Files:**
- Modify: `pyproject.toml` (dependencies)
- Modify: `src/geosave/geodata/viz/style.py`
- Modify: `src/geosave/geodata/viz/map.py` (`Overview.__init__` only)
- Modify: `src/geosave/geodata/viz/explorer/layers.py` (`RasterLayer._style` only)
- Modify: `src/geosave/geodata/core/dataset.py:563`, `src/geosave/geodata/core/dataarray.py:554`
- Test: `tests/geodata/viz/test_style.py`, `tests/geodata/viz/test_map.py`

**Interfaces:**
- Produces: `Style.from_band(array, *, cmap=None, clim=None, legend=None) -> Style`; `Style.from_column(values: pd.Series, *, theme: Theme = "light", cmap: str = DEFAULT_CMAP) -> Style`; `type Theme = Literal["light", "dark"]`; `CATEGORIES: dict[Theme, list[str]]`. `Style` has no `nodata` field. `Style.stretch` accepts an ndarray or a DataArray.

- [ ] **Step 1: Add hvplot**

Run: `uv add "hvplot>=0.12"`
Expected: `pyproject.toml` lists `"hvplot>=0.12"` in `dependencies`; `uv.lock` updated.

- [ ] **Step 2: Write the failing tests**

In `tests/geodata/viz/test_style.py`: rename `class TestOf` to `class TestFromBand` and every `Style.of(` to `Style.from_band(`; delete `test_continuous_skips_nodata` (nodata leaves `Style`); leave `class TestRgba` until Task 3 deletes it; add:

```python
import colorcet as cc
import pandas as pd


class TestFromColumn:
    def test_numbers_take_a_ramp(self) -> None:
        style = Style.from_column(pd.Series([1.0, 3.0, 2.0]))

        assert style.kind == "continuous" and style.clim == (1.0, 3.0)

    def test_text_takes_one_colour_per_category_in_order(self) -> None:
        style = Style.from_column(pd.Series(["rice", "palm", None, "rice"]))

        assert style.kind == "classes"
        assert style.colors == {"palm": cc.glasbey_dark[0], "rice": cc.glasbey_dark[1]}
        assert style.classes == {"palm": "palm", "rice": "rice"}

    def test_dark_theme_takes_light_hues(self) -> None:
        style = Style.from_column(pd.Series(["rice"]), theme="dark")

        assert style.colors == {"rice": cc.glasbey_light[0]}


def test_stretch_keeps_a_dataarray() -> None:
    stretched = Style("rgb", clim=(0.0, 10.0)).stretch(band([[5.0, 20.0, np.nan]]))

    assert isinstance(stretched, xr.DataArray)
    assert stretched.values.tolist()[0][:2] == [0.5, 1.0]
    assert np.isnan(stretched.values[0, 2])
```

In `tests/geodata/viz/test_map.py`: replace every `Style.of(` with `Style.from_band(`, and rewrite `test_nodata_draws_transparent`:

```python
    def test_nodata_draws_transparent(self) -> None:
        values = np.full((64, 64), -9999.0, "float32")
        values[:, :32] = 1.0
        raster = build_raster({"ndvi": (("y", "x"), values)}, grid(), nodata=-9999.0)
        array = raster[["ndvi"]].gs.to_array()

        overview = webmap.Overview(array, Style.from_band(array))

        assert np.isnan(overview.coarse.values).any()
        assert overview.style.clim == (1.0, 1.0)
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/geodata/viz/test_style.py tests/geodata/viz/test_map.py -q`
Expected: FAIL with `AttributeError: type object 'Style' has no attribute 'from_band'`.

- [ ] **Step 4: Implement in `style.py`**

Add the imports and constants under `DEFAULT_CMAP`:

```python
import colorcet as cc

type Theme = Literal["light", "dark"]

# Category colours readable on each theme's basemap: dark hues on light, light on dark.
CATEGORIES: dict[Theme, list[str]] = {"light": cc.glasbey_dark, "dark": cc.glasbey_light}
```

(`pandas` goes under `TYPE_CHECKING` as `import pandas as pd`.)

Replace the class docstring's attribute list and fields: drop `nodata`; `classes` and `colors` become `Mapping[object, str] | None` described as "Class code or category mapped to its name" and "... to `#rrggbb`, covering every class". Change the docstring example to `Style.from_band(ds["landcover"], legend=ds["landcover"].gs.legend).kind`.

Rename `of` to `from_band`, drop its `nodata` argument and docstring line, and drop `nodata=nodata` from its three `cls(...)` calls. Add after it:

```python
    @classmethod
    def from_column(
        cls, values: pd.Series, *, theme: Theme = "light", cmap: str = DEFAULT_CMAP
    ) -> Style:
        """Colour a table column: quantities on a ramp, anything else by category.

        Categories take the theme's glasbey palette in sorted order, so one
        category keeps its colour across geometry types and redraws.

        Args:
            values: Column of a frame.
            theme: Theme the map shows in, deciding which hues stay visible.
            cmap: Ramp of a quantity.

        Returns:
            Continuous style spanning the column, or classes naming each category.

        Examples:
            >>> list(Style.from_column(fields["crop"]).colors)
            ['palm', 'rice']
        """
        if values.dtype.kind in "biuf":
            return cls("continuous", cmap=cmap, clim=(float(values.min()), float(values.max())))
        palette = CATEGORIES[theme]
        names = sorted(values.dropna().unique(), key=str)
        return cls(
            "classes",
            classes={name: str(name) for name in names},
            colors={name: palette[i % len(palette)] for i, name in enumerate(names)},
        )
```

In `resolve`, delete the two lines filtering `self.nodata`. In `rgba`, delete `if self.nodata is not None: drawn &= values != self.nodata`. Retype `stretch`:

```python
    def stretch[T: (np.ndarray, xr.DataArray)](self, values: T) -> T:
        """Map composite values onto `[0, 1]`, clipping outside the limits; nan stays nan."""
        low, high = self.resolve(np.asarray(values)).clim  # type: ignore[misc]
        return np.clip((values.astype("float32") - low) / (high - low), 0.0, 1.0)
```

- [ ] **Step 5: Update the callers**

`dataset.py:563` and `dataarray.py:554`: `Style.of(` → `Style.from_band(`.

`layers.py` `RasterLayer._style`: return `Style.from_band(data[list(variable)].gs.to_array(), legend=band.gs.legend if len(variable) == 1 else None)`.

`map.py` `Overview.__init__`: replace

```python
        values = array.astype("float32")
        if style.nodata is not None:
            values = values.where(array != style.nodata)
```

with

```python
        # Nodata reads as nan, so a map leaves it clear
        values = array.gs.to_nan().astype("float32").gs.write_nodata(float("nan"))
```

- [ ] **Step 6: Run the viz tests**

Run: `uv run pytest tests/geodata/viz -q`
Expected: PASS (132 before, minus the deleted nodata test, plus 4 new).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/geosave/geodata/viz/style.py src/geosave/geodata/viz/map.py src/geosave/geodata/viz/explorer/layers.py src/geosave/geodata/core/dataset.py src/geosave/geodata/core/dataarray.py tests/geodata/viz/test_style.py tests/geodata/viz/test_map.py
git commit -m "refactor(viz): Style.from_band and from_column; maps read nodata as nan"
```

---

### Task 2: Static figures through hvPlot

**Files:**
- Modify: `src/geosave/geodata/viz/plot.py`
- Modify: `src/geosave/geodata/viz/map.py` (add `drawable`, `split_by_geometry`, `GEOMETRIES`)
- Test: `tests/geodata/viz/test_plot.py`, `tests/geodata/viz/test_map.py`

**Interfaces:**
- Consumes: `Style.stretch`, `Style.positions`, `Style.resolve` (Task 1).
- Produces: `map.GEOMETRIES = ("Polygon", "LineString", "Point")`; `map.drawable(frame) -> gpd.GeoDataFrame`; `map.split_by_geometry(frame) -> dict[str, gpd.GeoDataFrame]` keyed by `GEOMETRIES`, in that order, only types present, original index kept. `plot.figure` and `plot.outline` keep their signatures.

- [ ] **Step 1: Write the failing tests**

In `tests/geodata/viz/test_map.py` add:

```python
class TestSplitByGeometry:
    def test_folds_multi_into_each_type_in_drawing_order(self) -> None:
        from shapely.geometry import MultiPoint

        frame = gpd.GeoDataFrame(
            geometry=[Point(0, 0), box(0, 0, 1, 1), MultiPoint([(1, 1), (2, 2)])],
            crs=3857,
        )

        parts = webmap.split_by_geometry(frame)

        assert list(parts) == ["Polygon", "Point"]
        assert list(parts["Point"].index) == [0, 2]

    def test_refuses_nothing_to_draw_and_collections(self) -> None:
        with pytest.raises(ValueError, match="no geometry"):
            webmap.split_by_geometry(gpd.GeoDataFrame(geometry=[], crs=3857))
        with pytest.raises(ValueError, match="explode"):
            webmap.split_by_geometry(
                gpd.GeoDataFrame(geometry=[GeometryCollection([Point(0, 0)])], crs=3857)
            )
```

In `tests/geodata/viz/test_plot.py`, replace the body of `test_dark_reflectance_is_stretched_to_fill_the_display`'s assertion with:

```python
        assert float(np.nanmax(drawn.dimension_values("R"))) == 1.0
```

(hvPlot's RGB holds its channels as dimensions, not a Dataset; every other
figure test stays as written and pins the behaviour hvPlot must keep.)

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/viz/test_map.py::TestSplitByGeometry tests/geodata/viz/test_plot.py -q`
Expected: FAIL with `AttributeError: module 'geosave.geodata.viz.map' has no attribute 'split_by_geometry'`.

- [ ] **Step 3: Add the geometry split to `map.py`**

Under the colour constants:

```python
# Geometry types a map or figure draws, in drawing order. hvPlot and HoloViews
# draw one type per element, a Multi* geometry with its single type.
GEOMETRIES = ("Polygon", "LineString", "Point")


def drawable(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the rows of a frame that carry geometry a figure or map can draw.

    Raises:
        ValueError: The frame holds no geometry, or a GeometryCollection.
    """
    shown = frame[~(frame.geometry.is_empty | frame.geometry.isna())]
    if shown.empty:
        raise ValueError("nothing to draw: the frame holds no geometry")
    if unknown := set(shown.geom_type.str.removeprefix("Multi")) - set(GEOMETRIES):
        raise ValueError(f"cannot draw {sorted(unknown)}; explode them first")
    return cast("gpd.GeoDataFrame", shown)


def split_by_geometry(frame: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """Split a frame's drawable rows by geometry type, in drawing order.

    Returns:
        {"Polygon" | "LineString" | "Point": rows of that type, index kept},
        only the types the frame holds.

    Raises:
        ValueError: The frame holds no geometry, or a GeometryCollection.
    """
    shown = drawable(frame)
    types = shown.geom_type.str.removeprefix("Multi")
    return {kind: shown[types == kind] for kind in GEOMETRIES if (types == kind).any()}
```

`map.py` needs `import geopandas as gpd` at runtime now (move it out of `TYPE_CHECKING`).

- [ ] **Step 4: Rewrite `plot.py`**

Replace the imports and `figure`/`outline` bodies (keep `_styled` and `_captioned` unchanged):

```python
from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import holoviews as hv
import hvplot.xarray  # noqa: F401  — registers .hvplot on xarray
import hvplot.pandas  # noqa: F401  — registers .hvplot on spatialpandas frames
import numpy as np
import spatialpandas as spd
import xarray as xr
from matplotlib.colors import to_hex, to_rgba

from geosave.geodata.conventions import BAND_DIMENSION, SPATIAL_DIMENSIONS, TIME_COORDINATE
from geosave.geodata.viz.map import split_by_geometry

if TYPE_CHECKING:
    from geosave.geodata.viz.style import Style

hv.extension("matplotlib")


def figure(array, style, *, cols=4, title=None, xlabel=None):  # signature and docstring unchanged
    y, x = SPATIAL_DIMENSIONS
    # A time axis draws one held panel per date, laid out below
    grouped = {"groupby": TIME_COORDINATE, "dynamic": False} if TIME_COORDINATE in array.dims else {}

    if style.kind == "rgb":
        # One stretch for every date, so dates compare
        stretched = style.resolve(array.values).stretch(array)
        drawn = stretched.hvplot.rgb(x=x, y=y, bands=BAND_DIMENSION, **grouped)
    else:
        band = array.squeeze(BAND_DIMENSION, drop=True) if BAND_DIMENSION in array.dims else array
        if style.kind == "classes":
            # Classes draw on their position, so sparse codes leave no empty colours
            band = band.copy(data=style.positions(band.values))
        drawn = band.rename(band.name or "value").hvplot.image(x=x, y=y, **grouped)

    drawn = drawn.map(lambda element: _styled(element, style), [hv.Image, hv.RGB])
    return _captioned(drawn, array, cols=cols, title=title, xlabel=xlabel)


def outline(frame, *, color, alpha):  # signature and docstring unchanged
    color = to_hex(color)
    looks = {
        "Polygon": {"facecolor": to_rgba(color, alpha), "edgecolor": color},
        "LineString": {"color": color, "linewidth": 2},
        "Point": {"color": color},
    }
    # One hvPlot element per geometry type, overlaid
    drawn = [
        spd.GeoDataFrame(part).hvplot().opts(**looks[kind], backend="matplotlib")
        for kind, part in split_by_geometry(frame).items()
    ]
    return cast(
        "hv.Element",
        reduce(operator.mul, drawn).opts(aspect="equal", backend="matplotlib"),
    )
```

Update the module docstring's mention of `elements` (none remain in `plot.py`).

- [ ] **Step 5: Run the figure tests**

Run: `uv run pytest tests/geodata/viz/test_plot.py tests/geodata/viz/test_map.py tests/geodata/core -q -k "plot or Split"`
Expected: PASS. If hvPlot adds a title or axis label a caption test rejects, set `title=""`/`xlabel=""` in the hvPlot call rather than post-processing.

- [ ] **Step 6: Commit**

```bash
git add src/geosave/geodata/viz/plot.py src/geosave/geodata/viz/map.py tests/geodata/viz/test_plot.py tests/geodata/viz/test_map.py
git commit -m "refactor(viz): figures draw through hvPlot; one geometry split in map"
```

---

### Task 3: Layers render through HoloViews operations

**Files:**
- Modify: `src/geosave/geodata/viz/map.py`
- Modify: `src/geosave/geodata/viz/style.py` (delete `rgba`)
- Modify: `src/geosave/geodata/viz/explorer/layers.py`
- Modify: `src/geosave/geodata/viz/explorer/app.py` (`_draw`, `_swipe`)
- Delete: `src/geosave/geodata/viz/elements.py`, `tests/geodata/viz/test_elements.py`
- Test: `tests/geodata/viz/test_map.py`, `tests/geodata/viz/test_style.py`, `tests/geodata/viz/explorer/test_layers.py`, `tests/geodata/viz/explorer/test_app.py`

**Interfaces:**
- Consumes: `Style.from_column`, `Theme` (Task 1); `map.drawable`, `map.split_by_geometry` (Task 2).
- Produces:
  - `map.View.ranges -> dict` (keywords for `rasterize`), `map.View.bounds -> (left, bottom, right, top)`.
  - `map.Overview.view(view) -> xr.DataArray` on `view.geobox`, nan outside.
  - `map.clear(view) -> hv.RGB`; `map.rgb_element(channels: xr.DataArray, alpha: float) -> hv.RGB`; `map.pixels(image: hv.RGB) -> np.ndarray` `(rows, columns, 4)` uint8, north up.
  - `map.OUTLINE: dict[Theme, str]`; `map.COLOR`, `map.HIGHLIGHT`, `map.CMAPS`, `map.BASEMAPS`, `map.tiles` kept.
  - `Layer.theme = param.Selector("light", ["light", "dark"])`; `Layer.image(view) -> hv.RGB | None`.
  - `VectorLayer.style -> Style | None` (replaces `paint`).
  - Deleted: `map.Paint`, `aggregate`, `spread`, `stack`, `composite`, `on_web`, `plain_columns`, `View.canvas`, `Style.rgba`, `elements.*`.

- [ ] **Step 1: Write the failing tests**

`tests/geodata/viz/test_map.py`: delete `class TestShapes`, `test_composite_lays_the_last_image_on_top`, and the `elements`/`datashader`/`pandas` imports they used. Change `test_view_draws_on_the_screen_grid` and `test_class_codes_never_blend` to read `drawn.values` (a DataArray now) and `drawn.shape`. Add:

```python
def test_rgb_element_is_clear_where_a_channel_is_nan() -> None:
    view = webmap.View((0.0, 2.0), (0.0, 1.0), 2, 1)
    channels = xr.DataArray(
        np.array([[[1.0, np.nan]], [[0.0, 0.0]], [[0.0, 0.0]]]),
        dims=("band", "y", "x"),
        coords={"y": view.geobox.coords["y"].values, "x": view.geobox.coords["x"].values},
    )

    rgba = webmap.pixels(webmap.rgb_element(channels, 0.5))

    assert list(rgba[0, 0]) == [255, 0, 0, 128] and rgba[0, 1, 3] == 0


def test_clear_covers_the_view() -> None:
    view = webmap.View((0.0, 100.0), (0.0, 10.0), 50, 50)

    assert webmap.pixels(webmap.clear(view)).shape == (5, 50, 4)
```

`tests/geodata/viz/test_style.py`: delete `class TestRgba`.

`tests/geodata/viz/explorer/test_layers.py`: change the helper and colour expectations:

```python
def drawn(image) -> int:
    return int((webmap.pixels(image)[..., 3] > 0).sum())
```

In `test_draws_a_band_or_a_composite_in_colour` use `composite, classes = webmap.pixels(shown.image(view)), ...` after each draw (convert both with `webmap.pixels`). In `test_steps_through_its_dates` wrap each image in `webmap.pixels(...)`. In `test_outlines_the_selection_bright` wrap `plain` and `chosen` in `webmap.pixels(...)`. In `test_colours_one_category_alike_in_every_family` expect `palm, rice = cc.glasbey_dark[:2]`. Add:

```python
    def test_missing_categories_draw_nothing(self, fields: gpd.GeoDataFrame) -> None:
        shown = VectorLayer(fields.assign(crop=["rice", None, "rice"]), column="crop")

        assert shown.key().swatches == {"rice": cc.glasbey_dark[0]}
        assert drawn(shown.image(whole(shown))) > 0

    def test_dark_theme_recolours_categories(self, fields: gpd.GeoDataFrame) -> None:
        shown = VectorLayer(fields, column="crop")

        shown.theme = "dark"

        assert shown.key().swatches["palm"] == cc.glasbey_light[0]
```

`tests/geodata/viz/explorer/test_app.py` `TestSwipe`: add

```python
    def test_swipe_keeps_orientation(self, monthly: xr.Dataset) -> None:
        from geosave.geodata.viz import map as webmap

        explorer = Explorer(monthly)
        (raster,) = explorer.layers
        sides = list(explorer.param.left.objects.values())[1:]
        explorer.left, explorer.right = sides[0], sides[1]
        explorer.split = 1.0
        x_range, y_range = explorer._ranges
        view = webmap.View(x_range, y_range, 60, 60)

        swiped = webmap.pixels(explorer._swipe(view))
        left = webmap.pixels(raster.image(view, sides[0][1]))

        # Left of the cut line, the swipe shows the left side exactly as drawn.
        assert (swiped[:, :-2] == left[:, :-2]).all()
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/viz -q`
Expected: FAIL — `rgb_element`, `clear`, `pixels` missing; layers still return arrays.

- [ ] **Step 3: Rewrite `map.py`**

Module docstring:

```python
"""Draw on a Web Mercator map: the view on screen, rasters read at its detail, basemaps.

A map redraws on every pan and zoom, so it needs a live Python process: a
notebook kernel or `panel serve`. Each layer renders at the screen's size
through HoloViews' datashader operations, rasters regridded by GDAL first.
"""
```

Imports: `cast`, `geopandas as gpd`, `holoviews as hv`, `holoviews.plotting.bokeh  # noqa: F401`, `numpy as np`, `xarray as xr`, `tile_sources`, `GeoBox`, `BAND_DIMENSION`, `SPATIAL_DIMENSIONS`, `Style`, `Theme`. Drop `colorcet`, `pandas`, `process_cmap`, `datashader`, `odc.geo.xr`, `elements`.

Constants (replacing the old colour block, `CMAPS` moved here from `layers.py`):

```python
# Colour of features no column colours, and of selected features.
COLOR = "#1f77b4"
HIGHLIGHT = "#ffd400"
# Colour of the boundaries of polygons a column colours, visible on each theme's basemap.
OUTLINE: dict[Theme, str] = {"light": "#333333", "dark": "#e0e0e0"}
# Basemap a map opens on in each theme.
THEME_BASEMAP: dict[Theme, str] = {"light": "Light", "dark": "Dark"}
# Offered in the picker; any matplotlib, colorcet or bokeh name is accepted.
CMAPS = ["viridis", "cividis", "magma", "RdYlGn", "YlGn", "Spectral", "gray"]
```

`View`: delete `canvas`; add

```python
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return `(left, bottom, right, top)` in metres."""
        (left, right), (bottom, top) = self.x_range, self.y_range
        return left, bottom, right, top

    @property
    def ranges(self) -> dict[str, object]:
        """Return the keywords that make a datashader operation draw on this view's grid."""
        rows, columns = self.geobox.shape
        return {
            "x_range": self.x_range,
            "y_range": self.y_range,
            "width": columns,
            "height": rows,
            "dynamic": False,
        }
```

`Overview` (keep `bounds`, `value_at`, `window`; replace `__init__` and `view`):

```python
    def __init__(self, array: xr.DataArray, style: Style, *, size: int = 2048) -> None:
        if style.kind != "rgb" and BAND_DIMENSION in array.dims:
            array = array.squeeze(BAND_DIMENSION, drop=True)
        # Nodata reads as nan, so a map leaves it clear
        values = array.gs.to_nan().astype("float32").gs.write_nodata(float("nan"))
        # An array in memory warps lazily too, so a map warps only the windows it shows
        if values.chunks is None:
            y, x = SPATIAL_DIMENSIONS
            values = values.chunk({y: size, x: size})

        # Class codes must not blend; continuous values average into a display pixel
        self.classes = style.kind == "classes"
        kernel = "nearest" if self.classes else "average"
        self.full = values.gs.reproject(WEB_MERCATOR, resampling=kernel)
        grid = self.full.gs.geobox
        self.coarse = values.gs.reproject(
            grid.zoom_to(min(size, max(grid.shape))), resampling=kernel
        ).compute()
        self.size = size
        # Limits come from the whole raster, so colours hold still while panning.
        self.style = style.resolve(self.coarse.values)

    def view(self, view: View) -> xr.DataArray:
        """Return the values a view shows, one per screen pixel, north up, nan outside the raster."""
        window = self.window(view.x_range, view.y_range)
        # GDAL regrids to the screen, so no first-use compile delays a map
        return window.gs.reproject(view.geobox, resampling="mode" if self.classes else "average")
```

In `bounds`, read `self.coarse.gs.geobox.extent.boundingbox`.

Image helpers (replace `on_web` … `_hoverable`, keep `tiles`):

```python
def clear(view: View) -> hv.RGB:
    """Return a clear image covering a view."""
    rows, columns = view.geobox.shape
    return hv.RGB(np.zeros((rows, columns, 4), np.uint8), bounds=view.bounds)


def rgb_element(channels: xr.DataArray, alpha: float) -> hv.RGB:
    """Return three channels valued in `[0, 1]` as an RGB element, clear where any is nan.

    Args:
        channels: `(band, y, x)` on a view's grid, as `.gs.colorize` or
            `Style.stretch` return them.
        alpha: Opacity of every pixel drawn.
    """
    y, x = SPATIAL_DIMENSIONS
    drawn = channels.notnull().all(BAND_DIMENSION).values
    red, green, blue = (
        channels.isel({BAND_DIMENSION: i}).fillna(0.0).values for i in range(3)
    )
    return hv.RGB(
        (channels[x].values, channels[y].values, red, green, blue, drawn * alpha),
        vdims=["R", "G", "B", "A"],
    )


def pixels(image: hv.RGB) -> np.ndarray:
    """Return an RGB element's `(rows, columns, 4)` bytes, north up."""
    bands = [image.dimension_values(d, flat=False) for d in image.vdims]
    rgba = np.dstack(bands)
    if rgba.dtype != np.uint8:
        rgba = np.round(np.nan_to_num(rgba) * 255).astype(np.uint8)
    # HoloViews hands rows south to north when y ascends
    ys = image.dimension_values(1, expanded=False)
    return rgba[::-1] if len(ys) > 1 and ys[0] < ys[-1] else rgba
```

(Step 5 confirms `pixels` orientation against `test_swipe_keeps_orientation`; if HoloViews already returns rows north up, drop the flip and its comment.)

- [ ] **Step 4: Rewrite layer drawing in `layers.py`**

Imports: drop `elements`, `datashader as ds` stays, add `holoviews as hv`, `spatialpandas as spd`, `from holoviews.operation.datashader import rasterize, shade, spread, stack`, `from geosave.geodata.viz.style import DEFAULT_CMAP, Style, Theme`; take `CMAPS` from `webmap`. Add `ROW = "row"` beside `SIZE`:

```python
# Columns every shape carries: the position of its frame row, and its larger side in metres.
ROW = "row"
SIZE = "size"
```

`Layer`: add

```python
    theme = param.Selector(default="light", objects=["light", "dark"], precedence=-1)
```

and change `image`'s docstring return to "RGB element on the view's grid, None where the layer lies off the view."

`RasterLayer.image`:

```python
    def image(self, view: View, time: object = None) -> hv.RGB | None:
        """Draw the shown variable at a view's detail, on the bar's date or `time`.

        Returns:
            RGB element on the view's grid, None where the raster lies off the view.
        """
        if not _overlaps(self.bounds, view):
            return None
        overview = self.overview(self.variable, self.time if time is None else time)
        style = overview.style
        window = overview.view(view)

        # Classes take their legend's colours, a composite its stretch, a band its ramp
        if style.kind == "classes":
            return webmap.rgb_element(window.gs.colorize(), self.alpha)
        if style.kind == "rgb":
            return webmap.rgb_element(style.stretch(window), self.alpha)
        y, x = SPATIAL_DIMENSIONS
        return shade(
            hv.Image(window, kdims=[x, y]),
            cmap=self.cmap,
            clims=style.clim,
            cnorm="linear",
            alpha=round(self.alpha * 255),
            dynamic=False,
        )
```

`VectorLayer.__init__` (replacing everything from `web = webmap.on_web(frame)` on):

```python
        if frame.crs is None:
            raise ValueError("a map needs a CRS; set one with set_crs")
        web = webmap.drawable(frame).to_crs(webmap.WEB_MERCATOR)
        # Only numbers and text colour features
        columns = [
            str(c) for c in web.columns
            if c != web.geometry.name and _colourable(cast("pd.Series", web[c]))
        ]
        if column is not None and column not in columns:
            raise KeyError(f"{column!r} is not a plain column; the frame holds {columns}")
        super().__init__(name=name, alpha=alpha, color=to_hex(color))
        self.frame = web
        self.columns = columns
        self._crs = frame.crs
        self.param.column.objects = {"single colour": None, **{c: c for c in columns}}
        self.column = column

        # Rows carry their position and size; text becomes a category datashader counts
        width, height = (
            web.bounds[["maxx", "maxy"]].values - web.bounds[["minx", "miny"]].values
        ).T
        rows = web.assign(
            **{c: _categories(web[c]) for c in columns if web[c].dtype.kind not in "biuf"},
            **{ROW: np.arange(len(web)), SIZE: np.maximum(width, height)},
        )

        # Shapes by how they draw, converted once to spatialpandas
        parts = webmap.split_by_geometry(rows)
        shapes = {kind: part for kind, part in parts.items() if kind != "Point"}
        if "Polygon" in parts:
            polygons = parts["Polygon"]
            shapes["Boundary"] = polygons.assign(**{polygons.geometry.name: polygons.boundary})
        shapes["Dot"] = rows.assign(**{rows.geometry.name: rows.representative_point()})
        self._shapes = {kind: spd.GeoDataFrame(part) for kind, part in shapes.items()}
```

Replace `paint` with:

```python
    @property
    def style(self) -> Style | None:
        """Return how `column` colours the features in this theme, None for one colour."""
        if self.column is None:
            return None
        return Style.from_column(cast("pd.Series", self.frame[self.column]), theme=self.theme)
```

`key`:

```python
    def key(self) -> Key:
        """Return the colour of every feature, or of each value of `column`."""
        style = self.style
        if style is None:
            return Key(swatches={"every feature": self.color})
        if style.kind == "classes":
            colors = style.colors or {}
            return Key(swatches={name: colors[code] for code, name in (style.classes or {}).items()})
        return Key(ramp=style.palette, clim=style.clim)
```

`image`:

```python
    def image(self, view: View) -> hv.RGB | None:
        """Draw the frame at a view's detail.

        Polygons fill at `alpha` under a solid boundary; lines and points
        draw solid; a feature under two pixels across draws as a dot; the
        selected features draw outlined bright on top.

        Returns:
            RGB element on the view's grid, None where the frame lies off the view.
        """
        if not _overlaps(self.bounds, view):
            return None

        def large(kind: str) -> spd.GeoDataFrame:
            shapes = self._shapes[kind]
            return shapes[shapes[SIZE] >= 2 * view.pixel]

        dots = self._shapes["Dot"]
        dots = dots[dots[SIZE] < 2 * view.pixel]
        outline = webmap.OUTLINE[self.theme] if self.column else self.color

        # Shapes at least two pixels across, then dots
        images = []
        if "Polygon" in self._shapes:
            images.append(self._shade(hv.Polygons, large("Polygon"), view, self.alpha))
            images.append(self._shade(hv.Path, large("Boundary"), view, color=outline))
        if "LineString" in self._shapes:
            images.append(self._shade(hv.Path, large("LineString"), view))
        images.append(spread(self._shade(hv.Points, dots, view), px=2, dynamic=False))

        # The selection on top: outlines of shapes, rings round dots
        if self.selected:
            for kind in ("Boundary", "LineString"):
                if kind in self._shapes:
                    chosen = large(kind)
                    chosen = chosen[chosen[ROW].isin(self.selected)]
                    shaded = self._shade(hv.Path, chosen, view, color=webmap.HIGHLIGHT)
                    images.append(spread(shaded, px=1, dynamic=False))
            chosen = dots[dots[ROW].isin(self.selected)]
            shaded = self._shade(hv.Points, chosen, view, color=webmap.HIGHLIGHT)
            images.append(spread(shaded, px=3, dynamic=False))
        return stack(hv.Overlay(images), dynamic=False)

    def _shade(
        self,
        element: type[hv.Element],
        shapes: spd.GeoDataFrame,
        view: View,
        alpha: float = 1.0,
        color: str | None = None,
    ) -> hv.RGB:
        """Draw shapes on a view's grid: in `color`, else as `column` colours them."""
        if shapes.empty:
            return webmap.clear(view)
        drawn = element(shapes, vdims=[c for c in shapes.columns if c != shapes.geometry.name])
        opacity = round(alpha * 255)
        style = None if color else self.style

        # One colour marks coverage, a category counts its shapes, a quantity averages
        if style is None:
            covered = rasterize(drawn, aggregator=ds.any(), **view.ranges)
            fill = color or self.color
            return shade(covered, cmap=[fill, fill], alpha=opacity, dynamic=False)
        if style.kind == "classes":
            counts = rasterize(drawn, aggregator=ds.count_cat(self.column), **view.ranges)
            return shade(
                counts, color_key=dict(style.colors or {}), min_alpha=opacity, alpha=opacity, dynamic=False
            )
        means = rasterize(drawn, aggregator=ds.mean(self.column), **view.ranges)
        return shade(means, cmap=style.palette, clims=style.clim, cnorm="linear", alpha=opacity, dynamic=False)
```

Module helpers: delete `_plain`; keep `_overlaps`; change `_counted` to

```python
def _counted(frame: gpd.GeoDataFrame) -> str:
    """Count a frame's rows by geometry type, such as "1 polygon, 4 points"."""
    nouns = {"Polygon": "polygon", "LineString": "line", "Point": "point"}
    return ", ".join(
        f"{len(part)} {nouns[kind]}{'' if len(part) == 1 else 's'}"
        for kind, part in webmap.split_by_geometry(frame).items()
    )
```

and add

```python
def _colourable(values: pd.Series) -> bool:
    """Return whether a column's values can colour features: numbers or text."""
    if values.dtype.kind in "biuf":
        return True
    return bool(values.dropna().map(lambda value: isinstance(value, str)).all())


def _categories(values: pd.Series) -> pd.Categorical:
    """Return text as categories in sorted order, the order `Style.from_column` colours them."""
    return pd.Categorical(values, categories=sorted(values.dropna().unique(), key=str))
```

`CatalogLayer.__init__`: `self._items = webmap.drawable(frame)`.

- [ ] **Step 5: Stack in the explorer**

`app.py` imports `from holoviews.operation.datashader import stack`. `_draw`, from `images = []` on:

```python
        images = []
        for shown in self.stacked:
            if shown in compared:
                if shown is next(x for x in self.stacked if x in compared):
                    images.append(self._swipe(view))
            elif shown.visible:
                images.append(shown.image(view))
        drawn = [x for x in images if x is not None]
        composite = stack(hv.Overlay(drawn), dynamic=False) if drawn else webmap.clear(view)
        return composite.opts(hooks=[self._move], backend="bokeh")
```

`_swipe`:

```python
    def _swipe(self, view: webmap.View) -> hv.RGB:
        """Return the left side's raster left of the cut, the right side's right of it."""
        (left, left_time), (right, right_time) = self.left, self.right
        sides = [left.image(view, left_time), right.image(view, right_time)]
        image, other = (webmap.pixels(x if x is not None else webmap.clear(view)) for x in sides)
        cut = round(self.split * image.shape[1])
        image[:, cut:] = other[:, cut:]
        # A line marks the cut
        image[:, max(cut - 1, 0) : cut + 1] = CUT
        return hv.RGB(image, bounds=view.bounds)
```

- [ ] **Step 6: Delete `elements`**

```bash
git rm src/geosave/geodata/viz/elements.py tests/geodata/viz/test_elements.py
```

Remove `elements` from `viz/__init__.py`'s docstring: "`style` decides how values become colour, `plot` draws figures through hvPlot on matplotlib, `map` holds the Web Mercator view and raster overviews, and `explorer` lays maps and controls out as a Panel app."

- [ ] **Step 7: Run the viz tests**

Run: `uv run pytest tests/geodata/viz -q`
Expected: PASS. If `test_swipe_keeps_orientation` fails with an upside-down match, flip the `pixels` orientation rule as noted in Step 3.

- [ ] **Step 8: Commit**

```bash
git add -A src/geosave/geodata/viz tests/geodata/viz
git commit -m "refactor(viz): layers render through rasterize, shade and stack; drop hand-written canvas"
```

---

### Task 4: Light and dark themes

**Files:**
- Modify: `src/geosave/geodata/viz/explorer/app.py`
- Modify: `src/geosave/geodata/viz/explorer/docks.py` (`AttributeTable` table theme)
- Modify: `src/geosave/geodata/core/dataset.py`, `dataarray.py`, `vector.py` (`explore(basemap=None)`)
- Test: `tests/geodata/viz/explorer/test_app.py`, `tests/geodata/viz/explorer/test_docks.py`

**Interfaces:**
- Consumes: `Layer.theme`, `webmap.THEME_BASEMAP`, `webmap.OUTLINE` (Task 3).
- Produces: `Explorer.theme = param.Selector("light", ["light", "dark"])`; `Explorer(*layers, basemap: str | None = None, height=HEIGHT)`; `docks.TABLE_THEME: dict[Theme, str] = {"light": "materialize", "dark": "midnight"}`.

- [ ] **Step 1: Write the failing tests**

`test_app.py`:

```python
class TestTheme:
    def test_opens_on_the_themes_basemap(self, scene: xr.Dataset) -> None:
        assert Explorer(scene).basemap == "Light"

    def test_dark_theme_reaches_basemap_and_layers(
        self, scene: xr.Dataset, fields: gpd.GeoDataFrame
    ) -> None:
        explorer = Explorer(scene, fields)

        explorer.theme = "dark"

        assert explorer.basemap == "Dark"
        assert {x.theme for x in explorer.layers} == {"dark"}

    def test_theme_keeps_a_picked_basemap(self, scene: xr.Dataset) -> None:
        explorer = Explorer(scene, basemap="Imagery")

        explorer.theme = "dark"

        assert explorer.basemap == "Imagery"

    def test_new_layers_join_in_the_current_theme(
        self, scene: xr.Dataset, fields: gpd.GeoDataFrame
    ) -> None:
        explorer = Explorer(scene)
        explorer.theme = "dark"

        explorer.add(fields)

        assert explorer.layers[-1].theme == "dark"

    def test_the_toggle_steers_the_theme(self, scene: xr.Dataset) -> None:
        explorer = Explorer(scene)
        (toggle,) = pn.panel(explorer).select(pmui.ThemeToggle)

        toggle.value = True

        assert explorer.theme == "dark"
```

`test_docks.py` `TestAttributeTable`:

```python
    def test_tables_follow_the_theme(self, catalog: gpd.GeoDataFrame) -> None:
        explorer = Explorer(catalog)
        (table,) = tables(pn.panel(AttributeTable(explorer=explorer)))

        explorer.theme = "dark"

        assert table.theme == "midnight"
```

Update `test_multiplying_trees_another_layer` and any test passing `basemap="OpenStreetMap"` expectations to the new default where it asserted the default.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/viz/explorer -q -k "Theme or theme"`
Expected: FAIL — `Explorer` has no `theme`.

- [ ] **Step 3: Implement the theme in `app.py`**

Parameters:

```python
    theme = param.Selector(default="light", objects=["light", "dark"], doc="Theme the page shows in.")
    basemap = param.Selector(default="Light", objects=list(webmap.BASEMAPS))
```

`__init__(self, *layers, basemap: str | None = None, height: int = HEIGHT)`: after `stack = [...]`, call `super().__init__(basemap=basemap or webmap.THEME_BASEMAP["light"])`. `__mul__` passes `basemap=self.basemap`. Docstring `basemap`: "Basemap under the layers, as `viz.map.BASEMAPS` names it. None opens on the theme's own and follows it."

Watcher:

```python
    @param.depends("theme", watch=True)
    def _retheme(self) -> None:
        """Give every layer the theme; the theme's basemap follows unless one was picked."""
        for shown in self.layers:
            shown.theme = self.theme
        old = webmap.THEME_BASEMAP["dark" if self.theme == "light" else "light"]
        if self.basemap == old:
            self.basemap = webmap.THEME_BASEMAP[self.theme]
```

In `_restack`, before watching new layers: `for shown in self.layers: shown.theme = self.theme`.

In `__panel__`: the basemap bar gains a toggle in a notebook; a served page drives the theme from its own toggle:

```python
        toggle = pmui.ThemeToggle(value=self.theme == "dark")
        toggle.param.watch(self._toggled, "value")
        bar = [basemap] if _served() else [basemap, toggle]
        ...
        if _served():
            page = pmui.Page(..., dark_theme=self.theme == "dark")
            page.param.watch(self._toggled, "dark_theme")
            return page
```

```python
    def _toggled(self, event: param.parameterized.Event) -> None:
        """Follow the page's light or dark switch."""
        self.theme = "dark" if event.new else "light"
```

- [ ] **Step 4: Tables follow the theme in `docks.py`**

```python
# Tabulator's stylesheet in each theme.
TABLE_THEME = {"light": "materialize", "dark": "midnight"}
```

`AttributeTable._table` passes `theme=TABLE_THEME[self.explorer.theme]`; add

```python
    @param.depends("explorer.theme", watch=True)
    def _retheme(self) -> None:
        """Restyle every table for the page's theme."""
        for table in self._tables.values():
            table.theme = TABLE_THEME[self.explorer.theme]
```

- [ ] **Step 5: Accessors default to the theme's basemap**

In `dataset.py`, `dataarray.py`, `vector.py` `explore`: `basemap: str | None = None`; docstring "Basemap under the layers, as `viz.map.BASEMAPS` names it; None follows the theme."

- [ ] **Step 6: Run the explorer tests**

Run: `uv run pytest tests/geodata/viz -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/geosave/geodata/viz/explorer/app.py src/geosave/geodata/viz/explorer/docks.py src/geosave/geodata/core/dataset.py src/geosave/geodata/core/dataarray.py src/geosave/geodata/core/vector.py tests/geodata/viz/explorer
git commit -m "feat(explorer): light and dark themes steer basemap, categories and tables"
```

---

### Task 5: Sidebar layout and themed legends

**Files:**
- Modify: `src/geosave/geodata/viz/explorer/app.py` (`__panel__`, `HEIGHT`)
- Modify: `src/geosave/geodata/viz/explorer/docks.py` (`Properties._styled`, `_legend`)
- Test: `tests/geodata/viz/explorer/test_app.py`, `tests/geodata/viz/explorer/test_docks.py`

**Interfaces:**
- Consumes: `Key` (layers), `Explorer.theme` (Task 4).
- Produces: `docks._legend(key: Key) -> list[pn.viewable.Viewable]`; `app.HEIGHT = 640`.

- [ ] **Step 1: Write the failing tests**

`test_app.py`: replace `test_lays_out_the_stack_and_tabs_beside_the_map` and `test_served_as_a_page`:

```python
    def test_lays_out_the_sidebar_beside_the_map(self, scene: xr.Dataset) -> None:
        explorer = Explorer(scene, basemap="Imagery", height=400)

        (row,) = cast(pmui.Paper, pn.panel(explorer))
        side, main = cast(pmui.Row, row)
        assert side.width == 320
        # Catalogs, layers, then the selected layer's tabs, all on the left
        assert len(side.select(pmui.MenuList)) == 2 and side.select(pmui.Tabs)
        bar, pane, *_, table = cast(pmui.Column, main)
        assert bar.select(pmui.Select)[0].value == "Imagery"
        assert isinstance(pane, pn.pane.HoloViews) and pane.height == 400
        assert isinstance(table, pmui.Paper)

    def test_served_as_a_page(self, scene: xr.Dataset, monkeypatch) -> None:
        monkeypatch.setattr(app, "_served", lambda: True)

        page = pn.panel(Explorer(scene))

        assert isinstance(page, pmui.Page)
        sidebar = pmui.Column(*page.sidebar)
        assert sidebar.select(pmui.MenuList) and sidebar.select(pmui.Tabs)
        assert not page.contextbar
```

`test_docks.py`: in `test_offers_the_selected_layers_style_and_legend` read the legend text from Typography:

```python
        legend = " ".join(str(t.object) for t in shown.select(pmui.Typography))
        assert "trees (1)" in legend
```

and rewrite `test_legends_name_classes_and_categories`:

```python
def test_legends_name_classes_and_categories(
    scene: xr.Dataset, fields: gpd.GeoDataFrame
) -> None:
    explorer = Explorer(scene["cover"], fields)
    raster, vector = explorer.layers
    cast(VectorLayer, vector).column = "crop"

    def words(key) -> str:
        return " ".join(str(t.object) for x in _legend(key) for t in pmui.Column(x).select(pmui.Typography))

    assert "trees (1)" in words(raster.key())
    assert "palm" in words(vector.key())
    # Only swatches carry colour; text takes the theme's
    assert not any("color:" in str(h.object) for x in _legend(raster.key()) for h in pmui.Column(x).select(pn.pane.HTML))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/viz/explorer -q -k "layout or page or legend or sidebar"`
Expected: FAIL — tabs still in the contextbar, legend still HTML.

- [ ] **Step 3: Sidebar layout in `app.py`**

`HEIGHT = 640`. In `__panel__`, `side = [docks["catalogs"], docks["layers"], docks["properties"]]`; served:

```python
            page = pmui.Page(
                title="GeoSave Explorer",
                sidebar=side,
                sidebar_width=SIDE,
                main=[main],
                theme_config=THEME,
                dark_theme=self.theme == "dark",
            )
```

notebook: `pmui.Paper(pmui.Row(pmui.Column(*side, width=SIDE), main, sizing_mode="stretch_width"), theme_config=THEME, sizing_mode="stretch_width")`. Update the class docstring: "Beside the map, on the left, sit the catalogs, then the layers, one stack top first, then the selected layer's Style, Info and Identify tabs."

- [ ] **Step 4: Themed legend in `docks.py`**

```python
# Side of a legend swatch, in pixels.
SWATCH = 12


def _legend(key: Key) -> list[pn.viewable.Viewable]:
    """Return a layer's key: named swatches, a ramp between its limits, or a note.

    Text takes the theme's colour; only swatches and the ramp carry their own.
    """
    if key.ramp:
        low, high = key.clim or (float("nan"), float("nan"))
        stops = ", ".join(key.ramp[:: max(1, len(key.ramp) // 16)])
        ramp = pn.pane.HTML(
            f'<div style="height:10px;background:linear-gradient(to right, {stops})"></div>',
            sizing_mode="stretch_width",
        )
        limits = pmui.Row(
            pmui.Typography(f"{low:.4g}", variant="caption"),
            pn.HSpacer(),
            pmui.Typography(f"{high:.4g}", variant="caption"),
            sizing_mode="stretch_width",
        )
        return [ramp, limits]
    if not key.swatches:
        return [pmui.Typography(key.note, variant="caption")]
    return [
        pmui.Row(
            pn.pane.HTML(
                f'<div style="width:{SWATCH}px;height:{SWATCH}px;background:{color}"></div>',
                width=SWATCH,
                height=SWATCH,
                margin=(4, 6, 0, 0),
            ),
            pmui.Typography(name, variant="body2"),
        )
        for name, color in key.swatches.items()
    ]
```

In `Properties._styled`, the legend becomes

```python
        legend = pmui.Column(
            objects=pn.bind(
                lambda *_: _legend(shown.key()),
                *(shown.param[p] for p in [*shown.controls, "theme"]),
            ),
            sizing_mode="stretch_width",
        )
```

Delete the now-unused `html` import if nothing else uses it (`_text` still does; keep it then).

- [ ] **Step 5: Run the explorer tests**

Run: `uv run pytest tests/geodata/viz -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/geosave/geodata/viz/explorer tests/geodata/viz/explorer
git commit -m "feat(explorer): sidebar holds layers and tabs; legends take the theme's text colour"
```

---

### Task 6: Raster Info tab

**Files:**
- Modify: `src/geosave/geodata/viz/explorer/layers.py` (`Layer.info`, `RasterLayer.info`, `VectorLayer.info`, `CatalogLayer.info`, helpers)
- Modify: `src/geosave/geodata/viz/explorer/docks.py` (`_info`)
- Test: `tests/geodata/viz/explorer/test_layers.py`, `tests/geodata/viz/explorer/test_docks.py`

**Interfaces:**
- Consumes: `docks.TABLE_THEME`, `Explorer.theme` (Task 4).
- Produces: `type Section = dict[str, str] | pd.DataFrame`; `Layer.info() -> dict[str, Section]` (title → facts or table). `Layer.attrs()` unchanged.

- [ ] **Step 1: Write the failing tests**

`test_layers.py`, replace `test_describes_itself`:

```python
    def test_describes_itself(self, monthly: xr.Dataset) -> None:
        shown = RasterLayer(monthly.assign_attrs(title="Monthly scene", license="CC-BY-4.0"), "B04")

        info = shown.info()

        assert shown.summary == "4 variables, 3 dates, 10 m"
        assert list(info) == ["Overview", "Grid", "Time", "Bands", "Classes"]
        assert info["Overview"] == {"Title": "Monthly scene", "License": "CC-BY-4.0"}
        assert info["Grid"]["CRS"] == "EPSG:32749"
        assert info["Grid"]["Extent"].endswith("(W, S, E, N)")
        assert info["Time"]["First"] == "2024-03-05"
        assert list(info["Bands"]["variable"]) == ["B04", "B03", "B02", "cover"]
        assert info["Classes"][["code", "class", "colour"]].values.tolist() == [
            [0, "water", "#419bdf"],
            [1, "trees", "#397d49"],
        ]
        assert list(shown.attrs()) == ["raster", "B04", "B03", "B02", "cover"]

    def test_leaves_out_empty_sections(self, scene: xr.Dataset) -> None:
        info = RasterLayer(scene[["B04"]]).info()

        assert list(info) == ["Grid", "Bands"]
        assert list(info["Bands"].columns) == ["variable", "dtype"]

    def test_bands_list_spectral_and_units(self, scene: xr.Dataset) -> None:
        red = scene["B04"].gs.write_spectral(common_name="red", center_wavelength=0.665)
        info = RasterLayer(scene.assign(B04=red.assign_attrs(units="1")), "B04").info()

        row = info["Bands"].set_index("variable").loc["B04"]
        assert row["common name"] == "red" and row["centre wavelength"] == 0.665
        assert row["units"] == "1"
```

(Check `write_spectral`'s signature in `core/dataarray.py:191` before running; adjust keyword names to match it.)

`TestVectorLayer`: add

```python
    def test_describes_itself(self, fields: gpd.GeoDataFrame) -> None:
        info = VectorLayer(fields).info()

        assert list(info) == ["Overview"]
        assert info["Overview"]["Geometry"] == "2 polygons, 1 point"
```

`test_docks.py`, replace `test_info_tables_a_rasters_metadata`:

```python
    def test_info_lists_sections_then_every_attr(self, scene: xr.Dataset) -> None:
        explorer = Explorer(scene.assign_attrs(title="a scene"))

        shown = pn.panel(Properties(explorer=explorer))

        headings = [str(t.object) for t in shown.select(pmui.Typography)]
        assert {"Overview", "Grid", "Bands", "Classes"} <= set(headings)
        (raw,) = [t for t in tables(shown) if "section" in t.value]
        assert "a scene" in set(raw.value["value"])
        assert shown.select(pmui.Accordion)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/viz/explorer -q -k "describes or sections or spectral or info"`
Expected: FAIL — `info()` still flat.

- [ ] **Step 3: Implement sections in `layers.py`**

Imports: `import geosave.geodata.attrs as attrs`. Beside `Box`:

```python
# One part of the Info tab: named facts, or a table.
type Section = dict[str, str] | pd.DataFrame
```

`Layer.info` signature/docstring: `def info(self) -> dict[str, Section]: """Return the Info tab's sections, by title, those with nothing to show left out."""`.

`RasterLayer.info`:

```python
    def info(self) -> dict[str, Section]:
        """Return what the raster is, where and when it lies, its bands and its classes.

        Read from the attr groups the raster carries, as a STAC browser shows
        an Item: Overview (ACDD), Grid, Time, Bands and Classes.
        """
        data = self._data
        sections: dict[str, Section] = {}

        # What it is, as its ACDD attrs describe it
        acdd = attrs.parse(data, attrs.ACDD)
        described = {} if acdd is None else acdd.model_dump(exclude_none=True)
        if described:
            sections["Overview"] = {k.capitalize(): str(v) for k, v in described.items()}

        # Where it lies
        grid = data.odc.geobox
        west, south, east, north = grid.extent.to_crs("EPSG:4326").boundingbox
        sections["Grid"] = {
            "CRS": str(data.gs.crs_name),
            "Size": f"{grid.shape.x} × {grid.shape.y} px",
            "Pixel": f"{abs(grid.resolution.x):g} m",
            "Extent": f"{west:.4f}, {south:.4f}, {east:.4f}, {north:.4f} (W, S, E, N)",
        }

        # When it was observed
        if self.dates:
            first, last = pd.Timestamp(self.dates[0]), pd.Timestamp(self.dates[-1])
            time = {"Dates": str(len(self.dates)), "First": f"{first:%Y-%m-%d}", "Last": f"{last:%Y-%m-%d}"}
            spec = attrs.parse(data[TIME_COORDINATE], attrs.TimeSpec)
            if spec is not None:
                time["Step"] = str(spec.time_freq)
            sections["Time"] = time

        # Each variable, then the classes those with a legend name
        sections["Bands"] = _bands(data)
        classes = _classes(data)
        if not classes.empty:
            sections["Classes"] = classes
        return sections
```

Helpers:

```python
def _bands(data: xr.Dataset) -> pd.DataFrame:
    """Table each variable's dtype, fill, units, spectral band and packing; empty columns left out."""
    rows = []
    for name, band in data.data_vars.items():
        quantity = attrs.parse(band, attrs.Quantity)
        spectral, packing = band.gs.spectral, band.gs.packing
        rows.append({
            "variable": str(name),
            "dtype": str(band.dtype),
            "nodata": band.gs.nodata,
            "units": quantity and quantity.units,
            "common name": spectral and spectral.common_name,
            "centre wavelength": spectral and spectral.center_wavelength,
            "FWHM": spectral and spectral.full_width_half_max,
            "scale": packing and packing.scale_factor,
            "offset": packing and packing.add_offset,
        })
    return pd.DataFrame(rows).dropna(axis="columns", how="all")


def _classes(data: xr.Dataset) -> pd.DataFrame:
    """Table the classes every variable's legend names, with their colours."""
    rows = [
        {"variable": str(name), "code": code, "class": label, "colour": (legend.color_map or {}).get(code)}
        for name, band in data.data_vars.items()
        if (legend := band.gs.legend) is not None and legend.classes
        for code, label in legend.classes.items()
    ]
    return pd.DataFrame(rows, columns=["variable", "code", "class", "colour"])
```

(If a packing of `1.0`/`0.0` is reported for bands carrying none, read `band.gs.packing` as None-for-absent per `core/dataarray.py:98`; the `test_leaves_out_empty_sections` expectation pins it.)

`VectorLayer.info`: `return {"Overview": {...the four facts as today...}}`. `CatalogLayer.info`: `sections = super().info()` then add `"Dates"` into `sections["Overview"]`.

Delete the old `RasterLayer.info` body (CRS/Size/Pixel/Variables/Dates/Nodata).

- [ ] **Step 4: Render sections in `docks.py`**

```python
def _info(shown: Layer | None, theme: str) -> list[pn.viewable.Viewable]:
    """Return a layer's sections, each under its title, then every attr, collapsed."""
    if shown is None:
        return []
    parts: list[pn.viewable.Viewable] = []

    # Each section: facts as a list, tables as tables
    for title, section in shown.info().items():
        parts.append(pmui.Typography(title, variant="subtitle2"))
        if isinstance(section, pd.DataFrame):
            parts.append(pn.widgets.Tabulator(
                section,
                show_index=False,
                disabled=True,
                layout="fit_data_table",
                theme=TABLE_THEME[theme],
                formatters={"colour": {"type": "color"}},
                sizing_mode="stretch_width",
            ))
        else:
            parts.append(pmui.Markdown(_facts(section), sizing_mode="stretch_width"))

    # Every attr, grouped and searchable, opened on demand
    attrs = [
        (section, str(name), str(value))
        for section, found in shown.attrs().items()
        for name, value in found.items()
    ]
    if attrs:
        table = pn.widgets.Tabulator(
            pd.DataFrame(attrs, columns=["section", "attribute", "value"]),
            groupby=["section"],
            hidden_columns=["section"],
            header_filters=True,
            show_index=False,
            disabled=True,
            layout="fit_columns",
            theme=TABLE_THEME[theme],
            configuration={"groupStartOpen": False, "columnDefaults": {"tooltip": True}},
            sizing_mode="stretch_width",
            max_height=420,
        )
        parts.append(pmui.Accordion(("All attributes", table), active=[], sizing_mode="stretch_width"))
    return parts
```

In `Properties.__init__`, bind the Info tab to both: `pn.bind(_info, explorer.param.selected, explorer.param.theme)`.

- [ ] **Step 5: Run the viz tests**

Run: `uv run pytest tests/geodata/viz -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/geosave/geodata/viz/explorer tests/geodata/viz/explorer
git commit -m "feat(explorer): raster Info tab with overview, grid, time, bands and classes"
```

---

### Task 7: Docs, API reference and browser check in both themes

**Files:**
- Modify: `docs/reference/public-api.md`, `docs/reference/generated/*.md` (regenerated)
- Modify: `docs/contributing/specs/2026-10-14-viz-holoviz-design.md` (status line)
- Create (throwaway, not committed): `$CLAUDE_JOB_DIR/tmp/probe/explorer_check.py`

- [ ] **Step 1: Regenerate the API reference**

Run: `uv run python scripts/api_reference.py && uv run python scripts/api_reference.py --check`
Expected: second run exits 0; `viz.elements.*` and `viz.map.Paint` gone from `public-api.md`.

- [ ] **Step 2: Full checks**

Run: `uv run pytest -q && uv run ruff check . && uv run pre-commit run --all-files`
Expected: all pass. Fix anything the viz change broke; leave unrelated failures and report them.

- [ ] **Step 3: Browser check, light and dark**

Write `$CLAUDE_JOB_DIR/tmp/probe/explorer_check.py` serving an explorer over the test fixtures' shapes (a 64×64 three-band raster with `write_rgb` and a legend band, a frame with a text column and a point, a two-date catalog):

```python
import numpy as np, geopandas as gpd, panel as pn
from shapely.geometry import box, Point
from odc.geo.geobox import GeoBox
from geosave.geodata.core.create import raster
from geosave.geodata.viz.explorer import Explorer

g = GeoBox.from_bbox((500_000, 9_000_000, 500_640, 9_000_640), crs="EPSG:32749", shape=(64, 64), tight=True)
v = np.random.default_rng(0).random((64, 64)).astype("float32")
scene = raster({n: (("y", "x"), v * k) for n, k in (("B04", 1), ("B03", 2), ("B02", 3))}, g)
scene["cover"] = (("y", "x"), (v > 0.5).astype("uint8"))
scene = scene.gs.write_legend({0: "water", 1: "trees"}, {0: "#419bdf", 1: "#397d49"}, target="cover").gs.write_rgb("B04", "B03", "B02")
fields = gpd.GeoDataFrame({"crop": ["rice", "palm", "rice"]}, geometry=[box(500_000, 9_000_000, 500_100, 9_000_100), box(500_200, 9_000_000, 500_300, 9_000_100), Point(500_500, 9_000_500)], crs="EPSG:32749")
Explorer(scene, fields).servable()
```

Serve it (`uv run panel serve ... --port 5079`, background) and drive Brave headless with Playwright (`executable_path="/opt/brave-bin/brave"`): load, wait, screenshot; switch the vector's "Colour by" to `crop`; wheel-zoom; tick the page's theme toggle; screenshot; record console errors and the bokeh x-range before and after the toggle.
Expected: 0 console errors; the zoom range unchanged across the toggle; in both screenshots sidebar text, legend names and table text are legible and the `crop` categories visible on the basemap. Look at both screenshots before claiming this step.

- [ ] **Step 4: Mark the spec built and commit**

Set the spec's status line to `Status: built, 2026-10-14.` and commit:

```bash
git add docs/reference docs/contributing/specs/2026-10-14-viz-holoviz-design.md
git commit -m "docs: viz reference follows the HoloViz rebuild"
```
