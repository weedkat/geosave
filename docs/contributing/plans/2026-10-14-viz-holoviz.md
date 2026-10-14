# viz on HoloViz's Intended Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `geodata/viz` on hvPlot, HoloViews' datashader operations and geosave's own accessors, then give the explorer a left sidebar, light and dark themes, and a raster Info tab.

**Architecture:** a raster draws two ways, as HoloViews does: one band through a `ColorRamp`, or RGB channels from `as_rgb` (a composite, or classes through `.gs.colorize()`); figures draw through hvPlot on matplotlib; each explorer layer renders a HoloViews `RGB` for the view on screen with `rasterize`/`shade`/`spread`, and the explorer stacks them with `stack` into one persistent bokeh plot. Rasters reach Web Mercator through `.gs.to_nan()` and `.gs.reproject`; class rasters colour through `.gs.colorize()`.

**Tech Stack:** hvPlot 0.12, HoloViews 1.23, datashader 0.19, spatialpandas 0.6, Panel 1.9 + panel-material-ui 0.16, xarray, geopandas, pytest, Playwright with Brave for browser checks.

**Spec:** `docs/contributing/specs/2026-10-14-viz-holoviz-design.md`

## Global Constraints

- No GeoViews or cartopy: vectors reach hvPlot/HoloViews as `spatialpandas.GeoDataFrame`, rasters through `.gs.reproject`.
- `hvplot` joins the core `dependencies` in `pyproject.toml`, beside `holoviews`.
- Names take the term a GIS user knows and say what a value holds: `ColorRamp`, `Stretch`, `as_rgb`, `category_colors`, `View.canvas`; nothing named `of`, nothing vague or verbose.
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

### Task 1: Colour rules — `ColorRamp`, `Stretch`, `as_rgb`, `category_colors`

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (dependencies)
- Modify: `src/geosave/geodata/viz/style.py` (add beside `Style`; Task 3 deletes `Style`)
- Test: `tests/geodata/viz/test_style.py`

**Interfaces:**
- Produces, in `geosave.geodata.viz.style`:
  - `DEFAULT_CMAP = "viridis"`; `type Theme = Literal["light", "dark"]`; `CATEGORY_PALETTE: dict[Theme, list[str]]`
  - `ColorRamp(cmap=DEFAULT_CMAP, clim=None)`, `.fit(values: np.ndarray) -> ColorRamp`, `.palette -> list[str]`
  - `Stretch(clim=None)`, `.fit(values: np.ndarray) -> Stretch`, `.apply(channels: ndarray | DataArray) -> same type`
  - `as_rgb(raster: xr.Dataset, variables: tuple[str, ...]) -> xr.DataArray | None`
  - `category_colors(values: pd.Series, theme: Theme = "light") -> dict[object, str]`

- [ ] **Step 1: Add hvplot**

Run: `uv add "hvplot>=0.12"`
Expected: `"hvplot>=0.12"` in `pyproject.toml` `dependencies`; `uv.lock` updated.

- [ ] **Step 2: Write the failing tests**

Append to `tests/geodata/viz/test_style.py` (the existing `Style` tests stay until Task 3):

```python
import colorcet as cc
import pandas as pd

from geosave.geodata.core.create import raster as build_raster
from geosave.geodata.viz.style import ColorRamp, Stretch, as_rgb, category_colors
from tests.geodata.viz.conftest import grid


class TestColorRamp:
    def test_fits_the_finite_range(self) -> None:
        assert ColorRamp().fit(np.array([np.nan, 1.0, 3.0])).clim == (1.0, 3.0)

    def test_keeps_limits_already_set(self) -> None:
        assert ColorRamp(clim=(0.0, 1.0)).fit(np.array([5.0])).clim == (0.0, 1.0)

    def test_refuses_nothing_to_draw(self) -> None:
        with pytest.raises(ValueError, match="nothing to draw"):
            ColorRamp().fit(np.array([np.nan]))

    def test_palette_runs_low_to_high(self) -> None:
        palette = ColorRamp(["#000000", "#ffffff"]).palette

        assert len(palette) == 256 and palette[0] == "#000000" and palette[-1] == "#ffffff"


class TestStretch:
    def test_fits_the_middle_percentiles(self) -> None:
        clim = Stretch().fit(np.linspace(0.0, 1.0, 101)).clim

        assert clim == pytest.approx((0.02, 0.98))

    def test_eight_bit_draws_as_stored(self) -> None:
        assert Stretch().fit(np.zeros(4, "uint8")).clim == (0.0, 255.0)

    def test_refuses_flat_channels_and_inverted_limits(self) -> None:
        with pytest.raises(ValueError, match="flat"):
            Stretch().fit(np.ones(10))
        with pytest.raises(ValueError, match="inverted"):
            Stretch((1.0, 0.0)).fit(np.ones(10))

    def test_apply_keeps_a_dataarray_and_its_nan(self) -> None:
        stretched = Stretch((0.0, 10.0)).apply(band([[5.0, 20.0, np.nan]]))

        assert isinstance(stretched, xr.DataArray)
        assert stretched.values[0, :2].tolist() == [0.5, 1.0]
        assert np.isnan(stretched.values[0, 2])


class TestCategoryColors:
    def test_colours_categories_in_sorted_order(self) -> None:
        colors = category_colors(pd.Series(["rice", "palm", None, "rice"]))

        assert colors == {"palm": cc.glasbey_dark[0], "rice": cc.glasbey_dark[1]}

    def test_dark_theme_takes_light_hues(self) -> None:
        assert category_colors(pd.Series(["rice"]), "dark") == {"rice": cc.glasbey_light[0]}


class TestAsRgb:
    def raster(self) -> xr.Dataset:
        codes = np.tile(np.array([[0, 1], [8, 1]], "uint8"), (32, 32))
        values = np.random.default_rng(0).random((64, 64)).astype("float32")
        raster = build_raster(
            {"B04": (("y", "x"), values), "B03": (("y", "x"), values), "B02": (("y", "x"), values),
             "cover": (("y", "x"), codes)},
            grid(),
        )
        return raster.gs.write_legend({1: "trees", 8: "snow"}, {1: "#397d49", 8: "#b39fe1"}, target="cover")

    def test_three_variables_compose(self) -> None:
        rgb = as_rgb(self.raster(), ("B04", "B03", "B02"))

        assert rgb is not None and list(rgb["band"].values) == ["B04", "B03", "B02"]

    def test_classes_take_their_legends_colours(self) -> None:
        rgb = as_rgb(self.raster(), ("cover",))

        assert rgb is not None and rgb.sizes["band"] == 3
        # Code 0 names no class, so it draws nothing.
        assert np.isnan(rgb.values[:, 0, 0]).all()
        assert rgb.values[:, 0, 1] == pytest.approx([0x39 / 255, 0x7D / 255, 0x49 / 255])

    def test_a_band_without_classes_draws_through_a_ramp(self) -> None:
        assert as_rgb(self.raster(), ("B04",)) is None
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/geodata/viz/test_style.py -q`
Expected: FAIL with `ImportError: cannot import name 'ColorRamp'`.

- [ ] **Step 4: Implement in `style.py`**

Module docstring: `"""How raster values and table columns become colour, one rule for figures and maps."""`. Add imports `import colorcet as cc`, `import pandas as pd` and `import xarray as xr` under `TYPE_CHECKING`. Under `DEFAULT_CMAP`:

```python
type Theme = Literal["light", "dark"]

# Category colours readable on each theme's basemap: dark hues on light, light hues on dark.
CATEGORY_PALETTE: dict[Theme, list[str]] = {"light": cc.glasbey_dark, "dark": cc.glasbey_light}


@dataclass(frozen=True)
class ColorRamp:
    """A colormap between two limits: how one band or numeric column draws.

    Attributes:
        cmap: Matplotlib, colorcet or bokeh name, or a list of colours.
        clim: Values at the ramp's ends. None reads them from the values drawn.

    Examples:
        >>> ColorRamp("YlGn").fit(ds["ndvi"].values).clim
        (-0.12, 0.91)
    """

    cmap: str | Sequence[str] = DEFAULT_CMAP
    clim: tuple[float, float] | None = None

    def fit(self, values: np.ndarray) -> ColorRamp:
        """Fill in limits spanning the finite values; limits already set stay.

        Raises:
            ValueError: Every value is nan.
        """
        if self.clim is not None:
            return self
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError("every pixel is nan or nodata; nothing to draw")
        return replace(self, clim=(float(finite.min()), float(finite.max())))

    @property
    def palette(self) -> list[str]:
        """Return the ramp as 256 hex colours, low to high."""
        return process_cmap(self.cmap, ncolors=256)


@dataclass(frozen=True)
class Stretch:
    """The limits a composite's channels stretch between onto `[0, 1]`.

    Attributes:
        clim: Value drawn dark and value drawn full. None reads them from the channels.

    Examples:
        >>> Stretch((0.0, 0.3)).apply(ds[["B04", "B03", "B02"]].gs.to_array())
    """

    clim: tuple[float, float] | None = None

    def fit(self, values: np.ndarray) -> Stretch:
        """Fill in limits: `uint8` as stored, else the 2-98 percentile, so dark reflectance fills the display.

        Raises:
            ValueError: The limits set are empty or inverted, every value is
                nan, or the channels are flat.
        """
        if self.clim is not None:
            if not self.clim[1] > self.clim[0]:
                raise ValueError(f"stretch bounds {self.clim} are empty or inverted")
            return self
        if values.dtype == np.uint8:
            return Stretch((0.0, 255.0))
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError("every pixel is nan or nodata; nothing to draw")
        low, high = (float(v) for v in np.percentile(finite, (2, 98)))
        if not high > low:
            raise ValueError(f"channels are flat near {low}; pass clim=(low, high)")
        return Stretch((low, high))

    def apply[T: (np.ndarray, xr.DataArray)](self, channels: T) -> T:
        """Map channels onto `[0, 1]`, clipping outside the limits; nan stays nan."""
        low, high = self.fit(np.asarray(channels)).clim  # type: ignore[misc]
        return np.clip((channels.astype("float32") - low) / (high - low), 0.0, 1.0)


def as_rgb(raster: xr.Dataset, variables: tuple[str, ...]) -> xr.DataArray | None:
    """Return the colour channels a selection draws as, None for one band a colour ramp draws.

    Three variables compose red, green and blue. One band whose `Legend`
    lists classes is colour already: each pixel takes its class's colour.

    Args:
        raster: Raster holding the variables.
        variables: One variable, or three drawn red, green, blue.

    Returns:
        `(*axes, band, y, x)` channels: a composite's as stored, classes'
        valued in `[0, 1]` and nan where no class is named. None for a band
        without classes.

    Raises:
        ValueError: A class carries no colour.

    Examples:
        >>> as_rgb(ds, ("B04", "B03", "B02")).sizes["band"]
        3
        >>> as_rgb(ds, ("ndvi",)) is None
        True
    """
    if len(variables) == 3:
        return raster[list(variables)].gs.to_array()
    band = raster[variables[0]]
    legend = band.gs.legend
    if legend is not None and legend.classes:
        return band.gs.colorize()
    return None


def category_colors(values: pd.Series, theme: Theme = "light") -> dict[object, str]:
    """Colour each category of a text column from the theme's glasbey palette.

    Categories take colours in sorted order, so one keeps its colour across
    geometry types and redraws; missing values take none.

    Returns:
        {category: "#rrggbb"}

    Examples:
        >>> list(category_colors(fields["crop"]))
        ['palm', 'rice']
    """
    palette = CATEGORY_PALETTE[theme]
    names = sorted(values.dropna().unique(), key=str)
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}
```

- [ ] **Step 5: Run the style tests**

Run: `uv run pytest tests/geodata/viz/test_style.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/geosave/geodata/viz/style.py tests/geodata/viz/test_style.py
git commit -m "feat(viz): ColorRamp, Stretch, as_rgb and category_colors"
```

---

### Task 2: Static figures through hvPlot

**Files:**
- Modify: `src/geosave/geodata/viz/plot.py`
- Modify: `src/geosave/geodata/viz/map.py` (add `GEOMETRY_TYPES`, `drawable`, `split_by_geometry`)
- Modify: `src/geosave/geodata/core/dataset.py` (`plot`), `src/geosave/geodata/core/dataarray.py` (`plot`)
- Test: `tests/geodata/viz/test_plot.py`, `tests/geodata/viz/test_map.py`

**Interfaces:**
- Consumes: `ColorRamp`, `Stretch`, `as_rgb`, `DEFAULT_CMAP` (Task 1).
- Produces:
  - `plot.figure(raster: xr.Dataset, variables: tuple[str, ...], *, cmap=None, clim=None, cols=4, title=None, xlabel=None) -> hv.Element | hv.Layout`
  - `plot.outline(frame, *, color, alpha)` unchanged.
  - `map.GEOMETRY_TYPES = ("Polygon", "LineString", "Point")`; `map.drawable(frame) -> gpd.GeoDataFrame`; `map.split_by_geometry(frame) -> dict[str, gpd.GeoDataFrame]` keyed by `GEOMETRY_TYPES` in that order, only types present, original index kept.

- [ ] **Step 1: Write the failing tests**

`tests/geodata/viz/test_map.py`, add:

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

`tests/geodata/viz/test_plot.py`: replace `class TestClassMap` with

```python
class TestClassMap:
    def test_draws_each_class_in_its_colour(self, labels: xr.Dataset) -> None:
        drawn = labels.gs.plot("landcover")
        (image,) = drawn.select(hv.RGB)  # the map under the legend

        red = image.dimension_values("R", flat=False)
        # Pixel (0, 0) of the fixture is water, #419bdf.
        assert red[0, 0] == pytest.approx(0x41 / 255)

    def test_names_every_class_in_a_legend(self, labels: xr.Dataset) -> None:
        drawn = labels.gs.plot("landcover")

        names = [x.label for x in drawn if isinstance(x, hv.Points)]
        assert names == ["water", "trees", "snow"]

    def test_absent_pixels_draw_as_no_class(self) -> None:
        codes = np.array(
            [[0, 1, 8, 0], [1, 8, 0, 1], [8, 0, 1, 8], [0, 1, 8, 0]], dtype="uint8"
        )
        raster = build_raster({"cover": (("y", "x"), codes)}, geobox(), nodata=0)
        raster = raster.gs.write_legend(
            {1: "trees", 8: "snow"}, {1: "#397d49", 8: "#b39fe1"}, target="cover"
        )

        (image,) = raster.gs.plot("cover").select(hv.RGB)

        assert np.isnan(image.dimension_values("R")).any()

    def test_refuses_a_colormap_or_limits(self, labels: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="drop cmap"):
            labels.gs.plot("landcover", cmap="gray")
        with pytest.raises(ValueError, match="clim cannot apply"):
            labels.gs.plot("landcover", clim=(0, 1))

    def test_uncoloured_class_refuses(self) -> None:
        raster = build_raster(
            {"cover": (("y", "x"), np.zeros((4, 4), dtype="uint8"))}, geobox()
        )
        raster = raster.gs.write_legend(
            {0: "water", 1: "trees"}, {0: "#419bdf"}, target="cover"
        )
        with pytest.raises(ValueError, match="carry no colour"):
            raster.gs.plot("cover")
```

In `TestConstraints.test_dark_reflectance_is_stretched_to_fill_the_display`, assert `float(np.nanmax(drawn.dimension_values("R"))) == 1.0`. In `TestOutput.test_opts_reach_hvplot`, keep the assertion (`cmap == "RdYlGn"`). Add to `TestVariables`:

```python
    def test_a_composite_refuses_a_colormap(self, rgb: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="drop cmap"):
            rgb.gs.plot(cmap="gray")

    def test_a_stacked_array_draws_its_bands_as_colour(self, rgb: xr.Dataset) -> None:
        assert isinstance(rgb.gs.to_array().gs.plot(), hv.RGB)
```

(If `drawn.select(hv.RGB)` returns the whole overlay rather than a tuple of elements, read the map with `drawn.get(0)`; the overlay's first layer is the RGB.)

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/viz/test_plot.py tests/geodata/viz/test_map.py::TestSplitByGeometry -q`
Expected: FAIL — no `split_by_geometry`; classes still draw as an Image.

- [ ] **Step 3: Add the geometry split to `map.py`**

Move `import geopandas as gpd` out of `TYPE_CHECKING`. Under the colour constants:

```python
# Geometry types a map or figure draws, in drawing order. hvPlot and HoloViews
# draw one type per element, a Multi* geometry with its single type.
GEOMETRY_TYPES = ("Polygon", "LineString", "Point")


def drawable(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the rows of a frame that carry geometry a figure or map can draw.

    Raises:
        ValueError: The frame holds no geometry, or a GeometryCollection.
    """
    shown = frame[~(frame.geometry.is_empty | frame.geometry.isna())]
    if shown.empty:
        raise ValueError("nothing to draw: the frame holds no geometry")
    if unknown := set(shown.geom_type.str.removeprefix("Multi")) - set(GEOMETRY_TYPES):
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
    return {kind: shown[types == kind] for kind in GEOMETRY_TYPES if (types == kind).any()}
```

- [ ] **Step 4: Rewrite `plot.py`**

Module docstring: `"""Static figures behind `.gs.plot`, drawn through hvPlot and rendered with matplotlib.` + the existing `hv.save`/`+`/`*` paragraph.

Imports:

```python
from __future__ import annotations

import operator
from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import holoviews as hv
import hvplot.pandas  # noqa: F401  — registers .hvplot on spatialpandas frames
import hvplot.xarray  # noqa: F401  — registers .hvplot on xarray
import numpy as np
import spatialpandas as spd
import xarray as xr
from matplotlib.colors import to_hex, to_rgba

from geosave.geodata.conventions import BAND_DIMENSION, SPATIAL_DIMENSIONS, TIME_COORDINATE
from geosave.geodata.viz.map import split_by_geometry
from geosave.geodata.viz.style import DEFAULT_CMAP, ColorRamp, Stretch, as_rgb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geosave.geodata.attrs import Legend

hv.extension("matplotlib")
```

`figure`:

```python
def figure(
    raster: xr.Dataset,
    variables: tuple[str, ...],
    *,
    cmap: str | Sequence[str] | None = None,
    clim: tuple[float, float] | None = None,
    cols: int = 4,
    title: str | None = None,
    xlabel: str | None = None,
) -> hv.Element | hv.Layout:
    """Draw variables of a raster: one band through a colour ramp, else as colour.

    Three variables draw red, green and blue, stretched by `clim`. A band
    whose `Legend` lists classes draws each class in its colour, named in a
    legend beside the map. Values draw as stored, nodata included.

    Args:
        raster: Raster spanning the spatial pair and, optionally, `time`.
        variables: One variable, or three drawn red, green, blue.
        cmap: Colormap of one band. Refused where the variables draw as colour.
        clim: Ramp limits of one band, or a composite's stretch. Refused for classes.
        cols: Columns a `time` axis lays panels into. Ignored otherwise.
        title: Panel title above the axes. None suppresses the title.
        xlabel: Caption appended below each panel's timestamp. None adds no caption.

    Returns:
        Image, RGB or RGB with a legend over the raster's grid, or a Layout
        of one panel per `time` value, `cols` wide.

    Raises:
        ValueError: `cmap` accompanies colour, `clim` accompanies classes, a
            class carries no colour, or a composite's pixels are all nan or flat.

    Examples:
        >>> figure(ds, ("ndvi",), cmap="YlGn")
        >>> figure(ds, ("B04", "B03", "B02"), clim=(0.0, 0.3))
    """
    y, x = SPATIAL_DIMENSIONS
    rgb = as_rgb(raster, variables)

    # One band draws through a colour ramp
    if rgb is None:
        ramp = ColorRamp(cmap or DEFAULT_CMAP, clim)
        band = raster[variables[0]]
        band = band.rename(band.name or "value")
        drawn = _by_date(
            band,
            lambda frame: frame.hvplot.image(
                x=x, y=y, cmap=ramp.cmap, clim=ramp.clim, colorbar=True, data_aspect=1
            ),
        )
        return _captioned(drawn, raster, cols=cols, title=title, xlabel=xlabel)

    # Colour carries no quantity, and a class's colour is its legend's
    legend = raster[variables[0]].gs.legend if len(variables) == 1 else None
    if cmap is not None:
        raise ValueError("these variables draw as colour rather than a quantity; drop cmap")
    if legend is not None and clim is not None:
        raise ValueError(
            "the band lists classes, so clim cannot apply; its palette is Legend.color_map"
        )

    # A composite stretches once for every date, so dates compare
    if legend is None:
        rgb = Stretch(clim).apply(rgb)
    drawn = _by_date(
        rgb, lambda frame: frame.hvplot.rgb(x=x, y=y, bands=BAND_DIMENSION, data_aspect=1)
    )
    if legend is not None:
        drawn = (drawn * _swatches(legend)).opts(
            hv.opts.Overlay(legend_position="right", backend="matplotlib")
        )
    return _captioned(drawn, raster, cols=cols, title=title, xlabel=xlabel)


def _by_date(
    array: xr.DataArray, draw: Callable[[xr.DataArray], hv.Element]
) -> hv.Element | hv.HoloMap:
    """Draw an array once, or once per date in a HoloMap keyed by `time`."""
    if TIME_COORDINATE not in array.dims:
        return draw(array)
    return hv.HoloMap(
        {time: draw(array.sel({TIME_COORDINATE: time})) for time in array[TIME_COORDINATE].values},
        kdims=[TIME_COORDINATE],
    )


def _swatches(legend: Legend) -> hv.Overlay:
    """Return a legend entry per class: its name beside a square of its colour."""
    colors = legend.color_map or {}
    # A point off the axes carries each entry; the legend shows its marker
    return hv.Overlay(
        [
            hv.Points([(np.nan, np.nan)], label=name).opts(
                color=colors[code], marker="s", s=80, backend="matplotlib"
            )
            for code, name in (legend.classes or {}).items()
        ]
    )
```

`Stretch.apply` fits from the whole array, so one stretch serves every date.

`outline` (signature and docstring unchanged; "geometry families" → "geometry types"):

```python
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
    return cast("hv.Element", reduce(operator.mul, drawn).opts(aspect="equal", backend="matplotlib"))
```

Delete `_styled`. `_captioned` keeps its body; its `array: xr.DataArray` parameter becomes `raster: xr.Dataset` (it only reads the `time` coordinate).

- [ ] **Step 5: Point the accessors at the new `figure`**

`dataset.py` `plot`, from `from geosave.geodata.viz.plot import figure` on:

```python
        from geosave.geodata.viz.plot import figure

        selected = self._drawn(variables)
        place = self.anchor.locate() if xlabel is None else None
        caption = place.to_address() if place is not None else xlabel
        return figure(
            self._data, selected, cmap=cmap, clim=clim, cols=cols, title=title, xlabel=caption
        )
```

`dataarray.py` `plot`, replacing its body after the docstring (docstring adds: "An array stacking three bands on `band` draws them red, green, blue."):

```python
        # A stacked array draws its bands; a lone band draws under its name
        array = self._data if self._data.name is not None else self._data.rename("value")
        raster = array.gs.to_raster()
        return raster.gs.plot(
            *(str(name) for name in raster.data_vars),
            cmap=cmap, clim=clim, cols=cols, title=title, xlabel=xlabel,
        )
```

- [ ] **Step 6: Run the figure tests**

Run: `uv run pytest tests/geodata/viz/test_plot.py tests/geodata/viz/test_map.py tests/geodata/core -q`
Expected: PASS. If hvPlot adds a title a caption test rejects, pass `title=""` in the hvPlot call rather than post-processing.

- [ ] **Step 7: Commit**

```bash
git add src/geosave/geodata/viz/plot.py src/geosave/geodata/viz/map.py src/geosave/geodata/core/dataset.py src/geosave/geodata/core/dataarray.py tests/geodata/viz/test_plot.py tests/geodata/viz/test_map.py
git commit -m "refactor(viz): figures draw through hvPlot; classes draw as colour with a legend"
```

---

### Task 3: Layers render through HoloViews operations

**Files:**
- Modify: `src/geosave/geodata/viz/map.py`
- Modify: `src/geosave/geodata/viz/style.py` (delete `Style`, `Kind`)
- Modify: `src/geosave/geodata/viz/explorer/layers.py`
- Modify: `src/geosave/geodata/viz/explorer/app.py` (`_draw`, `_swipe`)
- Delete: `src/geosave/geodata/viz/elements.py`, `tests/geodata/viz/test_elements.py`
- Test: `tests/geodata/viz/test_map.py`, `tests/geodata/viz/test_style.py`, `tests/geodata/viz/explorer/test_layers.py`, `tests/geodata/viz/explorer/test_app.py`

**Interfaces:**
- Consumes: `ColorRamp`, `Stretch`, `as_rgb`, `category_colors`, `Theme` (Task 1); `map.drawable`, `map.split_by_geometry` (Task 2).
- Produces:
  - `map.View.canvas -> dict` (keywords putting a datashader operation on the view's grid); `map.View.bounds -> (left, bottom, right, top)`.
  - `map.Overview(raster: xr.Dataset, *, resampling: Resampling | Mapping[str, Resampling], size=2048)`; `.view(view) -> xr.Dataset` on `view.geobox`; `.value_at(x, y) -> dict[str, float] | None`; `.full`, `.coarse` Datasets.
  - `map.blank(view) -> hv.RGB`; `map.rgb_element(channels, alpha) -> hv.RGB`; `map.rgba(image) -> np.ndarray` `(rows, columns, 4)` uint8, north up.
  - `map.OUTLINE: dict[Theme, str]`, `map.THEME_BASEMAP: dict[Theme, str]`, `map.CMAPS`; `COLOR`, `HIGHLIGHT`, `BASEMAPS`, `tiles` kept.
  - `Layer.theme = param.Selector("light", ["light", "dark"])`; `Layer.image(view) -> hv.RGB | None`.
  - `RasterLayer.ramp -> ColorRamp` (bands), `RasterLayer.stretch -> Stretch` (composites); `VectorLayer.colors -> dict | None`, `VectorLayer.ramp -> ColorRamp | None`.
  - Deleted: `style.Style`, `map.Paint`, `aggregate`, `spread`, `stack`, `composite`, `on_web`, `plain_columns`, `_hoverable`, `elements.*`.

- [ ] **Step 1: Write the failing tests**

`tests/geodata/viz/test_style.py`: delete every test of `Style` (`TestOf`, `TestResolve`, `test_positions_leave_no_gap_for_sparse_codes`, `TestRgba`) and the `LEGEND`/`composite` helpers they alone used.

`tests/geodata/viz/test_map.py`: delete `class TestShapes`, `test_composite_lays_the_last_image_on_top`, and the `elements`, `datashader`, `pandas`, `Style` imports. Rewrite `TestOverview` against a Dataset:

```python
def lazy_raster(size: int = 64) -> xr.Dataset:
    values = da.random.random((size, size), chunks=size // 4).astype("float32")
    return build_raster({"ndvi": (("y", "x"), values)}, grid(size))


class TestOverview:
    def test_reads_lazily_beyond_its_coarse_copy(self) -> None:
        overview = webmap.Overview(lazy_raster(256), resampling="average", size=64)

        assert isinstance(overview.full["ndvi"].data, da.Array)
        assert max(overview.coarse["ndvi"].shape) == 64
        assert isinstance(overview.coarse["ndvi"].data, np.ndarray)

    def test_warps_a_raster_in_memory_lazily(self) -> None:
        overview = webmap.Overview(lazy_raster(256).compute(), resampling="average", size=64)

        # Only the windows a map shows are warped at full detail.
        assert isinstance(overview.full["ndvi"].data, da.Array)

    def test_wide_views_read_the_coarse_copy(self) -> None:
        overview = webmap.Overview(lazy_raster(256), resampling="average", size=64)
        left, bottom, right, top = overview.bounds

        assert overview.window((left, right), (bottom, top)) is overview.coarse

    def test_small_views_read_full_detail(self) -> None:
        overview = webmap.Overview(lazy_raster(256), resampling="average", size=64)
        left, bottom, right, top = overview.bounds
        step = (right - left) / 8

        window = overview.window((left, left + step), (top - step, top))

        assert isinstance(window["ndvi"].data, np.ndarray)
        assert 0 < window.sizes["x"] <= 256 // 8 + 2

    def test_view_draws_on_the_screen_grid(self) -> None:
        overview = webmap.Overview(lazy_raster(), resampling="average")
        left, bottom, right, top = overview.bounds

        drawn = overview.view(webmap.View((left, right), (bottom, top), 40, 30))

        # A square raster fills the screen's height; its pixels stay square.
        assert drawn["ndvi"].shape == (30, 30)

    def test_class_codes_never_blend(self) -> None:
        codes = np.tile(np.array([[0, 8], [8, 1]], "uint8"), (32, 32))
        raster = build_raster({"cover": (("y", "x"), codes)}, grid())
        raster = raster.gs.write_legend(
            {0: "water", 1: "trees", 8: "snow"},
            {0: "#419bdf", 1: "#397d49", 8: "#b39fe1"},
            target="cover",
        )
        overview = webmap.Overview(raster, resampling="nearest")
        left, bottom, right, top = overview.bounds

        drawn = overview.view(webmap.View((left, right), (bottom, top), 17, 13))["cover"].values

        assert set(drawn[~np.isnan(drawn)]) <= {0.0, 1.0, 8.0}

    def test_nodata_reads_as_nan(self) -> None:
        values = np.full((64, 64), -9999.0, "float32")
        values[:, :32] = 1.0
        raster = build_raster({"ndvi": (("y", "x"), values)}, grid(), nodata=-9999.0)

        overview = webmap.Overview(raster, resampling="average")

        assert np.isnan(overview.coarse["ndvi"].values).any()
        assert np.nanmax(overview.coarse["ndvi"].values) == 1.0


def test_rgb_element_is_clear_where_a_channel_is_nan() -> None:
    view = webmap.View((0.0, 2.0), (0.0, 1.0), 2, 1)
    channels = xr.DataArray(
        np.array([[[1.0, np.nan]], [[0.0, 0.0]], [[0.0, 0.0]]]),
        dims=("band", "y", "x"),
        coords={"y": view.geobox.coords["y"].values, "x": view.geobox.coords["x"].values},
    )

    rgba = webmap.rgba(webmap.rgb_element(channels, 0.5))

    assert list(rgba[0, 0]) == [255, 0, 0, 128] and rgba[0, 1, 3] == 0


def test_blank_covers_the_view() -> None:
    view = webmap.View((0.0, 100.0), (0.0, 10.0), 50, 50)

    assert webmap.rgba(webmap.blank(view)).shape == (5, 50, 4)
```

`tests/geodata/viz/explorer/test_layers.py`:

```python
def drawn(image) -> int:
    return int((webmap.rgba(image)[..., 3] > 0).sum())
```

Wrap every raw image in `webmap.rgba(...)` where a test indexes channels (`test_draws_a_band_or_a_composite_in_colour`, `test_steps_through_its_dates`, `test_outlines_the_selection_bright`). Replace style assertions:

```python
    def test_offers_a_colormap_only_where_it_colours(self, scene: xr.Dataset) -> None:
        shown = RasterLayer(scene, "B04", cmap="magma")

        assert shown.ramp.cmap == "magma"
        assert shown.param.cmap.precedence > 0
        shown.variable = ("cover",)
        assert shown.param.cmap.precedence < 0

    def test_dates_share_colour_limits(self, monthly: xr.Dataset) -> None:
        shown = RasterLayer(monthly, "B04")

        before = shown.ramp.clim
        shown.time = shown.dates[1]

        assert shown.ramp.clim == before

    def test_takes_a_band(self, scene: xr.Dataset) -> None:
        shown = RasterLayer(scene["cover"])

        assert shown.name == "cover"
        assert shown.key().swatches == {"water (0)": "#419bdf", "trees (1)": "#397d49"}
```

In `test_colours_one_category_alike_in_every_family`, expect `palm, rice = cc.glasbey_dark[:2]`. Add to `TestVectorLayer`:

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

`tests/geodata/viz/explorer/test_app.py` `TestSwipe`, add:

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

        swiped = webmap.rgba(explorer._swipe(view))
        left = webmap.rgba(raster.image(view, sides[0][1]))

        # Left of the cut line, the swipe shows the left side exactly as drawn.
        assert (swiped[:, :-2] == left[:, :-2]).all()
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/viz -q`
Expected: FAIL — `Overview` still takes an array and a Style; `rgb_element`, `blank`, `rgba` missing.

- [ ] **Step 3: Rewrite `map.py`**

Module docstring:

```python
"""Draw on a Web Mercator map: the view on screen, rasters read at its detail, basemaps.

A map redraws on every pan and zoom, so it needs a live Python process: a
notebook kernel or `panel serve`. Each layer renders at the screen's size
through HoloViews' datashader operations, rasters regridded by GDAL first.
"""
```

Imports: `Mapping`, `cast`, `geopandas as gpd`, `holoviews as hv`, `holoviews.plotting.bokeh  # noqa: F401  — registers bokeh's options`, `numpy as np`, `xarray as xr`, `tile_sources`, `GeoBox`, `BAND_DIMENSION`, `SPATIAL_DIMENSIONS`, `Theme`; `Resampling` from `geosave.geodata.transform.warp` under `TYPE_CHECKING`. Drop `colorcet`, `pandas`, `process_cmap`, `odc.geo.xr`, `elements`, `Style`.

Constants:

```python
# Colour of features no column colours, and of selected features.
COLOR = "#1f77b4"
HIGHLIGHT = "#ffd400"
# Colour of the boundaries of polygons a column colours, visible on each theme's basemap.
OUTLINE: dict[Theme, str] = {"light": "#333333", "dark": "#e0e0e0"}
# Basemap a map opens on in each theme.
THEME_BASEMAP: dict[Theme, str] = {"light": "Light", "dark": "Dark"}
# Offered in the colormap picker; any matplotlib, colorcet or bokeh name is accepted.
CMAPS = ["viridis", "cividis", "magma", "RdYlGn", "YlGn", "Spectral", "gray"]
```

`View`: keep `pixel`, `geobox`; replace `canvas` with

```python
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return `(left, bottom, right, top)` in metres."""
        (left, right), (bottom, top) = self.x_range, self.y_range
        return left, bottom, right, top

    @property
    def canvas(self) -> dict[str, object]:
        """Return the keywords that put a datashader operation on this view's grid."""
        rows, columns = self.geobox.shape
        return {
            "x_range": self.x_range,
            "y_range": self.y_range,
            "width": columns,
            "height": rows,
            "dynamic": False,
        }
```

`Overview`:

```python
class Overview:
    """Variables of a raster on Web Mercator, read at the detail a view needs.

    Zoomed out, a view draws a small copy computed once; zoomed in, it reads
    only the window it shows at full detail, so a lazy raster is never read
    whole after the first pass.

    Args:
        raster: The variables a layer shows, holding one date.
        resampling: GDAL kernel, or one per variable; class codes take
            `nearest`, which `.gs.reproject` insists on.
        size: Longest side of the zoomed-out copy, in pixels.

    Examples:
        >>> overview = Overview(ds[["ndvi"]], resampling="average")
        >>> overview.view(View(x_range, y_range))["ndvi"].shape
        (600, 800)
    """

    def __init__(
        self,
        raster: xr.Dataset,
        *,
        resampling: Resampling | Mapping[str, Resampling],
        size: int = 2048,
    ) -> None:
        # Nodata reads as nan, so a map leaves it clear
        values = raster.gs.to_nan().astype("float32").gs.write_nodata(float("nan"))
        # A raster in memory warps lazily too, so a map warps only the windows it shows
        y, x = SPATIAL_DIMENSIONS
        if all(band.chunks is None for band in values.data_vars.values()):
            values = values.chunk({y: size, x: size})

        self.resampling = resampling
        self.full = values.gs.reproject(WEB_MERCATOR, resampling=resampling)
        grid = self.full.gs.geobox
        self.coarse = values.gs.reproject(
            grid.zoom_to(min(size, max(grid.shape))), resampling=resampling
        ).compute()
        self.size = size
```

Keep `bounds` (read `self.coarse.gs.geobox.extent.boundingbox`) and `window` (unchanged; it slices the Dataset). Replace `value_at` and `view`:

```python
    def value_at(self, x: float, y: float) -> dict[str, float] | None:
        """Read each variable's full-detail value under a Web Mercator point, None outside."""
        left, bottom, right, top = self.bounds
        if not (left <= x <= right and bottom <= y <= top):
            return None
        ys, xs = SPATIAL_DIMENSIONS
        pixel = self.full.sel({xs: x, ys: y}, method="nearest").compute()
        return {str(name): float(value) for name, value in pixel.data_vars.items()}

    def view(self, view: View) -> xr.Dataset:
        """Return the values a view shows, one per screen pixel, north up, nan outside the raster."""
        window = self.window(view.x_range, view.y_range)
        # Class codes take the most common code under a screen pixel, never a blend
        kernels = {
            name: "mode" if self._kernel(name) == "nearest" else "average"
            for name in window.data_vars
        }
        # GDAL regrids to the screen, so no first-use compile delays a map
        return window.gs.reproject(view.geobox, resampling=kernels)

    def _kernel(self, name: str) -> Resampling:
        """Return the kernel one variable warps with."""
        if isinstance(self.resampling, str):
            return self.resampling
        return self.resampling.get(name, self.resampling.get("*", "average"))
```

Image helpers (replace `on_web` … `_hoverable`; keep `tiles`):

```python
def blank(view: View) -> hv.RGB:
    """Return an empty image covering a view."""
    rows, columns = view.geobox.shape
    return hv.RGB(np.zeros((rows, columns, 4), np.uint8), bounds=view.bounds)


def rgb_element(channels: xr.DataArray, alpha: float) -> hv.RGB:
    """Return three channels valued in `[0, 1]` as an RGB element, clear where any is nan.

    Args:
        channels: `(band, y, x)` on a view's grid, as `as_rgb` or `Stretch.apply` return them.
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


def rgba(image: hv.RGB) -> np.ndarray:
    """Return an RGB element's `(rows, columns, 4)` bytes, north up."""
    channels = np.dstack([image.dimension_values(d, flat=False) for d in image.vdims])
    if channels.dtype != np.uint8:
        channels = np.round(np.nan_to_num(channels) * 255).astype(np.uint8)
    # HoloViews hands rows south to north when y ascends
    ys = image.dimension_values(1, expanded=False)
    return channels[::-1] if len(ys) > 1 and ys[0] < ys[-1] else channels
```

(Step 7 confirms `rgba`'s orientation through `test_swipe_keeps_orientation`; if HoloViews already returns rows north up, drop the flip and its comment.)

- [ ] **Step 4: Delete `Style` from `style.py`**

Delete `type Kind`, the `Style` class and the `ListedColormap`/`to_rgba_array`/`BAND_DIMENSION`/`Legend`/`Colormap`/`Mapping` imports only it used. The module now holds `DEFAULT_CMAP`, `Theme`, `CATEGORY_PALETTE`, `ColorRamp`, `Stretch`, `as_rgb`, `category_colors`.

- [ ] **Step 5: Raster layers in `layers.py`**

Imports: drop `elements`, `dataclass(replace)` if unused, `Style`; add `holoviews as hv`, `spatialpandas as spd`, `from holoviews.operation.datashader import rasterize, shade, spread, stack`, `from geosave.geodata.viz.style import DEFAULT_CMAP, ColorRamp, Stretch, Theme, as_rgb, category_colors`; `CMAPS` from `webmap`. Constants:

```python
# Columns every shape carries: the position of its frame row, and its larger side in metres.
ROW = "row"
SIZE = "size"
```

`Layer` gains

```python
    theme = param.Selector(default="light", objects=["light", "dark"], precedence=-1)
```

and `image`'s docstring returns "RGB element on the view's grid, None where the layer lies off the view."

`RasterLayer`: `self._limits: dict[tuple[str, ...], tuple[float, float]]` keeps one ramp or stretch per variable across dates. Delete the `style` property and `_style`; replace `_offer_cmap`, `key`, `identify`, `overview`, `image`, and add `_legend`, `ramp`, `stretch`:

```python
    @param.depends("variable", watch=True, on_init=True)
    def _offer_cmap(self) -> None:
        """Offer the colormap only where it colours: classes and composites carry their own."""
        if self.variable is not None:
            self.param.cmap.precedence = 0.3 if self._legend is None and len(self.variable) == 1 else -1

    @property
    def _legend(self) -> Legend | None:
        """Return the shown band's legend where it lists classes."""
        legend = self._data[self.variable[0]].gs.legend if len(self.variable) == 1 else None
        return legend if legend is not None and legend.classes else None

    @property
    def ramp(self) -> ColorRamp:
        """Return the shown band's colour ramp, limits read from its first date drawn."""
        overview = self.overview(self.variable, self.time)
        if self.variable not in self._limits:
            values = overview.coarse[self.variable[0]].values
            self._limits[self.variable] = ColorRamp().fit(values).clim  # type: ignore[assignment]
        return ColorRamp(self.cmap, self._limits[self.variable])

    @property
    def stretch(self) -> Stretch:
        """Return the shown composite's stretch, read from its first date drawn."""
        overview = self.overview(self.variable, self.time)
        if self.variable not in self._limits:
            values = overview.coarse[list(self.variable)].to_array().values
            self._limits[self.variable] = Stretch().fit(values).clim  # type: ignore[assignment]
        return Stretch(self._limits[self.variable])

    def key(self) -> Key:
        """Return the shown variable's classes, ramp or stretch."""
        legend = self._legend
        if legend is not None:
            colors = legend.color_map or {}
            return Key(swatches={
                f"{name} ({code})": colors[code].lower()
                for code, name in (legend.classes or {}).items()
            })
        if len(self.variable) == 3:
            low, high = self.stretch.clim  # type: ignore[misc]
            return Key(note=f"RGB stretch {low:.4g} – {high:.4g}")
        ramp = self.ramp
        return Key(ramp=ramp.palette, clim=ramp.clim)

    def identify(self, x: float, y: float, tolerance: float) -> dict[str, str]:
        """Return the shown variable's full-detail value under a point, a class by its name."""
        values = self.overview(self.variable, self.time).value_at(x, y)
        if values is None or all(np.isnan(v) for v in values.values()):
            return {}
        legend = self._legend
        if legend is not None:
            (name, value), = values.items()
            code = int(value)
            return {name: f"{(legend.classes or {}).get(code, 'unnamed')} ({code})"}
        return {name: f"{value:.4g}" for name, value in values.items()}

    def overview(self, variable: tuple[str, ...], time: object = None) -> Overview:
        """Return one variable's map copy on one date, warped the first time it shows."""
        key = (variable, time)
        if key not in self._overviews:
            data = self._data if time is None else self._data.sel({TIME_COORDINATE: time})
            # Class codes warp without blending; anything else averages
            classes = len(variable) == 1 and self._legend is not None
            self._overviews[key] = webmap.Overview(
                data[list(variable)], resampling="nearest" if classes else "average"
            )
        return self._overviews[key]

    def image(self, view: View, time: object = None) -> hv.RGB | None:
        """Draw the shown variable at a view's detail, on the bar's date or `time`.

        Returns:
            RGB element on the view's grid, None where the raster lies off the view.
        """
        if not _overlaps(self.bounds, view):
            return None
        window = self.overview(self.variable, self.time if time is None else time).view(view)

        # One band draws through its ramp; a composite stretches; classes take their colours
        rgb = as_rgb(window, self.variable)
        if rgb is None:
            y, x = SPATIAL_DIMENSIONS
            return shade(
                hv.Image(window[self.variable[0]], kdims=[x, y]),
                cmap=self.cmap,
                clims=self.ramp.clim,
                cnorm="linear",
                alpha=round(self.alpha * 255),
                dynamic=False,
            )
        if len(self.variable) == 3:
            rgb = self.stretch.apply(rgb)
        return webmap.rgb_element(rgb, self.alpha)
```

(`Legend` imports from `geosave.geodata.attrs` under `TYPE_CHECKING`. `info` still reads `self._data[self.variable[0]].gs.nodata`; Task 6 replaces it.)

- [ ] **Step 6: Vector layers in `layers.py`**

`VectorLayer.__init__`, replacing everything from `web = webmap.on_web(frame)` on:

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

        # Rows carry their position and size; text becomes the categories datashader counts
        width, height = (
            web.bounds[["maxx", "maxy"]].values - web.bounds[["minx", "miny"]].values
        ).T
        rows = web.assign(
            **{c: _categories(web[c]) for c in columns if web[c].dtype.kind not in "biuf"},
            **{ROW: np.arange(len(web)), SIZE: np.maximum(width, height)},
        )

        # Shapes by how they draw, converted to spatialpandas once
        parts = webmap.split_by_geometry(rows)
        shapes = {kind: part for kind, part in parts.items() if kind != "Point"}
        if "Polygon" in parts:
            polygons = parts["Polygon"]
            shapes["Boundary"] = polygons.assign(**{polygons.geometry.name: polygons.boundary})
        shapes["Dot"] = rows.assign(**{rows.geometry.name: rows.representative_point()})
        self._shapes = {kind: spd.GeoDataFrame(part) for kind, part in shapes.items()}
```

Replace `paint` and `key`:

```python
    @property
    def colors(self) -> dict[object, str] | None:
        """Return each category of a text `column` with its colour in this theme, else None."""
        if self.column is None or self.frame[self.column].dtype.kind in "biuf":
            return None
        return category_colors(cast("pd.Series", self.frame[self.column]), self.theme)

    @property
    def ramp(self) -> ColorRamp | None:
        """Return the colour ramp of a numeric `column`, else None."""
        if self.column is None or self.frame[self.column].dtype.kind not in "biuf":
            return None
        return ColorRamp().fit(self.frame[self.column].to_numpy(dtype=float))

    def key(self) -> Key:
        """Return the colour of every feature, or of each value of `column`."""
        if self.column is None:
            return Key(swatches={"every feature": self.color})
        colors = self.colors
        if colors is not None:
            return Key(swatches={str(name): color for name, color in colors.items()})
        ramp = cast("ColorRamp", self.ramp)
        return Key(ramp=ramp.palette, clim=ramp.clim)
```

`image` and `_shade`:

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
                    outlined = self._shade(hv.Path, chosen, view, color=webmap.HIGHLIGHT)
                    images.append(spread(outlined, px=1, dynamic=False))
            chosen = dots[dots[ROW].isin(self.selected)]
            ringed = self._shade(hv.Points, chosen, view, color=webmap.HIGHLIGHT)
            images.append(spread(ringed, px=3, dynamic=False))
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
            return webmap.blank(view)
        drawn = element(shapes, vdims=[c for c in shapes.columns if c != shapes.geometry.name])
        opacity = round(alpha * 255)

        # One colour marks coverage, a category counts its shapes, a quantity averages
        if color is not None or self.column is None:
            fill = color or self.color
            covered = rasterize(drawn, aggregator=ds.any(), **view.canvas)
            return shade(covered, cmap=[fill, fill], alpha=opacity, dynamic=False)
        colors = self.colors
        if colors is not None:
            counts = rasterize(drawn, aggregator=ds.count_cat(self.column), **view.canvas)
            return shade(counts, color_key=colors, min_alpha=opacity, alpha=opacity, dynamic=False)
        ramp = cast("ColorRamp", self.ramp)
        means = rasterize(drawn, aggregator=ds.mean(self.column), **view.canvas)
        return shade(means, cmap=ramp.palette, clims=ramp.clim, cnorm="linear", alpha=opacity, dynamic=False)
```

Module helpers: delete `_plain`; keep `_overlaps`; `_counted` becomes

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
    """Return text as categories in the sorted order `category_colors` colours them."""
    return pd.Categorical(values, categories=sorted(values.dropna().unique(), key=str))
```

`CatalogLayer.__init__`: `self._items = webmap.drawable(frame)`.

- [ ] **Step 7: Stack in the explorer**

`app.py` imports `from holoviews.operation.datashader import stack`. In `_draw`, after the loop collecting `images`:

```python
        drawn = [x for x in images if x is not None]
        composite = stack(hv.Overlay(drawn), dynamic=False) if drawn else webmap.blank(view)
        return composite.opts(hooks=[self._move], backend="bokeh")
```

`_swipe`:

```python
    def _swipe(self, view: webmap.View) -> hv.RGB:
        """Return the left side's raster left of the cut, the right side's right of it."""
        (left, left_time), (right, right_time) = self.left, self.right
        drawn = [left.image(view, left_time), right.image(view, right_time)]
        image, other = (webmap.rgba(x if x is not None else webmap.blank(view)) for x in drawn)
        cut = round(self.split * image.shape[1])
        image[:, cut:] = other[:, cut:]
        # A line marks the cut
        image[:, max(cut - 1, 0) : cut + 1] = CUT
        return hv.RGB(image, bounds=view.bounds)
```

- [ ] **Step 8: Delete `elements`**

```bash
git rm src/geosave/geodata/viz/elements.py tests/geodata/viz/test_elements.py
```

`viz/__init__.py` docstring: "`style` decides how values become colour, `plot` draws figures through hvPlot on matplotlib, `map` holds the Web Mercator view and raster overviews, and `explorer` lays maps and controls out as a Panel app."

- [ ] **Step 9: Run the viz tests**

Run: `uv run pytest tests/geodata/viz tests/geodata/core -q`
Expected: PASS. If `test_swipe_keeps_orientation` matches upside down, flip `rgba`'s orientation rule as noted in Step 3.

- [ ] **Step 10: Commit**

```bash
git add -A src/geosave/geodata/viz tests/geodata/viz
git commit -m "refactor(viz): layers render through rasterize, shade and stack; Style and elements go"
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
