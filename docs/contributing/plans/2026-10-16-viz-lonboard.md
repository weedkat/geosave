# Explorer map on Lonboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the explorer's HoloViews map with one `lonboard.Map`, keeping `Explorer`, its docks, layer parameters and `.gs.explore()`.

**Architecture:** Each `Layer` owns the Lonboard layer it draws (`deck`) and keeps its traits in step with its parameters, so style changes redraw in the browser only. Rasters reach the browser as tiles over the widget comm: COG-backed selections through `RasterLayer.from_geotiff`, in-memory ones through Web Mercator XYZ windows. Vectors draw as Lonboard GeoArrow layers; filters run on the GPU through `DataFilterExtension`.

**Tech Stack:** lonboard 0.17 (deck.gl, anywidget), async-geotiff, obstore, Panel 1.9 + panel-material-ui, ipywidgets_bokeh, xarray/odc-geo, geopandas.

**Spec:** `docs/contributing/specs/2026-10-16-viz-lonboard-design.md`

## Global Constraints

- `lonboard[geotiff]>=0.17,<0.18`, `obstore`, `ipywidgets_bokeh` join the core dependencies beside Panel.
- `geodata` imports no torch; `viz` imports no HoloViews outside `plot.py`.
- Tile layers (basemap, rasters) are `pickable=False`.
- Vectors above `VIEW_LIMIT = 250_000` rows hold only the rows in view.
- The map is given `view_state` at construction.
- Served page: `pmui.Page` with `pn.pane.IPyWidget(map)`. Notebook page: `ipywidgets.HBox([pn.ipywidget(sidebar), ipywidgets.VBox([pn.ipywidget(bar), map, pn.ipywidget(below)])])`.
- What a raster shows: a band with a `Legend` draws its classes, `.gs.rgb` a composite, else the first band (unchanged).
- Code style (CLAUDE.md): verbs for functions, values for variables, loops written out, behaviour on the owning class, Google docstrings, comments say what, not how.

## Review Focus

1. A raster whose selected variables come from two files (Sentinel-2 bands on Earth Search) must draw through the xarray path, not fail opening a GeoTIFF — test in Task 3.
2. A tile outside the raster, or wholly nodata, must return no image, not a transparent PNG or an error — test in Task 3.
3. A vector frame with mixed polygons, lines and points must draw each kind, and a selection spanning kinds outlines all of them — test in Task 4.
4. Filtering away every row must leave the layer drawing nothing, not raise — test in Task 4.
5. Removing the selected layer, or removing the last raster while comparing, must leave the map with the remaining layers in order — test in Task 6.

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies |
| `src/geosave/geodata/io/storage.py` | `open_geotiff(location, storage_options)`: a location to an async-geotiff `GeoTIFF` |
| `src/geosave/geodata/viz/style.py` | Colour primitives: `Stretch`, `get_category_colors`, `get_ramp`, `encode_image` |
| `src/geosave/geodata/viz/map.py` | `WEB_MERCATOR`, `BASEMAPS`, `View` (bbox ↔ view state, pixel size at a zoom), `create_basemap` |
| `src/geosave/geodata/viz/explorer/layers/base.py` | `Layer`: parameters, `deck`, `_show` |
| `src/geosave/geodata/viz/explorer/layers/raster.py` | `RasterLayer`: rule, limits, colour, two tile readers |
| `src/geosave/geodata/viz/explorer/layers/vector.py` | `VectorLayer`: one Lonboard layer per geometry kind, filters, selection, view rows |
| `src/geosave/geodata/viz/explorer/layers/catalog.py` | `CatalogLayer`: unchanged behaviour on the new base |
| `src/geosave/geodata/viz/explorer/app.py` | `Explorer`: the map, stack, clicks, view, compare, page |
| delete `src/geosave/geodata/viz/raster.py`, `tests/geodata/viz/test_raster.py` | `Overview` has no user left |

---

### Task 1: Dependencies and `open_geotiff`

**Files:**
- Modify: `pyproject.toml` (core `dependencies`)
- Modify: `src/geosave/geodata/io/storage.py`
- Test: `tests/geodata/io/test_storage.py`

**Interfaces:**
- Produces: `open_geotiff(path: Location, storage_options: StorageOptions | None = None) -> async_geotiff.GeoTIFF`

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, add to `[project] dependencies`, beside `panel`:

```toml
    "lonboard[geotiff]>=0.17,<0.18",
    "obstore>=0.8",
    "ipywidgets_bokeh>=1.6",
```

Run: `uv sync`
Expected: resolves; `uv run python -c "import lonboard, async_geotiff, obstore, ipywidgets_bokeh"` exits 0.

- [ ] **Step 2: Write the failing test**

```python
# tests/geodata/io/test_storage.py
import numpy as np
import xarray as xr

from geosave.geodata.io.storage import open_geotiff


def test_open_geotiff_reads_a_local_cog(tmp_path) -> None:
    band = xr.DataArray(
        np.arange(256 * 256, dtype="uint16").reshape(256, 256),
        dims=("y", "x"),
        coords={"y": 9_000_000 - 10 * np.arange(256), "x": 500_000 + 10 * np.arange(256)},
    ).rio.write_crs("EPSG:32749")
    path = tmp_path / "band.tif"
    band.rio.to_raster(path, driver="COG")

    geotiff = open_geotiff(path)

    assert (geotiff.height, geotiff.width) == (256, 256)
    assert geotiff.crs.to_epsg() == 32749
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/geodata/io/test_storage.py::test_open_geotiff_reads_a_local_cog -v`
Expected: FAIL with `ImportError: cannot import name 'open_geotiff'`

- [ ] **Step 4: Implement**

Append to `src/geosave/geodata/io/storage.py`:

```python
def open_geotiff(
    path: Location, storage_options: StorageOptions | None = None
) -> GeoTIFF:
    """Open a GeoTIFF where it lives, reading its header only.

    Args:
        path: Local path or URL of the file.
        storage_options: Credentials and endpoint for a bucket, as `location` takes them.

    Returns:
        GeoTIFF whose tiles and overviews read on demand.

    Examples:
        >>> open_geotiff("s3://sentinel-cogs/.../B04.tif").overviews
    """
    from async_geotiff import GeoTIFF
    from obstore.store import from_url

    url = absolute_location(location(path, storage_options))
    directory, name = url.rsplit("/", 1)
    store = from_url(directory, **dict(storage_options or {}))
    # GeoTIFF opens only asynchronously; a notebook already runs an event loop,
    # so the open runs on a thread with its own.
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, GeoTIFF.open(name, store=store)).result()
```

Add the imports at the top: `import asyncio`, `from concurrent.futures import ThreadPoolExecutor`, and under `TYPE_CHECKING`: `from async_geotiff import GeoTIFF`. If `absolute_location` returns a local path without a scheme, prefix `file://` before `from_url` (check its docstring; local paths need `file:///abs/dir`).

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/geodata/io/test_storage.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/geosave/geodata/io/storage.py tests/geodata/io/test_storage.py
git commit -m "feat(io): open a GeoTIFF where it lives with async-geotiff"
```

---

### Task 2: Colour primitives and the view

**Files:**
- Modify: `src/geosave/geodata/viz/style.py`
- Modify: `src/geosave/geodata/viz/map.py` (replace `create_tiles` with `create_basemap`; add `View.view_state`, `View.from_view_state`, `get_pixel_size`)
- Test: `tests/geodata/viz/test_style.py`, `tests/geodata/viz/test_map.py`

**Interfaces:**
- Produces:
  - `Stretch.apply(channels: np.ndarray | xr.DataArray)` — accepts ndarray too
  - `get_ramp(cmap: str) -> np.ndarray` — `(256, 3)` uint8
  - `encode_image(rgba: np.ndarray) -> EncodedImage | None` — WebP; None when every pixel is clear
  - `View.view_state -> dict[str, float]` (`longitude`, `latitude`, `zoom`)
  - `View.from_view_state(state, width, height) -> View`
  - `get_pixel_size(zoom: float, latitude: float) -> float` — metres per screen pixel
  - `create_basemap(name: str) -> tuple[lonboard.basemap.MaplibreBasemap | None, lonboard.BitmapTileLayer | None]`
  - `BASEMAPS: dict[str, str | None]` — name to Carto style URL or XYZ tile URL

- [ ] **Step 1: Write the failing tests**

```python
# tests/geodata/viz/test_style.py (add)
import numpy as np

from geosave.geodata.viz.style import Stretch, encode_image, get_ramp


def test_stretch_applies_to_arrays() -> None:
    values = np.array([[0.0, 5.0, 10.0, np.nan]])
    stretched = Stretch((0.0, 10.0)).apply(values)
    np.testing.assert_allclose(stretched, [[0.0, 0.5, 1.0, np.nan]])


def test_ramp_has_256_colours() -> None:
    ramp = get_ramp("viridis")
    assert ramp.shape == (256, 3) and ramp.dtype == np.uint8


def test_encodes_webp_and_skips_clear_tiles() -> None:
    rgba = np.zeros((8, 8, 4), "uint8")
    assert encode_image(rgba) is None
    rgba[0, 0] = (255, 0, 0, 255)
    image = encode_image(rgba)
    assert image.media_type == "image/webp" and image.data[:4] == b"RIFF"
```

```python
# tests/geodata/viz/test_map.py (add)
from odc.geo.geom import BoundingBox

from geosave.geodata.viz.map import BASEMAPS, View, create_basemap, get_pixel_size


def test_view_round_trips_through_a_view_state() -> None:
    view = View(BoundingBox(1_000_000, -200_000, 1_100_000, -100_000, crs="EPSG:3857"), 800, 600)
    again = View.from_view_state(view.view_state, 800, 600)
    assert abs(again.bbox.left - view.bbox.left) < 2_000
    assert abs(again.bbox.top - view.bbox.top) < 2_000


def test_pixel_size_halves_per_zoom() -> None:
    assert abs(get_pixel_size(10, 0.0) / get_pixel_size(11, 0.0) - 2.0) < 1e-9


def test_basemaps_are_a_style_or_tiles() -> None:
    style, tiles = create_basemap("Light")
    assert style is not None and tiles is None
    style, tiles = create_basemap("Imagery")
    assert style is None and tiles.pickable is False
    assert create_basemap("None") == (None, None)
    assert set(BASEMAPS) >= {"Light", "Dark", "OpenStreetMap", "Imagery", "None"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/viz/test_style.py tests/geodata/viz/test_map.py -v`
Expected: FAIL with ImportError on `encode_image`, `get_ramp`, `create_basemap`.

- [ ] **Step 3: Implement `style.py` additions**

```python
def apply(self, channels: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
    """Map channels onto `[0, 1]`, clipping outside the limits; nan stays nan."""
    values = channels.values if hasattr(channels, "values") else channels
    low, high = self.get_limits(np.asarray(values))
    return ((channels.astype("float32") - low) / (high - low)).clip(0.0, 1.0)


def get_ramp(cmap: str) -> np.ndarray:
    """Return a colormap as 256 RGB rows, low to high."""
    from matplotlib import colormaps

    samples = colormaps[cmap](np.linspace(0.0, 1.0, 256))
    return (samples[:, :3] * 255).round().astype("uint8")


def encode_image(rgba: np.ndarray) -> EncodedImage | None:
    """Encode an RGBA tile as WebP, None when every pixel is clear."""
    from lonboard.raster import EncodedImage
    from PIL import Image

    if not rgba[..., 3].any():
        return None
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="WEBP", quality=80, method=2)
    return EncodedImage(data=buffer.getvalue(), media_type="image/webp")
```

Update the module docstring: "Colour primitives the map and figures share: a composite's stretch, ramps, each theme's category colours, and tile encoding." Add `import io`; under `TYPE_CHECKING`: `from lonboard.raster import EncodedImage`.

- [ ] **Step 4: Implement `map.py`**

Replace `create_tiles`, the HoloViews imports and `_GRAY`/`_DARK_GRAY` with:

```python
from lonboard import BitmapTileLayer
from lonboard.basemap import CartoStyle, MaplibreBasemap

# Basemap names in the picker: a Carto vector style, a raster tile URL, or nothing.
BASEMAPS: dict[str, str | None] = {
    "Light": CartoStyle.Positron,
    "Dark": CartoStyle.DarkMatter,
    "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "None": None,
}

# Earth's circumference at the equator, in Web Mercator metres; deck.gl tiles are 512 px.
EQUATOR = 40_075_016.686
TILE_PIXELS = 512


def create_basemap(name: str) -> tuple[MaplibreBasemap | None, BitmapTileLayer | None]:
    """Return the map style or the bottom tile layer a picker name calls for."""
    source = BASEMAPS[name]
    if source is None:
        return None, None
    if isinstance(source, CartoStyle):
        return MaplibreBasemap(style=source), None
    return None, BitmapTileLayer(data=source, tile_size=256, pickable=False)


def get_pixel_size(zoom: float, latitude: float) -> float:
    """Return the ground size of one screen pixel at a zoom, in metres."""
    return EQUATOR * math.cos(math.radians(latitude)) / (TILE_PIXELS * 2**zoom)
```

Add to `View`:

```python
@property
def view_state(self) -> dict[str, float]:
    """Return the deck.gl view state showing this box: centre and zoom."""
    centre = self.bbox.to_crs("EPSG:4326")
    longitude = (centre.left + centre.right) / 2
    latitude = (centre.bottom + centre.top) / 2
    # Web Mercator metres per pixel at zoom 0 is EQUATOR / TILE_PIXELS
    zoom = math.log2(EQUATOR / (TILE_PIXELS * self.pixel_size))
    return {"longitude": longitude, "latitude": latitude, "zoom": zoom}

@classmethod
def from_view_state(cls, state: Mapping[str, float], width: int, height: int) -> View:
    """Return the box a deck.gl view state shows on a map of a size."""
    size = EQUATOR / (TILE_PIXELS * 2 ** state["zoom"])
    x, y = Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True).transform(
        state["longitude"], state["latitude"]
    )
    half_x, half_y = size * width / 2, size * height / 2
    return cls(BoundingBox(x - half_x, y - half_y, x + half_x, y + half_y, crs=WEB_MERCATOR), width, height)
```

Imports: `import math`, `from collections.abc import Mapping`, `from pyproj import Transformer`. Remove `View.canvas` (datashader only) and `THEME_BASEMAP`'s values stay `{"light": "Light", "dark": "Dark"}`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/geodata/viz/test_style.py tests/geodata/viz/test_map.py -v`
Expected: PASS (delete the old `test_map.py` cases that test `create_tiles` or `View.canvas`).

- [ ] **Step 6: Commit**

```bash
git add src/geosave/geodata/viz/style.py src/geosave/geodata/viz/map.py tests/geodata/viz/test_style.py tests/geodata/viz/test_map.py
git commit -m "feat(viz): colour primitives, deck.gl view states and lonboard basemaps"
```

---

### Task 3: Layer base and RasterLayer

**Files:**
- Modify: `src/geosave/geodata/viz/explorer/layers/base.py`
- Modify: `src/geosave/geodata/viz/explorer/layers/raster.py`
- Delete: `src/geosave/geodata/viz/raster.py`, `tests/geodata/viz/test_raster.py`
- Test: `tests/geodata/viz/explorer/test_layers.py` (`TestRasterLayer`)

**Interfaces:**
- Consumes: `open_geotiff` (Task 1); `Stretch`, `get_ramp`, `encode_image`, `WEB_MERCATOR`, `View` (Task 2)
- Produces:
  - `Layer.deck -> lonboard.BaseLayer` (abstract property); `Layer._show()` syncs `visible`/`alpha`
  - `RasterLayer.deck` — `lonboard.RasterLayer`, cached per `(variable, time, cmap)`
  - `RasterLayer.colorize(values: np.ndarray) -> np.ndarray` — `(bands, h, w)` float, nan clear → `(h, w, 4)` uint8
  - `RasterLayer.read_tile(x: int, y: int, z: int) -> np.ndarray | None` — `(bands, 256, 256)` float window, None outside
  - `RasterLayer.source -> str | None` — the one file the selection lives in, None when in memory or spread over files

- [ ] **Step 1: Write the failing tests** (replace `test_draws_every_selection_as_rgb`, `test_classes_draw_in_their_legend_colours`, `test_renders_one_image_at_its_opacity`)

```python
import math

import lonboard
import numpy as np
import xarray as xr

from geosave.geodata.io.openers import open_raster
from geosave.geodata.viz.explorer import RasterLayer
from geosave.geodata.viz.map import WEB_MERCATOR


def tile_of(layer: RasterLayer, zoom: int) -> tuple[int, int, int]:
    """The XYZ tile under a layer's centre."""
    box = layer.bbox.to_crs("EPSG:4326")
    longitude, latitude = (box.left + box.right) / 2, (box.bottom + box.top) / 2
    x = int((longitude + 180) / 360 * 2**zoom)
    y = int((1 - math.asinh(math.tan(math.radians(latitude))) / math.pi) / 2 * 2**zoom)
    return x, y, zoom


class TestRasterLayer:
    def test_draws_on_a_lonboard_raster_layer(self, scene) -> None:
        layer = RasterLayer(scene)
        assert isinstance(layer.deck, lonboard.RasterLayer)
        assert layer.deck.pickable is False

    def test_opacity_and_visibility_change_the_same_deck(self, scene) -> None:
        layer = RasterLayer(scene)
        deck = layer.deck
        layer.alpha = 0.3
        layer.visible = False
        assert layer.deck is deck
        assert deck.opacity == 0.3 and deck.visible is False

    def test_a_new_variable_draws_a_new_deck_and_back_reuses_it(self, scene) -> None:
        layer = RasterLayer(scene)
        first = layer.deck
        layer.variable = "B04"
        assert layer.deck is not first
        layer.variable = scene.gs.rgb
        assert layer.deck is first

    def test_reads_a_window_inside_and_nothing_outside(self, scene) -> None:
        layer = RasterLayer(scene, "B04")
        x, y, z = tile_of(layer, 14)
        window = layer.read_tile(x, y, z)
        assert window.shape == (1, 256, 256) and np.isfinite(window).any()
        assert layer.read_tile(0, 0, 14) is None

    def test_colours_a_band_classes_and_a_composite(self, scene) -> None:
        values = np.full((1, 2, 2), np.nan, "float32")
        values[0, 0, 0] = 1.0
        layer = RasterLayer(scene, "cover")                    # classes
        rgba = layer.colorize(values)
        assert tuple(rgba[0, 0]) == (0x39, 0x7D, 0x49, 255)    # "trees" #397d49
        assert rgba[1, 1, 3] == 0                              # nan stays clear
        layer = RasterLayer(scene, "B04")                      # band through its ramp
        assert layer.colorize(values)[0, 0, 3] == 255
        layer = RasterLayer(scene)                             # composite
        assert layer.colorize(np.ones((3, 2, 2), "float32")).shape == (2, 2, 4)

    def test_a_cog_selection_reads_its_file(self, catalog) -> None:
        raster = catalog.iloc[[0]].iloc[0].gs.to_xarray()
        layer = RasterLayer(raster)
        assert layer.source is not None and layer.source.endswith(".tif")

    def test_a_selection_across_files_reads_as_a_window(self, tmp_path, scene) -> None:
        paths = scene[["B04", "B03", "B02"]].gs.to_cog(tmp_path / "s_{band}.tif")
        raster = xr.merge([open_raster(path) for path in paths])
        layer = RasterLayer(raster, rgb=("B04", "B03", "B02"))
        assert layer.source is None
        assert isinstance(layer.deck, lonboard.RasterLayer)
```

Keep `test_identifies_the_value_under_a_point`, `test_dates_share_colour_limits`, `test_steps_through_its_dates` and the Info tests; they must still pass.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/viz/explorer/test_layers.py::TestRasterLayer -v`
Expected: FAIL with `AttributeError: 'RasterLayer' object has no attribute 'deck'`

- [ ] **Step 3: Implement `Layer` in `base.py`**

Replace `render` with:

```python
    @property
    @abstractmethod
    def deck(self) -> BaseLayer:
        """Return the lonboard layer the map draws for this layer's settings."""

    @param.depends("visible", "alpha", watch=True)
    def _show(self) -> None:
        """Show the deck at the layer's visibility and opacity; nothing is redrawn on the server."""
        self.deck.visible = self.visible
        self.deck.opacity = self.alpha
```

Drop the `holoviews` and `View` imports; import `from lonboard import BaseLayer` under `TYPE_CHECKING`. Update the class docstring: "draws itself: `deck` is the lonboard layer the map overlays, changed in place as settings change".

- [ ] **Step 4: Implement `RasterLayer`**

Remove `render`, `draw`, `get_overview`, `_overviews`, `RAMP_COLORS` and the HoloViews imports. Add:

```python
# Pixels per side of a tile the in-memory reader returns.
TILE_SIZE = 256
# Workers reading in-memory tiles; tiles of a file read asynchronously instead.
TILE_WORKERS = ThreadPoolExecutor(4)
# Longest side of the copy colour limits are read from.
LIMIT_SAMPLE = 1024
```

```python
    @property
    def source(self) -> str | None:
        """Return the one file the shown selection lives in, None in memory or across files."""
        sources = set()
        for name in self.band_names:
            sources.add(self._raster[name].encoding.get("source"))
        if len(sources) != 1 or None in sources:
            return None
        return str(next(iter(sources)))

    @property
    def deck(self) -> lonboard.RasterLayer:
        """Return the tiles of the shown selection on its date, built once per setting."""
        key = (self.variable, self.time, self.cmap)
        if key not in self._decks:
            self._decks[key] = self._build_deck()
            self._decks[key].visible = self.visible
            self._decks[key].opacity = self.alpha
        return self._decks[key]

    def _build_deck(self) -> lonboard.RasterLayer:
        """Read a file's own tiles where the selection lives in one, else windows of the raster."""
        source = self.source
        if source is not None:
            geotiff = open_geotiff(source)
            bands = self._read_band_indexes()
            return lonboard.RasterLayer.from_geotiff(
                geotiff,
                render_tile=lambda tile: encode_image(
                    self.colorize(self._unpack(np.asarray(tile.array.data)[bands]))
                ),
                pickable=False,
            )
        box = self.bbox.to_crs("EPSG:4326")
        return lonboard.RasterLayer(
            _tile_matrix_set=None,
            _crs=WEB_MERCATOR,
            _fetch_tile=self._fetch_tile,
            _render_tile=lambda window: None if window is None else encode_image(self.colorize(window)),
            _bounds=(box.left, box.bottom, box.right, box.top),
            _center=((box.left + box.right) / 2, (box.bottom + box.top) / 2),
            pickable=False,
        )

    async def _fetch_tile(self, x: int, y: int, z: int) -> np.ndarray | None:
        """Read one tile's window on a worker, so the map keeps answering."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(TILE_WORKERS, self.read_tile, x, y, z)

    def read_tile(self, x: int, y: int, z: int) -> np.ndarray | None:
        """Return the shown selection under a Web Mercator tile, nan outside, None off the raster."""
        tile = GeoBox.from_bbox(get_tile_bbox(x, y, z), shape=(TILE_SIZE, TILE_SIZE), tight=True)
        raster = self._dated_raster()[self.band_names]
        grid = raster.odc.geobox
        box = tile.extent.to_crs(grid.crs).boundingbox
        pad = 2 * abs(grid.resolution.x)
        ys, xs = SPATIAL_DIMENSIONS
        window = raster.sel({xs: slice(box.left - pad, box.right + pad), ys: slice(box.top + pad, box.bottom - pad)})
        if window.sizes[xs] == 0 or window.sizes[ys] == 0:
            return None
        window = window.gs.to_nan().astype("float32").compute()
        # Class codes are never blended; values average onto the tile
        kernel = "nearest" if self.legend is not None else "average"
        values = window.odc.reproject(tile, resampling=kernel).to_array(BAND_DIMENSION).values
        return values if np.isfinite(values).any() else None

    def colorize(self, values: np.ndarray) -> np.ndarray:
        """Colour `(bands, h, w)` values as the shown selection draws them; nan stays clear."""
        clear = ~np.isfinite(values).all(axis=0)
        rgba = np.zeros((*values.shape[1:], 4), "uint8")
        if self.legend is not None:
            # Each class in its legend's colour; codes without one stay clear
            codes = np.nan_to_num(values[0], nan=-1).astype("int64")
            for code, color in (self.legend.color_map or {}).items():
                hit = codes == code
                rgba[hit, :3] = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
                rgba[hit, 3] = 255
            rgba[clear, 3] = 0
            return rgba
        if isinstance(self.variable, str):
            low, high = self.clim
            position = np.clip((values[0] - low) / (high - low), 0.0, 1.0)
            ramp = get_ramp(self.cmap)
            rgba[..., :3] = ramp[(np.nan_to_num(position) * 255).astype("int64")]
        else:
            channels = Stretch(self.clim).apply(values)
            rgba[..., :3] = (np.nan_to_num(channels).transpose(1, 2, 0) * 255).astype("uint8")
        rgba[..., 3] = np.where(clear, 0, 255)
        return rgba
```

Helpers on the class, each one purpose:

```python
    def _dated_raster(self) -> xr.Dataset:
        """Return the raster on the layer's date, or as given when timeless."""
        if self.time is None:
            return self._raster
        return self._raster.sel({TIME_COORDINATE: self.time})

    def _read_band_indexes(self) -> list[int]:
        """Return each shown variable's band position in its file, from the opener's encoding."""
        indexes = []
        for name in self.band_names:
            # rioxarray records the 1-based band it read
            indexes.append(int(self._raster[name].encoding.get("band", 1)) - 1)
        return indexes

    def _unpack(self, values: np.ndarray) -> np.ndarray:
        """Turn stored file values into the raster's: nodata to nan, scale and offset applied."""
        unpacked = values.astype("float32")
        for index, name in enumerate(self.band_names):
            variable = self._raster[name]
            if variable.gs.nodata is not None:
                unpacked[index][values[index] == variable.gs.nodata] = np.nan
            packing = variable.gs.packing
            if packing is not None:
                unpacked[index] = unpacked[index] * (packing.scale_factor or 1) + (packing.add_offset or 0)
        return unpacked
```

`clim` reads from a strided copy instead of the overview:

```python
        if self.variable not in self._limits:
            raster = self._dated_raster()[self.band_names].gs.to_nan()
            ys, xs = SPATIAL_DIMENSIONS
            step = max(1, max(raster.sizes[ys], raster.sizes[xs]) // LIMIT_SAMPLE)
            values = raster.isel({ys: slice(None, None, step), xs: slice(None, None, step)}).to_array().values
            ...  # the rest unchanged
```

`identify(x, y, tolerance)` reads the full-detail pixel directly:

```python
        point_x, point_y = Transformer.from_crs(WEB_MERCATOR, self._raster.odc.crs, always_xy=True).transform(x, y)
        ys, xs = SPATIAL_DIMENSIONS
        bbox = self._raster.odc.geobox.boundingbox
        if not (bbox.left <= point_x <= bbox.right and bbox.bottom <= point_y <= bbox.top):
            return {}
        pixel = self._dated_raster()[self.band_names].gs.to_nan().sel({xs: point_x, ys: point_y}, method="nearest").compute()
        values = {str(name): float(value) for name, value in pixel.data_vars.items()}
        ...  # legend naming unchanged
```

`get_key` builds the ramp as hex from `get_ramp(self.cmap)`: `["#%02x%02x%02x" % tuple(row) for row in get_ramp(self.cmap)]`.

Put `get_tile_bbox(x, y, z) -> BoundingBox` in `viz/map.py` (Task 2 file; add it there with a test that `get_tile_bbox(0, 0, 0)` spans ±20037508.34).

In `__init__`, replace `self._overviews = {}` with `self._decks: dict[tuple[object, ...], lonboard.RasterLayer] = {}`.

- [ ] **Step 5: Delete `Overview`**

```bash
git rm src/geosave/geodata/viz/raster.py tests/geodata/viz/test_raster.py
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/geodata/viz/explorer/test_layers.py::TestRasterLayer -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -A src/geosave/geodata/viz tests/geodata/viz
git commit -m "feat(viz): raster layers draw lonboard tiles from their file or their window"
```

---

### Task 4: VectorLayer and CatalogLayer

**Files:**
- Modify: `src/geosave/geodata/viz/explorer/layers/vector.py`
- Modify: `src/geosave/geodata/viz/explorer/layers/catalog.py` (imports only)
- Test: `tests/geodata/viz/explorer/test_layers.py` (`TestVectorLayer`, `TestCatalogLayer`)

**Interfaces:**
- Consumes: `Layer.deck`, `Layer._show` (Task 3); `get_category_colors`, `SELECTION_COLOR`, `FEATURE_COLOR` (style)
- Produces:
  - `VectorLayer.deck -> lonboard.BaseLayer` — the first kind's layer (for `_show` on all kinds see below)
  - `VectorLayer.decks -> list[lonboard.BaseLayer]` — one per geometry kind, then the selection outline
  - `VectorLayer.show_view(view: View) -> None` — above `VIEW_LIMIT`, keep only rows in view
  - `VIEW_LIMIT = 250_000`

`Layer._show` sets one deck; a vector has several. Override `_show` in `VectorLayer` to loop over `self.decks`. The Explorer adds `layer.decks` when a layer has them, else `[layer.deck]` — give `Layer` a `decks` property returning `[self.deck]` and use `decks` everywhere in the Explorer.

- [ ] **Step 1: Write the failing tests** (replace the datashader drawing tests)

```python
from lonboard import PathLayer, PolygonLayer, ScatterplotLayer

from geosave.geodata.viz.explorer import VectorLayer
from geosave.geodata.viz.explorer.layers.vector import VIEW_LIMIT


class TestVectorLayer:
    def test_draws_each_geometry_kind_on_its_own_layer(self, fields) -> None:
        kinds = [type(deck) for deck in VectorLayer(fields).decks]
        assert kinds[:2] == [PolygonLayer, ScatterplotLayer]

    def test_sends_only_the_columns_it_shows(self, fields) -> None:
        deck = VectorLayer(fields).decks[0]
        assert set(deck.table.schema.names) <= {"crop", "area", "_row", "geometry"}

    def test_colours_by_a_text_column(self, fields) -> None:
        layer = VectorLayer(fields, column="crop")
        colors = np.asarray(layer.decks[0].get_fill_color)
        assert colors.shape == (2, 4)                      # two polygons, RGBA each
        assert not (colors[0] == colors[1]).all()          # rice and palm differ

    def test_alpha_changes_every_deck_in_place(self, fields) -> None:
        layer = VectorLayer(fields)
        decks = layer.decks
        layer.alpha = 0.5
        assert layer.decks[0] is decks[0]
        assert all(deck.opacity == 0.5 for deck in decks)

    def test_search_hides_rows_on_the_gpu(self, fields) -> None:
        layer = VectorLayer(fields)
        layer.search = "palm"
        kept = np.asarray(layer.decks[0].get_filter_value)
        assert kept.tolist() == [0.0, 1.0]

    def test_filtering_every_row_draws_nothing(self, fields) -> None:
        layer = VectorLayer(fields)
        layer.search = "no such crop"
        assert np.asarray(layer.decks[0].get_filter_value).sum() == 0

    def test_selection_outlines_every_kind_hit(self, fields) -> None:
        layer = VectorLayer(fields)
        layer.selected = [0, 2]                            # a polygon and the point
        outline = layer.decks[-1]
        assert outline.table.num_rows == 2

    def test_large_frames_hold_only_the_rows_in_view(self, fields, monkeypatch) -> None:
        monkeypatch.setattr("geosave.geodata.viz.explorer.layers.vector.VIEW_LIMIT", 1)
        layer = VectorLayer(fields)
        first = layer.frame.iloc[[0]].total_bounds
        layer.show_view(View(BoundingBox(*first, crs=WEB_MERCATOR)))
        assert layer.decks[0].table.num_rows == 1
```

Keep: `test_offers_columns_of_numbers_and_text`, `test_refuses_an_absent_column`, `test_offers_one_colour_only_while_no_column_colours`, `test_identify_selects_the_features_hit`, `test_dark_theme_recolours_categories` (assert the fill colours change), the catalog tests.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/viz/explorer/test_layers.py::TestVectorLayer -v`
Expected: FAIL with `AttributeError: ... 'decks'`

- [ ] **Step 3: Implement**

Remove `render`, `draw`, `highlight`, `_shade_coverage`, `_build_shapes`, the datashader/HoloViews/spatialpandas imports and the spread constants. Keep `__init__` (column discovery, `self.frame` in Web Mercator, `_row_column`), `filter_rows`, `find_matching_rows`, `find_nearby_rows`, `identify`, `describe_rows`, `get_info`, `get_key`, `category_colors`, `clim`.

```python
# Rows above which a layer holds only the rows in view: past it, sending the
# whole table makes a map open slowly (800k buildings took ~55 s).
VIEW_LIMIT = 250_000
# Pixels a point draws at, and a line or edge.
POINT_RADIUS = 4
LINE_WIDTH = 1


    @property
    def deck(self) -> lonboard.BaseLayer:
        """Return the layer of the first geometry kind."""
        return self.decks[0]

    @property
    def decks(self) -> list[lonboard.BaseLayer]:
        """Return one lonboard layer per geometry kind in view, then the selection outline."""
        if self._decks is None:
            self._decks = self._build_decks()
        return self._decks

    def show_view(self, view: View) -> None:
        """Hold only the rows in a view when the frame is too large to send whole."""
        if len(self.frame) <= VIEW_LIMIT:
            return
        rows = np.sort(self.frame.sindex.query(box(*view.bbox), predicate="intersects"))
        if not np.array_equal(rows, self._rows):
            self._rows = rows[:VIEW_LIMIT]
            self._decks = None
            self.param.trigger("redrawn")

    def _build_decks(self) -> list[lonboard.BaseLayer]:
        """Build each geometry kind's layer from the rows held and the columns shown."""
        shown = self.frame.iloc[self._rows]
        columns = [*self.columns, self._row_column]
        decks = []
        for kind, part in shapes.group_by_kind(shown).items():
            table = part[[*columns, part.geometry.name]].to_crs("EPSG:4326")
            decks.append(self._create_kind_layer(kind, table))
        decks.append(self._create_outline())
        for deck in decks:
            deck.visible, deck.opacity = self.visible, self.alpha
        self._paint()
        self._filter()
        return decks
```

`_create_kind_layer(kind, table)` builds `PolygonLayer.from_geopandas(table, pickable=True, auto_highlight=True, line_width_min_pixels=LINE_WIDTH, extensions=[DataFilterExtension(filter_size=1)], filter_range=(0.5, 1.0))`, `PathLayer.from_geopandas(..., width_min_pixels=LINE_WIDTH, ...)` or `ScatterplotLayer.from_geopandas(..., radius_min_pixels=POINT_RADIUS, ...)` by `kind` (`"polygon"`, `"line"`, `"point"`).

`shapes.group_by_kind(frame) -> dict[str, gpd.GeoDataFrame]` replaces `group_by_element` (keys are plain kind names now; keep `select_drawable` and `summarize_geometry`). Update `plot.py`'s use of `group_by_element`/`ELEMENTS` to the new name in the same commit.

```python
    @param.depends("column", "color", "theme", watch=True)
    def _paint(self) -> None:
        """Colour each kind's rows by `column`, or all in `color`; selection stays bright."""
        if self._decks is None:
            return
        for deck in self._decks[:-1]:
            rows = np.asarray(deck.table[self._row_column])
            values = self.frame[self.column].iloc[rows] if self.column else None
            if values is None:
                colors = np.array([*to_rgb255(self.color), 255], "uint8")
            elif self.category_colors is not None:
                colors = apply_categorical_cmap(values.astype(str), {str(k): [*to_rgb255(v), 255] for k, v in self.category_colors.items()})
            else:
                low, high = self.clim
                colors = apply_continuous_cmap((values.to_numpy() - low) / (high - low or 1), colormaps[DEFAULT_CMAP])
            if isinstance(deck, PolygonLayer):
                deck.get_fill_color = colors
                deck.get_line_color = colors
            elif isinstance(deck, PathLayer):
                deck.get_color = colors
            else:
                deck.get_fill_color = colors

    @param.depends("search", "filtered_rows", watch=True)
    def _filter(self) -> None:
        """Hide the rows the search and table drop, on the GPU, sending one number per row."""
        if self._decks is None:
            return
        kept = set(self.find_matching_rows().tolist())
        for deck in self._decks[:-1]:
            rows = np.asarray(deck.table[self._row_column])
            deck.get_filter_value = np.array([1.0 if row in kept else 0.0 for row in rows], "float32")

    @param.depends("selected", watch=True)
    def _outline(self) -> None:
        """Redraw the outline of the selected rows on top."""
        if self._decks is not None:
            self._decks[-1] = self._create_outline()
            self.param.trigger("redrawn")
```

```python
    def _create_outline(self) -> lonboard.PathLayer:
        """Return the selected rows' edges and lines, bright and wide, over the layer.

        Polygons outline by their boundary; a selected point draws as a ring of
        its own radius, so one path layer outlines every kind.
        """
        selected = self.frame.iloc[[row for row in self.selected if row in set(self._rows.tolist())]]
        outlines = selected.geometry.boundary.where(selected.geom_type.str.contains("Polygon"), selected.geometry)
        # A point has no boundary to draw; ring it at its drawn size, in metres at the frame's scale
        points = selected.geom_type.str.contains("Point")
        outlines[points] = selected.geometry[points].buffer(self._point_ring).boundary
        frame = gpd.GeoDataFrame(geometry=outlines, crs=WEB_MERCATOR).to_crs("EPSG:4326")
        return lonboard.PathLayer.from_geopandas(
            frame, get_color=[*to_rgb255(SELECTION_COLOR), 255], width_min_pixels=3, pickable=False
        )
```

`self._point_ring` is set in `__init__` to 1% of the frame's longer span (a ring visible at the frame's zoom). An empty selection builds the same layer from an empty frame, so the stack keeps its shape.

Add `redrawn = param.Event(precedence=-1)` to `Layer` (Task 3 file) so the Explorer can swap decks when a layer rebuilds them; `RasterLayer` triggers it when `variable`, `time` or `cmap` change.

`to_rgb255(color) -> tuple[int, int, int]` goes in `style.py` (`matplotlib.colors.to_rgb` × 255).

Initialise `self._decks = None` and `self._rows = np.arange(len(self.frame))` in `__init__`, and cap `_rows` at the first view's rows when `len(frame) > VIEW_LIMIT` (the Explorer calls `show_view` before first drawing).

`CatalogLayer`: drop the `shapes` import if unused; behaviour unchanged.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/geodata/viz/explorer/test_layers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A src/geosave/geodata/viz tests/geodata/viz
git commit -m "feat(viz): vectors draw as lonboard layers, filtered on the GPU"
```

---

### Task 5: Explorer on the lonboard map

**Files:**
- Modify: `src/geosave/geodata/viz/explorer/app.py`
- Test: `tests/geodata/viz/explorer/test_app.py`

**Interfaces:**
- Consumes: `Layer.decks`, `Layer.redrawn`, `VectorLayer.show_view`, `create_basemap`, `View.view_state`, `View.from_view_state`, `get_pixel_size`
- Produces: `Explorer.map -> lonboard.Map`; public API unchanged (`add`, `zoom`, `run_action`, `layers`, `selected`, `basemap`, `theme`, `time`, `identified`, `compare`, `left`, `right`)

- [ ] **Step 1: Write the failing tests** (replace `test_overlays_each_shown_layer_over_the_basemap`, `test_outlines_selected_features_and_skips_hidden_layers`, `test_keeps_one_map_as_the_stack_changes`, `test_zooming_moves_the_map_it_has`)

```python
import lonboard


class TestMap:
    def test_draws_rasters_under_vectors_on_one_map(self, scene, fields) -> None:
        explorer = Explorer(fields, scene)
        decks = list(explorer.map.layers)
        raster, vector = explorer.layers[1], explorer.layers[0]
        assert decks.index(raster.deck) < decks.index(vector.decks[0])

    def test_keeps_one_map_as_the_stack_changes(self, scene, fields) -> None:
        explorer = Explorer(scene)
        map_ = explorer.map
        explorer.add(fields)
        explorer.run_action(explorer.layers[0], "remove")
        assert explorer.map is map_
        assert list(map_.layers)[-len(explorer.layers[0].decks):] == explorer.layers[0].decks

    def test_a_rebuilt_raster_deck_replaces_the_old_one(self, scene) -> None:
        explorer = Explorer(scene)
        layer = explorer.layers[0]
        old = layer.deck
        layer.variable = "B04"
        assert layer.deck in explorer.map.layers and old not in explorer.map.layers

    def test_zoom_flies_to_a_box(self, scene) -> None:
        explorer = Explorer(scene)
        explorer.zoom(explorer.layers[0].bbox)
        state = explorer.map.view_state
        assert abs(state.longitude - explorer._view.view_state["longitude"]) < 1e-6

    def test_a_click_identifies_visible_layers(self, scene, fields) -> None:
        explorer = Explorer(scene, fields)
        x, y = fields.to_crs(WEB_MERCATOR).geometry.iloc[0].centroid.coords[0]
        lon, lat = Transformer.from_crs(WEB_MERCATOR, "EPSG:4326", always_xy=True).transform(x, y)
        explorer._identify_click((lon, lat))
        assert [name for name, _ in explorer.identified][0] == explorer.layers[1].name

    def test_the_basemap_picker_swaps_style_or_tiles(self, scene) -> None:
        explorer = Explorer(scene)
        explorer.basemap = "Imagery"
        assert isinstance(list(explorer.map.layers)[0], lonboard.BitmapTileLayer)
        explorer.basemap = "Dark"
        assert not isinstance(list(explorer.map.layers)[0], lonboard.BitmapTileLayer)

    def test_removing_the_selected_layer_keeps_the_rest_in_order(self, scene, fields, monthly) -> None:
        explorer = Explorer(scene, monthly, fields)
        explorer.selected = explorer.layers[1]
        explorer.run_action(explorer.layers[1], "remove")
        assert explorer.selected in explorer.layers
        assert [d for d in explorer.map.layers if d in explorer.layers[0].decks]
```

Keep the time, catalog, actions, accessors, theme and dock tests; adapt any that read `explorer._pane` to read `explorer.map`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/viz/explorer/test_app.py::TestMap -v`
Expected: FAIL with `AttributeError: 'Explorer' object has no attribute 'map'`

- [ ] **Step 3: Implement**

Remove `MAP_OPTIONS`, `_draw_stack`, `_draw_frame`, `_get_view`, `_keep_plot`, `_frame_map`, `_create_basemap`, `_ranges`, `_size`, `_plot`, the HoloViews imports and streams.

`__init__` builds the map once:

```python
        self._view = View.frame(bbox_union(layer.bbox for layer in map_layers))
        style, _tiles = create_basemap(self.basemap)
        self.map = lonboard.Map(
            [],
            basemap=style,
            view_state=self._view.view_state,      # given now: Panel reads state while the widget builds
            height=height,
            show_tooltip=True,
            show_side_panel=False,
        )
        self.map.on_click(self._identify_click)
        self.map.observe(self._follow_view, "view_state")
```

Then, after `self.layers = map_layers`:

```python
    @param.depends("layers", "basemap", "edited", watch=True)
    def _draw_layers(self) -> None:
        """Lay the basemap tiles, then each layer's decks in drawing order, on the map."""
        style, tiles = create_basemap(self.basemap)
        self.map.basemap = style
        decks = [] if tiles is None else [tiles]
        for layer in self.draw_order:
            decks.extend(layer.decks)
        self.map.layers = decks
```

Watch `redrawn` on every layer in `_sync_layers` (it already watches every parameter and triggers `edited`; `redrawn` is a parameter, so `_draw_layers` follows through `edited` with no extra watcher).

```python
    def zoom(self, bbox: BoundingBox) -> None:
        """Move the map to show a box, with a margin round it."""
        self._view = View.frame(bbox)
        self.map.fly_to(**self._view.view_state, duration=500)

    def _follow_view(self, change: dict[str, object]) -> None:
        """Keep the view on screen; large vectors hold the rows in it."""
        state = change["new"]
        self._view = View.from_view_state(
            {"longitude": state.longitude, "latitude": state.latitude, "zoom": state.zoom},
            self._view.width, self._view.height,
        )
        for layer in self.layers:
            if isinstance(layer, VectorLayer):
                layer.show_view(self._view)

    def _identify_click(self, coordinate: tuple[float, float]) -> None:
        """Read every visible layer near a clicked longitude and latitude, top first."""
        longitude, latitude = coordinate
        x, y = TO_MERCATOR.transform(longitude, latitude)
        state = self.map.view_state
        tolerance = get_pixel_size(state.zoom, latitude) * TAP_TOLERANCE
        identified = []
        for layer in reversed(self.draw_order):
            if layer.visible and (values := layer.identify(x, y, tolerance)):
                identified.append((layer.name, values))
        self.identified = identified
```

`TO_MERCATOR = Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True)` at module level. Call `layer.show_view(self._view)` for each new `VectorLayer` in `_sync_layers` before drawing.

Page (`__panel__`): served —

```python
            page = pmui.Page(title="GeoSave Explorer", sidebar=sidebar, sidebar_width=SIDEBAR_WIDTH,
                             main=[pmui.Column(map_toolbar, pn.pane.IPyWidget(self.map, height=self.height,
                                   sizing_mode="stretch_width"), *get_docks("main"), sizing_mode="stretch_width")],
                             theme_config=THEME, dark_theme=self.theme == "dark")
```

notebook — `__panel__` cannot return an ipywidget, so give `Explorer` `_repr_mimebundle_` for the notebook and keep `__panel__` for serving:

```python
    def _repr_mimebundle_(self, include=None, exclude=None):
        """Show the page in a notebook: Panel parts beside the map, never around it.

        A lonboard map nested inside a Panel layout fails under Panel's
        ipywidgets and vscode comms; side by side it works under every comm.
        """
        sidebar = pmui.Paper(pmui.Column(*self._build_sidebar(), width=SIDEBAR_WIDTH), theme_config=THEME)
        sidebar.param.watch(self._update_theme, "dark_theme")
        main = ipywidgets.VBox([pn.ipywidget(self._build_toolbar()), self.map,
                                pn.ipywidget(pmui.Column(*self._get_docks("main"), sizing_mode="stretch_width"))],
                               layout=ipywidgets.Layout(width="100%"))
        return ipywidgets.HBox([pn.ipywidget(sidebar), main], layout=ipywidgets.Layout(width="100%"))._repr_mimebundle_(include=include, exclude=exclude)
```

Split today's `__panel__` body into `_build_sidebar()`, `_build_toolbar()` and `_get_docks(region)` so both pages build from the same parts.

Theme: `_apply_theme` also re-reads the basemap (unchanged rule) and layers recolour through their `theme` parameter (Task 4 `_paint`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/geodata/viz/explorer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/geosave/geodata/viz/explorer/app.py tests/geodata/viz/explorer/test_app.py
git commit -m "feat(viz): the explorer draws on one lonboard map, served and in notebooks"
```

---

### Task 6: Compare

**Files:**
- Modify: `src/geosave/geodata/viz/explorer/app.py` (`_build_map_panes`)
- Test: `tests/geodata/viz/explorer/test_app.py` (`TestCompare`)

**Interfaces:**
- Consumes: `RasterLayer.deck`, `create_basemap`, `self.map`
- Produces: comparing shows `pn.Swipe` of two maps whose `view_state` is linked in the browser

- [ ] **Step 1: Write the failing test**

```python
class TestCompare:
    def test_swipes_two_maps_linked_in_the_browser(self, monthly) -> None:
        explorer = Explorer(monthly)
        explorer.compare = True
        left, right = explorer._compare_maps
        assert left.layers[-1] is not right.layers[-1]
        assert explorer._view_link.source == (left, "view_state")

    def test_removing_the_last_raster_ends_the_comparison(self, scene, fields) -> None:
        explorer = Explorer(scene, fields)
        explorer.compare = True
        explorer.run_action(explorer.layers[0], "remove")
        assert explorer.compare is False
        assert list(explorer.map.layers)[-1] in explorer.layers[0].decks
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/geodata/viz/explorer/test_app.py::TestCompare -v`
Expected: FAIL with `AttributeError: ... '_compare_maps'`

- [ ] **Step 3: Implement**

```python
    def _build_map_panes(self, compare, left, right):
        """Return the map, or while comparing, a swipe between two maps that move as one."""
        if not compare or left is None or right is None:
            return [pn.pane.IPyWidget(self.map, height=self.height, sizing_mode="stretch_width")]
        maps = []
        for raster, date in (left, right):
            style, tiles = create_basemap(self.basemap)
            side = RasterLayer(raster._raster.sel({TIME_COORDINATE: date}) if date is not None else raster._raster,
                               name=raster.name, cmap=raster.cmap, alpha=raster.alpha)
            side.variable = raster.variable
            decks = [] if tiles is None else [tiles]
            maps.append(lonboard.Map([*decks, side.deck], basemap=style, view_state=self.map.view_state, height=self.height))
        self._compare_maps = maps
        # Linked in the browser: a pan on one side moves the other without a server round trip
        self._view_link = ipywidgets.jslink((maps[0], "view_state"), (maps[1], "view_state"))
        return [pn.Swipe(*(pn.pane.IPyWidget(m, height=self.height, sizing_mode="stretch_width") for m in maps),
                         sizing_mode="stretch_width")]
```

In `run_action("remove")`, set `self.compare = False` when no raster is left.

Then run the browser check from Task 7 for compare, served and in JupyterLab. If panning one side does not move the other, replace the swipe with a toggle: `pmui.ToggleButtonGroup` choosing which side's deck the one map shows (`self.map.layers = [..., side.deck]`), keep the test for the linked case marked `xfail(reason="jslink of view_state does not cross ipywidgets_bokeh")`, and record that in the spec's Risks.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/geodata/viz/explorer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/geosave/geodata/viz/explorer/app.py tests/geodata/viz/explorer/test_app.py
git commit -m "feat(viz): compare swipes two maps linked in the browser"
```

---

### Task 7: Clean up, docs and the browser check

**Files:**
- Modify: `src/geosave/geodata/viz/__init__.py`, `docs/reference/*.md` and `docs/guides/*.md` that mention `Overview`, datashader on the map or `create_tiles` (`grep -rn "Overview\|create_tiles\|datashader" docs src/geosave/geodata/viz`)
- Modify: `docs/contributing/specs/2026-10-16-viz-lonboard-design.md` (status), `docs/contributing/specs/2026-10-14-viz-holoviz-design.md` (status: map half superseded)
- Delete: `docs/contributing/plans/2026-10-14-viz-holoviz.md` (obsolete plan)

- [ ] **Step 1: Remove what nothing uses**

Run: `uv run ruff check src tests` and `grep -rn "holoviews\|datashader\|spatialpandas" src/geosave/geodata/viz/explorer`
Expected: no HoloViews, datashader or spatialpandas import under `explorer/`. Remove `spatialpandas` from `pyproject.toml` if `plot.py` no longer needs it (`grep -rn spatialpandas src`).

- [ ] **Step 2: Full test run**

Run: `uv run pytest tests/geodata/viz tests/geodata/io -q`
Expected: all pass.

- [ ] **Step 3: Browser check** (not committed; scripts in the session scratchpad, reusing the smoke harness from 2026-10-16)

For each of: `panel serve` page; JupyterLab under default comms; JupyterLab under `pn.extension(comms="vscode")`:
1. `Explorer(catalog_of_remote_s2_item, buildings_200k)`: tick one Item → raster draws; pan 6× → tiles settle < 1.5 s; pan frame median ≤ 20 ms.
2. Change opacity, variable, colormap, basemap (Light, Dark, OpenStreetMap, Imagery) → map updates, zero console errors.
3. Click a building → Identify lists it; search the table → hidden rows vanish from the map.
4. Compare two dates of a dated raster → swipe; pan one side → the other follows (or the toggle fallback from Task 6).
5. Screenshots light and dark; text and categories legible.

Expected: zero console errors and zero jupyter_bokeh sync errors in all three.

- [ ] **Step 4: Update docs and spec status**

Spec status line: `Status: built 2026-10-<day>; compare <swipe|toggle>.` Old spec status: `Map half superseded by [Explorer map on Lonboard](2026-10-16-viz-lonboard-design.md).`

- [ ] **Step 5: Commit**

```bash
git add -A docs src tests pyproject.toml uv.lock
git commit -m "docs(viz): explorer map on lonboard; drop the HoloViews map"
```
