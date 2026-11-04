# The interactive explorer

```python
scene.gs.explore("ndvi", basemap="Imagery")
```

`explore` returns a native Panel Material UI app with a light/dark theme toggle.
The compact sidebar groups catalogs, rasters and vectors separately, with
visibility and zoom/remove actions on each layer. Style, Info and Identify tabs
contain the selected layer’s controls, legend and metadata.
The map sits beside it; a collapsible catalog or attribute table sits below
the map. A row's checkbox draws its polygon; a catalog's **raster** column
opens that row's raster with a click and removes it with another. Clicking the
map picks features without opening rasters. Column filters and date ranges
retain original row identity. Table guidance explains the two controls; checking
Raster again retries a failed load. **Fit visible layers** frames the layers
currently displayed, respecting date and comparison selection.

A raster opens with its declared RGB channels, otherwise its first spatial
band. Explicit `band` or `rgb` arguments take priority. The **Display** dropdown lists
each spatial band and **RGB** when a declared or explicit composite exists. Each raster
keeps its own choices, opacity and stretch. While a raster has dates, the
sidebar's **Date** slider steps the whole map by year, month or day: each dated
raster shows its last date in that period, rasters without one there are
hidden, and timeless rasters always show. **Color by** selects a vector
attribute. The optional comparison card switches between two open rasters.

The map uses Panel, HoloViews, GeoViews and Bokeh. Dense point, line and polygon layers
use Datashader for numeric, categorical and single-color rendering. Imagery, light, dark and OpenStreetMap basemaps include their
providers' attribution. Basemaps need network access; the explorer needs a
live notebook kernel or `panel serve`.

Each raster reads the pixels around the view at one web-map zoom level,
through `.gs.read_pixels`: a catalog row reads each file's window from the
overview its pixel size needs; any other Dataset crops and keeps every n-th
pixel before computing, so a zoomed-out computed raster still computes the
chunks under the view. Pans within the cached area at the same zoom level
and opacity changes neither read nor resend pixels; hidden rasters do no work.
Recent viewport windows share a 128 MiB pixel-cache budget per raster layer,
so returning to a retained zoom avoids another source read. Returning to a
zoom can still transfer an image to the browser. Changing bands, date, or
nonspatial axes invalidates those windows. The active image remains reusable
even when larger than the history budget; rendered arrays and the RGB overview
also occupy memory outside that budget. Removing a raster or closing the
explorer clears its caches. Set `layer.cache_bytes = 0` to retain only the active
window, or use `layer.clear_cache()` to release all its cached buffers.
Percentile sliders apply on release.
RGB stretches each channel separately using a cached scene overview of at most
256 × 256 pixels. Pan and zoom keep the same color mapping. Changing the RGB
bands, date or selected axis refreshes the overview; changing the percentile
slider reuses it. **Advanced contrast** sets one manual range for all channels.
The RGB overview samples the whole scene and can touch every source chunk of a
computed raster, while keeping the sampled array bounded.
Display choices do not mutate the source raster or its RGB metadata.

Layers compose with `scene.gs.explore() * prediction * fields`, or with
`Explorer(scene, prediction, fields)`; `create_layer(fields, name="fields")`
names one. Two explorers also compose:
`raster1.gs.explore() * raster2.gs.explore()` lists both under Rasters, with
independent display settings and no catalog table. Adding a catalog shows its
checkbox table; adding a vector shows its attribute table. For local serving, put
`scene.gs.explore().servable()` in a script and run `panel serve explore.py`.

To launch directly from a notebook or Python script:

```python
server = catalog.gs.explore().serve()
```

This starts a local server at `http://localhost:5006`. In notebooks it runs in a
background thread, so the cell stays free, and opens a browser tab. Rerunning the
cell replaces the server the earlier explorer started on that port. In scripts it keeps the process
running until you press Ctrl+C. Google Colab uses its kernel-port proxy to
display the served explorer in an iframe; keep the Colab runtime connected.
Colab support requires its hosted frontend and is not a public deployment.

If another program holds the port, `.serve()` raises `OSError`; pass
`.serve(port=5007)`. Use `.serve(open=False)` to start
without opening a tab or iframe. Local connections accept both `localhost` and
`127.0.0.1`. For remote Jupyter kernels, forward the chosen port to your browser's
machine. Stop a notebook server with `explorer.stop()`.
