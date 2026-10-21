# The interactive explorer

```python
scene.gs.explore("ndvi", basemap="Imagery")
```

`explore` returns a native Panel Material UI app with a light/dark theme toggle.
The compact sidebar groups catalogs, rasters and vectors separately, with
visibility and zoom/remove actions on each layer. Style, Info and Identify tabs
contain the selected layer’s controls, legend and metadata.
The map sits beside it; a collapsible catalog or attribute table sits below
the map. Checking catalog rows opens their rasters automatically; unchecking
removes them. Column filters and date ranges retain original row identity.

A raster opens with its declared RGB channels, otherwise its first spatial
band. Explicit `band` or `rgb` arguments take priority. The **Display** dropdown lists
each spatial band and **RGB** when a declared or explicit composite exists. Each raster
keeps its own choices, opacity, stretch and date. **Color by** selects a vector
attribute. The optional comparison card switches between two open rasters.

The map uses Panel, HoloViews, GeoViews and Bokeh. Dense point, line and polygon layers
use Datashader for numeric, categorical and single-color rendering. Imagery, light, dark and OpenStreetMap basemaps include their
providers' attribution. Basemaps need network access; the explorer needs a
live notebook kernel or `panel serve`.

COG display reads windows and overviews before reprojection. Computed xarray
rasters select and sample the viewport before Dask execution. A zoomed-out
computed overview can still execute every source chunk; the explorer does
not create a pyramid. Opacity and visibility changes reuse displayed raster
pixels; hidden rasters do no viewport work. Percentile sliders apply on release.
Display choices do not mutate the source raster or its RGB metadata.

Layers compose with `scene.gs.explore() * prediction * fields`, or with
`scene.gs.explore(overlay={"fields": fields})`. Two explorers also compose:
`raster1.gs.explore() * raster2.gs.explore()` lists both under Rasters, with
independent display settings and no catalog table. Adding a catalog shows its
checkbox table; adding a vector shows its attribute table. For local serving, put
`scene.gs.explore().servable()` in a script and run `panel serve explore.py`.

To launch directly from a notebook or Python script:

```python
server = catalog.gs.explore().serve()
```

This starts a local server at `http://localhost:5006`. In notebooks it runs in a
background thread and opens a browser tab. In scripts it keeps the process
running until you press Ctrl+C. Google Colab uses its kernel-port proxy to
display the served explorer in an iframe; keep the Colab runtime connected.
Colab support requires its hosted frontend and is not a public deployment.

Use `.serve(port=5007)` if port 5006 is occupied, or `.serve(open=False)` to start
without opening a tab or iframe. Local connections accept both `localhost` and
`127.0.0.1`. For remote Jupyter kernels, forward the chosen port to your browser's
machine. Stop a notebook server with `server.stop()`.
