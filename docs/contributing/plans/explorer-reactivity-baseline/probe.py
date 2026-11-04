"""Read-only application profiling; outputs JSON, no browser or network."""

from collections import Counter
from contextlib import ExitStack
from functools import wraps
from importlib.metadata import version
import json
from statistics import median
from time import perf_counter
from unittest.mock import patch

import dask.array as da
from dask.callbacks import Callback
from bokeh.document import Document
import geopandas as gpd
import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from shapely.geometry import box
import xarray as xr
from geosave.geodata.viz.explore import Explorer, RasterLayer, VectorLayer
from geosave.geodata.viz.explore.page.map import Map
from geosave.geodata.viz.explore.page.sidebar import LayerList
from geosave.geodata.viz.explore.page import sidebar

counts = Counter()


def instrument(owner, name, label):
    original = getattr(owner, name)

    @wraps(original)
    def measured(*args, **kwargs):
        counts[label] += 1
        return original(*args, **kwargs)

    return patch.object(owner, name, measured)


grid = GeoBox.from_bbox((0, 0, 40960, 40960), crs=3857, shape=(256, 256), tight=True)
y, x = da.indices((256, 256), chunks=(64, 64))
values = ((x + y) / 512).astype("float32")
scene = xr.Dataset(
    {
        "red": xr.DataArray(values, dims=("y", "x"), coords=xr_coords(grid)),
        "green": xr.DataArray(values * 0.8, dims=("y", "x"), coords=xr_coords(grid)),
        "blue": xr.DataArray(values * 0.6, dims=("y", "x"), coords=xr_coords(grid)),
    }
).gs.write_rgb("red", "green", "blue")
fields = gpd.GeoDataFrame(
    {"crop": ["rice", "palm"], "area": [1.0, 2.0]},
    geometry=[box(1000, 1000, 3000, 3000), box(5000, 5000, 9000, 9000)],
    crs=3857,
)
report = {
    "versions": {},
    "fixture": "256x256 lazy RGB; chunks 64x64; two polygons; full Panel page; no browser",
    "operations": [],
}
for name in ["panel", "param", "holoviews", "geoviews", "bokeh"]:
    report["versions"][name] = version(name)
with ExitStack() as stack:
    for owner, name, label in [
        (Explorer, "_offer_periods", "period_updates"),
        (Map, "_draw", "map_draws"),
        (LayerList, "_list_entries", "menu_builds"),
        (sidebar, "format_legend", "legend_formats"),
        (VectorLayer, "render", "vector_renders"),
        (RasterLayer, "_read_pixels", "pixel_reader_calls"),
    ]:
        stack.enter_context(instrument(owner, name, label))
    viewer = Explorer(scene, fields, basemap="None", height=400)
    raster, vector = viewer.layers
    document = Document()
    root = viewer.__panel__().get_root(document)
    document.add_root(root)
    # Materialize both style panels, then return to raster.
    viewer.active_layer = vector
    viewer.active_layer = raster
    updates = []

    def record(event):
        if getattr(event, "attr", None) == "data":
            data = getattr(event.model, "data", {})
            if "image" in data:
                size = 0
                for image in data["image"]:
                    size += np.asarray(image).nbytes
                updates.append(size)

    document.on_change(record)
    initial_x = viewer.map.ranges.x_range
    initial_y = viewer.map.ranges.y_range

    def pan(i):
        dx = (initial_x[1] - initial_x[0]) * 0.001 * (i + 1)
        viewer.map.ranges.event(
            x_range=(initial_x[0] + dx, initial_x[1] + dx), y_range=initial_y
        )

    def zoom(scale):
        cx = sum(initial_x) / 2
        cy = sum(initial_y) / 2
        hx = (initial_x[1] - initial_x[0]) * scale / 2
        hy = (initial_y[1] - initial_y[0]) * scale / 2
        viewer.map.ranges.event(x_range=(cx - hx, cx + hx), y_range=(cy - hy, cy + hy))

    def measure(name, action, repeats=5):
        elapsed = []
        rows = []
        for i in range(repeats):
            counts.clear()
            updates.clear()
            tasks = []
            before = len(document.models)
            started = perf_counter()
            with Callback(pretask=lambda key, *_: tasks.append(key)):
                action(i)
            elapsed.append((perf_counter() - started) * 1000)
            row = dict(counts)
            row.update(
                dask_tasks=len(tasks),
                image_updates=len(updates),
                image_array_bytes=sum(updates),
                model_count_change=len(document.models) - before,
                pixel_cache_bytes=raster._cached_pixels.nbytes
                if raster._cached_pixels is not None
                else 0,
            )
            rows.append(row)
        report["operations"].append(
            {
                "name": name,
                "repeats": repeats,
                "median_ms": round(median(elapsed), 3),
                "min_ms": round(min(elapsed), 3),
                "max_ms": round(max(elapsed), 3),
                "work": rows,
            }
        )

    measure("raster opacity", lambda i: setattr(raster, "alpha", 0.3 + i * 0.05))
    measure(
        "vector color",
        lambda i: setattr(vector, "color", ["#112233", "#334455"][i % 2]),
    )
    measure(
        "active layer",
        lambda i: setattr(viewer, "active_layer", vector if i % 2 == 0 else raster),
    )
    measure("cached pan", pan)
    measure("zoom in", lambda i: zoom(0.25), 1)
    measure("return to original zoom", lambda i: zoom(1), 1)
    raster.visible = False
    measure("hidden raster pan", lambda i: zoom(0.2 + i * 0.02))
    viewer.close()
print(json.dumps(report, indent=2))
