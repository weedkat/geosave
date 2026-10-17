"""Measure real COG byte ranges, computed chunks and glyph-only changes.

Run `uv run python scripts/measure_explorer.py`. Fixture creation is excluded
from display timings. Output is JSON Lines; browser latency is measured by
`tests/geodata/viz/explorer/test_browser.py` separately.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from multiprocessing import get_context
from time import perf_counter

import dask.array as da
import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from dask.callbacks import Callback
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from rasterio.enums import ColorInterp
from shapely.geometry import Point

import geosave  # noqa: F401
from geosave.geodata.io import open_raster
from geosave.geodata.viz.explorer import Explorer, RasterLayer, VectorLayer
from geosave.geodata.viz.explorer.raster import read_window


def serve_range(path, connection):
    """Serve ranges in a separate process so GDAL cannot block its Python server."""

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            if self.path != "/image.tif":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            if self.path != "/image.tif":
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            byte_range = self.headers.get("Range")
            if byte_range:
                first, last = byte_range.removeprefix("bytes=").split("-")
                start, end = (
                    int(first),
                    min(int(last) if last else size - 1, size - 1),
                )
            self.send_response(206 if byte_range else 200)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as file:
                file.seek(start)
                self.wfile.write(file.read(end - start + 1))
            connection.send(end - start + 1)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    connection.send(server.server_port)
    server.serve_forever()


def measure():
    """Print measured work for synthetic but full-size source rasters."""
    size = int(os.environ.get("GEOSAVE_PROOF_SIZE", "8192"))
    grid = GeoBox.from_bbox(
        (-1000000, 5000000, -900000, 5100000), crs=3857, shape=(size, size), tight=True
    )
    requests = []
    source_reads = []
    with TemporaryDirectory(prefix="geosave-measure-") as directory:
        path = Path(directory) / "image.tif"
        with rasterio.open(
            path,
            "w",
            driver="COG",
            height=size,
            width=size,
            count=3,
            dtype="uint16",
            crs=grid.crs,
            transform=grid.affine,
            compress="DEFLATE",
            nodata=0,
        ) as file:
            file.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            for _, window in file.block_windows(1):
                y, x = np.mgrid[: int(window.height), : int(window.width)]
                pixels = []
                for factor in (1, 2, 3):
                    pixels.append(
                        (
                            (x + window.col_off + y + window.row_off) * factor % 10000
                            + 1
                        ).astype("uint16")
                    )
                file.write(np.stack(pixels), window=window)

        context = get_context("spawn")
        connection, child = context.Pipe()
        server = context.Process(target=serve_range, args=(path, child), daemon=True)
        server.start()
        port = connection.recv()

        def read_requests():
            values = []
            while connection.poll():
                values.append(connection.recv())
            return values

        try:
            data = open_raster(f"http://127.0.0.1:{port}/image.tif")
            requests = read_requests()
            print(
                json.dumps(
                    {
                        "phase": "COG metadata",
                        "requests": len(requests),
                        "bytes": sum(requests),
                        "file_bytes": path.stat().st_size,
                    }
                ),
                flush=True,
            )
            for label, bounds in [
                ("COG overview", (-1000000, 5000000, -900000, 5100000)),
                ("COG zoom", (-980000, 5020000, -970000, 5030000)),
            ]:
                requests.clear()
                tasks = []
                start = perf_counter()
                with Callback(pretask=lambda key, *_: tasks.append(key)):
                    read_window(
                        data,
                        GeoBox.from_bbox(
                            bounds, crs=3857, shape=(450, 650), tight=True
                        ),
                    )
                requests = read_requests()
                print(
                    json.dumps(
                        {
                            "phase": label,
                            "seconds": perf_counter() - start,
                            "requests": len(requests),
                            "bytes": sum(requests),
                            "dask_tasks": len(tasks),
                        }
                    ),
                    flush=True,
                )

            def generate(block, block_info=None):
                if not block.size:
                    return block.astype("float32")
                source_reads.append(block.size * 4)
                return np.linspace(0, 1, block.size, dtype="float32").reshape(
                    block.shape
                )

            values = da.empty((size, size), chunks=(256, 256)).map_blocks(
                generate, dtype="float32"
            )
            computed = xr.Dataset(
                {
                    name: xr.DataArray(
                        values * factor, dims=("y", "x"), coords=xr_coords(grid)
                    )
                    for name, factor in [("red", 1), ("green", 2), ("blue", 3)]
                }
            ).gs.write_rgb("red", "green", "blue")
            layer = RasterLayer(computed)
            for label, x_range, y_range in [
                ("computed overview", (-1000000, -900000), (5000000, 5100000)),
                ("computed zoom", (-980000, -970000), (5020000, 5030000)),
            ]:
                source_reads.clear()
                tasks = []
                start = perf_counter()
                with Callback(pretask=lambda key, *_: tasks.append(key)):
                    layer.render(x_range, y_range)
                print(
                    json.dumps(
                        {
                            "phase": label,
                            "seconds": perf_counter() - start,
                            "source_chunks": len(source_reads),
                            "source_bytes": sum(source_reads),
                            "dask_tasks": len(tasks),
                        }
                    ),
                    flush=True,
                )
            fields = gpd.GeoDataFrame(
                {"value": [1.0, 2.0]},
                geometry=[Point(-990000, 5010000), Point(-970000, 5030000)],
                crs=3857,
            )
            vector = VectorLayer(fields)
            cog = RasterLayer(data, name="COG")
            viewer = Explorer(layer, cog, vector, basemap="None")
            viewer.map.get_root()
            for label, raster in [("computed contrast", layer), ("COG contrast", cog)]:
                read_requests()
                source_reads.clear()
                tasks = []
                start = perf_counter()
                with Callback(pretask=lambda key, *_: tasks.append(key)):
                    raster.stretch = (5, 95)
                    raster.clim = (0.1, 1.5)
                requests = read_requests()
                print(
                    json.dumps(
                        {
                            "phase": label,
                            "seconds": perf_counter() - start,
                            "source_chunks": len(source_reads),
                            "http_bytes": sum(requests),
                            "dask_tasks": len(tasks),
                        }
                    ),
                    flush=True,
                )
            read_requests()
            source_reads.clear()
            tasks = []
            start = perf_counter()
            with Callback(pretask=lambda key, *_: tasks.append(key)):
                layer.alpha = 0.3
                cog.alpha = 0.3
                vector.column = "value"
            requests = read_requests()
            print(
                json.dumps(
                    {
                        "phase": "opacity and vector color",
                        "seconds": perf_counter() - start,
                        "source_chunks": len(source_reads),
                        "http_bytes": sum(requests),
                        "dask_tasks": len(tasks),
                    }
                ),
                flush=True,
            )
            layer.visible = cog.visible = False
            read_requests()
            source_reads.clear()
            tasks = []
            start = perf_counter()
            with Callback(pretask=lambda key, *_: tasks.append(key)):
                viewer.zoom(layer.bbox.buffered(-20000, -20000))
            requests = read_requests()
            print(
                json.dumps(
                    {
                        "phase": "hidden raster pan",
                        "seconds": perf_counter() - start,
                        "source_chunks": len(source_reads),
                        "http_bytes": sum(requests),
                        "dask_tasks": len(tasks),
                    }
                ),
                flush=True,
            )
            viewer.close()
            data.close()
        finally:
            server.terminate()
            server.join()
            connection.close()
            child.close()


if __name__ == "__main__":
    measure()
