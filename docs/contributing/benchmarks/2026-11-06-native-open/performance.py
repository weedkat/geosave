"""Compare native and ODC loading on aligned local TIFFs; no resampling.

Run from the checkout with .venv/bin/python and --output results.json.
Warm filesystem caches, one warmup, rotated execution order, five repeats.
Setup and compute are timed separately; validation and task counting are untimed.
"""

from datetime import UTC, datetime, timedelta
from importlib.metadata import version
import argparse
import json
from pathlib import Path
import platform
import statistics
from tempfile import TemporaryDirectory
from time import perf_counter

import dask
import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pystac
import rasterio
from affine import Affine
from odc.geo.geobox import GeoBox
import xarray as xr

from geosave.geodata import io
from geosave.geodata.stac.catalog import create_catalog


def fixtures(folder, size, count):
    grid = GeoBox((size, size), Affine(10, 0, 500000, 0, -10, 9000000), "EPSG:32749")
    footprint = grid.extent.to_crs("EPSG:4326")
    items = []
    expected = []
    rng = np.random.default_rng(42)
    for index in range(count):
        entry = pystac.Item(
            str(index),
            footprint.json,
            list(footprint.boundingbox),
            datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=index),
            {},
            collection="local",
        )
        entry.stac_extensions = [
            "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
        ]
        values = {}
        for name in ("red", "green", "nir"):
            pixels = rng.integers(1, 10000, (size, size), dtype="uint16")
            values[name] = pixels
            path = folder / f"{index}-{name}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=size,
                height=size,
                count=1,
                dtype="uint16",
                crs=str(grid.crs),
                transform=grid.transform,
                nodata=65535,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
            ) as file:
                file.write(pixels, 1)
            entry.add_asset(
                name,
                pystac.Asset(
                    str(path),
                    media_type=pystac.MediaType.GEOTIFF,
                    roles=["data"],
                    extra_fields={
                        "proj:code": str(grid.crs),
                        "proj:shape": list(grid.shape),
                        "proj:transform": list(grid.transform)[:6],
                        "raster:bands": [{"data_type": "uint16", "nodata": 65535}],
                    },
                ),
            )
        items.append(entry)
        expected.append(values)
    return items, expected, grid


def dataset(result):
    if isinstance(result, xr.DataTree):
        rasters = list(result.gs.datasets.values())
        assert len(rasters) == 1
        return rasters[0]
    return result


def check(result, expected, grid, lazy):
    raster = dataset(result)
    assert raster.gs.geobox == grid
    assert set(raster.data_vars) == {"red", "green", "nir"}
    for name in raster.data_vars:
        assert raster[name].dtype == np.dtype("uint16")
        assert isinstance(raster[name].data, da.Array) == lazy
        if not lazy:
            values = raster[name].values
            if values.ndim == 2:
                values = values[None]
            assert values.shape[0] == len(expected)
            for index, observation in enumerate(expected):
                np.testing.assert_array_equal(values[index], observation[name])


def run(open_result, expected, grid):
    start = perf_counter()
    result = open_result()
    setup = perf_counter() - start
    try:
        check(result, expected, grid, lazy=True)
        start = perf_counter()
        computed = result.compute()
        compute = perf_counter() - start
        check(computed, expected, grid, lazy=False)
    finally:
        start = perf_counter()
        result.close()
        close = perf_counter() - start
    return {
        "setup_s": setup,
        "compute_s": compute,
        "close_s": close,
        "total_s": setup + compute + close,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--items", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    chunks = {"x": 512, "y": 512}
    with TemporaryDirectory(prefix="geosave-native-perf-") as temporary:
        items, expected, grid = fixtures(Path(temporary), args.size, args.items)
        for count in (1, args.items):
            selected = items[:count]
            values = expected[:count]
            table = create_catalog(selected)

            def native():
                if count == 1:
                    return io.open_item(selected[0], chunks=chunks)
                return table.gs.to_datatree(chunks=chunks)

            operations = {
                "native": native,
                "odc_dataset": lambda: io.load_dataset(
                    selected,
                    geobox=grid,
                    chunks=chunks,
                    dtype="uint16",
                    nodata=65535,
                    groupby="id",
                ),
                "odc_datatree": lambda: io.load_datatree(
                    selected,
                    geobox=grid,
                    chunks=chunks,
                    dtype="uint16",
                    nodata=65535,
                    groupby="id",
                ),
            }
            measurements = {name: [] for name in operations}
            for operation in operations.values():
                run(operation, values, grid)
            names = list(operations)
            for repeat in range(args.repeats):
                order = names[repeat % len(names) :] + names[: repeat % len(names)]
                for name in order:
                    measurements[name].append(run(operations[name], values, grid))
            for name, operation in operations.items():
                tasks = []
                with Callback(posttask=lambda *args: tasks.append(args[0])):
                    result = operation()
                    assert tasks == [], "setup executed Dask tasks"
                    try:
                        result.compute()
                    finally:
                        result.close()
                samples = measurements[name]
                medians = {}
                for key in samples[0]:
                    medians[key] = statistics.median(sample[key] for sample in samples)
                row = {
                    "items": count,
                    "method": name,
                    "raw_mib": count * 3 * args.size**2 * 2 / 2**20,
                    "median": medians,
                    "samples": samples,
                    "executed_tasks": len(tasks),
                }
                rows.append(row)
                print(
                    json.dumps(
                        {key: value for key, value in row.items() if key != "samples"}
                    ),
                    flush=True,
                )
    packages = {}
    for name in ("xarray", "dask", "rasterio", "odc-stac", "odc-loader", "odc-geo"):
        packages[name] = version(name)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": packages,
        "size": args.size,
        "chunks": chunks,
        "threads": 4,
        "repeats": args.repeats,
        "cache": "warm; OS caches not cleared",
        "checks": "all bands equal fixture pixels, exact grid, uint16, lazy setup",
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    with dask.config.set(scheduler="threads", num_workers=4):
        main()
