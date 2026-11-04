"""Offline GeoSave benchmark; run from a checkout with its development environment.

Writes raw JSON to --output. Fixtures/output files live in a temporary directory.
Timings are wall time, with one untimed warmup and repeated fresh operations.
Disk caches are not cleared. Run without other benchmarks or test suites active.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


def measure(name, function, check, rows, *, repeats=5, raw_mib=None, tasks=False):
    """Time only the operation; verify its result after stopping the clock."""
    result = function()
    check(result)
    del result
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        elapsed.append(perf_counter() - start)
        check(result)
        del result
    row = {
        "name": name,
        "seconds": elapsed,
        "median_s": statistics.median(elapsed),
        "min_s": min(elapsed),
        "max_s": max(elapsed),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
    }
    if raw_mib is not None:
        row["raw_mib"] = raw_mib
        row["raw_mib_per_s"] = raw_mib / row["median_s"]
    if tasks:
        from dask.callbacks import Callback

        completed = []
        with Callback(posttask=lambda *args: completed.append(args[0])):
            result = function()
        check(result)
        del result
        row["dask_tasks_separate_probe"] = len(completed)
    rows.append(row)
    print(json.dumps(row), flush=True)


def check_shape(result, shape):
    assert tuple(result.shape) == tuple(shape), (result.shape, shape)


def store_worker(kind, native):
    import numpy as np
    import xarray as xr
    from affine import Affine
    from odc.geo.geobox import GeoBox
    import geosave as gs

    rows = []
    pixels = np.random.default_rng(7).integers(1, 10000, (4, 512, 512), dtype="uint16")
    variables = {}
    for i, name in enumerate(["red", "green", "blue", "nir"]):
        variables[name] = (("y", "x"), pixels[i])
    grid = GeoBox((512, 512), Affine(10, 0, 500000, 0, -10, 5005120), "EPSG:32633")
    scene = xr.Dataset(variables).gs.write_grid(grid).gs.write_nodata(0)
    with TemporaryDirectory(prefix="gs-perf-store-") as folder:
        path = Path(folder) / ("scene.zarr" if kind == "zarr" else "scene.nc")

        def write():
            if native:
                if kind == "zarr":
                    return scene.to_zarr(path, mode="w")
                return scene.to_netcdf(path, engine="netcdf4")
            if kind == "zarr":
                return scene.gs.to_zarr(path, overwrite=True)
            return scene.gs.to_netcdf(path, overwrite=True)

        def verify_write(_):
            assert path.exists()

        label = f"{kind}.{'native' if native else 'geosave'}"
        measure(label + ".write", write, verify_write, rows, repeats=3, raw_mib=2)
        print("WRITE_COMPLETE", flush=True)

        def read():
            if native:
                if kind == "zarr":
                    return xr.open_zarr(path, chunks={}, mask_and_scale=False)
                return xr.open_dataset(
                    path, engine="netcdf4", chunks={}, mask_and_scale=False
                )
            return gs.open_raster(path)

        def verify_read(opened):
            try:
                np.testing.assert_array_equal(opened.red.values, pixels[0])
                assert opened.gs.geobox == grid
            finally:
                opened.close()

        measure(label + ".open", read, verify_read, rows, repeats=3)

        def load():
            with read() as opened:
                return opened.load()

        measure(
            label + ".open_load",
            load,
            lambda r: np.testing.assert_array_equal(r.red, pixels[0]),
            rows,
            repeats=3,
            raw_mib=2,
        )
    return rows


def raster_worker(sizes, repeats):
    import dask
    import numpy as np
    import rasterio
    import torch
    import xarray as xr
    from affine import Affine
    from odc.geo.geobox import GeoBox
    import geosave as gs

    torch.set_num_threads(1)
    dask.config.set(scheduler="threads", num_workers=4)
    rows = []
    names = ["red", "green", "blue", "nir"]
    with TemporaryDirectory(prefix="gs-perf-") as folder:
        for size in sizes:
            print(f"FIXTURE {size}", flush=True)
            pixels = np.random.default_rng(7).integers(
                1, 10000, (4, size, size), dtype="uint16"
            )
            grid = GeoBox(
                (size, size),
                Affine(10, 0, 500000, 0, -10, 5000000 + size * 10),
                "EPSG:32633",
            )
            variables = {}
            for i, name in enumerate(names):
                variables[name] = (("y", "x"), pixels[i])
            scene = xr.Dataset(variables).gs.write_grid(grid).gs.write_nodata(0)
            lazy = scene.chunk({"y": 256, "x": 256})
            tree = lazy.gs.to_datatree("optical")
            coarse_grid = grid.zoom_out(2)
            coarse = (
                xr.Dataset(
                    {
                        "height": (
                            ("y", "x"),
                            np.full(tuple(coarse_grid.shape), 12, dtype="uint16"),
                        )
                    }
                )
                .gs.write_grid(coarse_grid)
                .gs.write_nodata(0)
                .chunk({"y": 256, "x": 256})
            )
            mixed = xr.merge([tree, coarse.gs.to_datatree("optical")])
            mib = pixels.nbytes / 2**20
            prefix = f"{size}."

            def timed(name, fn, check, *, materialized=False, tasks=False):
                measure(
                    prefix + name,
                    fn,
                    check,
                    rows,
                    repeats=repeats,
                    raw_mib=mib if materialized else None,
                    tasks=tasks,
                )

            def same_grid(result):
                assert result.gs.geobox == grid

            timed(
                "to_datatree",
                lambda: lazy.gs.to_datatree("optical"),
                lambda t: same_grid(t["optical/r10m"]),
                tasks=True,
            )
            timed(
                "to_dataset.same_grid",
                lambda: tree["optical"].gs.to_dataset(),
                same_grid,
                tasks=True,
            )
            timed(
                "to_dataset.multires.plan",
                lambda: mixed["optical"].gs.to_dataset(),
                same_grid,
                tasks=True,
            )
            merged = mixed["optical"].gs.to_dataset()
            timed(
                "to_dataset.multires.compute",
                lambda: merged.compute(),
                lambda r: (
                    same_grid(r),
                    np.testing.assert_array_equal(r.red, pixels[0]),
                    np.testing.assert_array_equal(r.height, 12),
                ),
                materialized=True,
                tasks=True,
            )
            timed(
                "to_numpy.eager",
                scene.gs.to_numpy,
                lambda r: np.testing.assert_array_equal(r, pixels),
                materialized=True,
            )
            timed(
                "to_tensor.eager.uint16",
                scene.gs.to_tensor,
                lambda r: (
                    check_shape(r, pixels.shape),
                    np.testing.assert_array_equal(r.numpy(), pixels),
                ),
                materialized=True,
            )
            timed(
                "to_tensor.eager.float32",
                lambda: scene.gs.to_tensor(dtype="float32"),
                lambda r: (
                    assert_float(r),
                    np.testing.assert_array_equal(r.numpy(), pixels),
                ),
                materialized=True,
            )
            timed(
                "to_tensor.lazy.uint16",
                lazy.gs.to_tensor,
                lambda r: np.testing.assert_array_equal(r.numpy(), pixels),
                materialized=True,
                tasks=True,
            )
            timed(
                "to_tensor.tree",
                tree.gs.to_tensor,
                lambda r: np.testing.assert_array_equal(
                    r["optical/r10m"].numpy(), pixels
                ),
                materialized=True,
                tasks=True,
            )
            timed(
                "baseline.numpy_stack_torch",
                lambda: torch.from_numpy(np.stack([scene[n].values for n in names])),
                lambda r: np.testing.assert_array_equal(r.numpy(), pixels),
                materialized=True,
            )
            timed(
                "baseline.xarray_stack",
                lambda: scene.to_array(dim="band").values,
                lambda r: np.testing.assert_array_equal(r, pixels),
                materialized=True,
            )
            packed = lazy.copy()
            for n in names:
                packed[n].attrs["scale_factor"] = 0.0001
            timed(
                "decode.plan",
                packed.gs.mask_and_scale,
                lambda r: same_grid(r),
                tasks=True,
            )
            decoded = packed.gs.mask_and_scale()
            timed(
                "decode.compute",
                decoded.compute,
                lambda r: np.testing.assert_allclose(
                    r.red, pixels[0] * 0.0001, rtol=1e-6
                ),
                materialized=True,
                tasks=True,
            )
            timed(
                "ndvi.plan",
                lambda: gs.features.ndvi(packed, nir="nir", red="red"),
                same_grid,
                tasks=True,
            )
            ndvi = gs.features.ndvi(packed, nir="nir", red="red")
            expected = (pixels[3].astype("float32") - pixels[0]) / (
                pixels[3].astype("float32") + pixels[0]
            )
            timed(
                "ndvi.compute",
                ndvi.compute,
                lambda r: np.testing.assert_allclose(r, expected, atol=2e-7),
                materialized=True,
                tasks=True,
            )
            target = grid.zoom_out(2)
            timed(
                "reproject.plan",
                lambda: lazy.gs.reproject(target, resampling="nearest"),
                lambda r: assert_grid(r, target),
                tasks=True,
            )
            warped = lazy.gs.reproject(target, resampling="nearest")
            timed(
                "reproject.compute",
                warped.compute,
                lambda r: (
                    assert_grid(r, target),
                    np.testing.assert_array_equal(r.red, pixels[0, 1::2, 1::2]),
                ),
                materialized=True,
                tasks=True,
            )
            timed(
                "chips.layout",
                lambda: gs.Chips([tree], size=256, resolution=10),
                lambda c: assert_count(c, (size // 256) ** 2),
                tasks=True,
            )
            chips = gs.Chips([tree], size=256, resolution=10)
            timed(
                "chips.select",
                lambda: chips[0],
                lambda r: assert_grid(r[1], grid[:256, :256]),
                tasks=True,
            )
            chip = chips[0][1]["optical"].gs.to_dataset()
            timed(
                "chips.to_tensor",
                chip.gs.to_tensor,
                lambda r: np.testing.assert_array_equal(
                    r.numpy(), pixels[:, :256, :256]
                ),
                tasks=True,
            )
            destination = Path(folder) / str(size)
            path = destination / "scene.tif"

            def write():
                return scene.gs.to_cog(
                    destination, path_format="scene.tif", overwrite=True
                )

            def check_write(paths):
                assert list(paths)[0].exists()

            timed("cog.write", write, check_write, materialized=True)
            with gs.open_raster(path) as result:
                for i, n in enumerate(names):
                    np.testing.assert_array_equal(result[n], pixels[i])
                same_grid(result)
            rows.append(
                {
                    "name": prefix + "cog.bytes",
                    "file_bytes": path.stat().st_size,
                    "raw_bytes": pixels.nbytes,
                }
            )

            def verify_open(result):
                same_grid(result)
                result.close()

            timed("cog.open", lambda: gs.open_raster(path), verify_open)

            def load():
                with gs.open_raster(path) as opened:
                    return opened.load()

            timed(
                "cog.open_load",
                load,
                lambda r: np.testing.assert_array_equal(r.red, pixels[0]),
                materialized=True,
                tasks=True,
            )

            def tensor():
                with gs.open_raster(path) as opened:
                    return opened[names].gs.to_tensor()

            timed(
                "cog.open_tensor",
                tensor,
                lambda r: np.testing.assert_array_equal(r.numpy(), pixels),
                materialized=True,
                tasks=True,
            )

            def window():
                with gs.open_raster(path) as opened:
                    return (
                        opened[names]
                        .isel(y=slice(0, 256), x=slice(0, 256))
                        .gs.to_tensor()
                    )

            timed(
                "cog.window_tensor",
                window,
                lambda r: np.testing.assert_array_equal(
                    r.numpy(), pixels[:, :256, :256]
                ),
                tasks=True,
            )

            def native_read():
                with rasterio.open(path) as opened:
                    return opened.read()

            timed(
                "baseline.rasterio.read",
                native_read,
                lambda r: np.testing.assert_array_equal(r, pixels),
                materialized=True,
            )
    return rows


def assert_float(value):
    import torch

    assert value.dtype == torch.float32


def assert_grid(value, grid):
    assert value.gs.geobox == grid


def assert_count(value, count):
    assert len(value) == count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("performance.json"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--store", choices=["zarr", "netcdf"])
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.store:
        store_worker(args.store, args.native)
        return
    versions = {}
    for name in [
        "geosave",
        "numpy",
        "xarray",
        "dask",
        "rasterio",
        "rioxarray",
        "odc-geo",
        "torch",
        "zarr",
        "netCDF4",
    ]:
        versions[name] = importlib.metadata.version(name)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": versions,
        "torch_threads": 1,
        "dask_workers": 4,
        "dask_scheduler": "threads",
        "warmup": 1,
        "repeats": args.repeats,
        "seed": 7,
        "sizes": args.sizes,
        "bands": 4,
        "dtype": "uint16",
        "chunks": 256,
        "cache": "OS page cache not dropped; repeated warm-cache reads",
        "memory": "process lifetime peak RSS, not per-operation allocation",
    }
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            environment["cpu"] = line.split(":", 1)[1].strip()
            break
    root = Path(__file__).resolve().parents[4]
    digest = hashlib.sha256()
    count = 0
    files = sorted((root / "src").rglob("*")) + [
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    for path in files:
        if path.is_file() and path.suffix in {".py", ".yaml", ".toml", ".lock"}:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
            count += 1
    environment["source_sha256"] = digest.hexdigest()
    environment["source_files_hashed"] = count
    environment["thread_environment"] = {
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    results = {"environment": environment, "imports": [], "rows": [], "stores": []}
    for module in ["geosave", "torch"]:
        times = []
        for _ in range(3):
            command = f"import time; start=time.perf_counter(); import {module}; print(time.perf_counter()-start)"
            run = subprocess.run(
                [sys.executable, "-c", command],
                capture_output=True,
                text=True,
                check=True,
                timeout=45,
            )
            times.append(float(run.stdout.strip()))
        results["imports"].append(
            {
                "module": module,
                "fresh_process_seconds": times,
                "median_s": statistics.median(times),
            }
        )
    results["rows"] = raster_worker(args.sizes, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for kind in ["netcdf", "zarr"]:
        for native in [False, True]:
            command = [sys.executable, __file__, "--store", kind]
            if native:
                command.append("--native")
            try:
                run = subprocess.run(
                    command, capture_output=True, text=True, timeout=40
                )
                record = {
                    "format": kind,
                    "native": native,
                    "exit_code": run.returncode,
                    "stdout": run.stdout,
                    "stderr": run.stderr,
                }
            except subprocess.TimeoutExpired as error:
                output = error.stdout or b""
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                record = {
                    "format": kind,
                    "native": native,
                    "timeout_s": 40,
                    "stdout": output,
                }
            results["stores"].append(record)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print("STORE", kind, native, record.get("exit_code", "timeout"), flush=True)


if __name__ == "__main__":
    main()
