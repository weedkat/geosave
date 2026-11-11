"""Four-reader feasibility study with exact chip equality and isolated RSS.

Run: .venv/bin/python docs/contributing/benchmarks/2026-11-06-chip-readers/benchmark.py
Each measurement is a fresh subprocess. Timings exclude imports/verification.
"""

import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

METHODS = ("chips", "blocks", "rasterio", "npy")
BLOCK = 1024
CACHE = 32 * 2**20


def values(top, left, height, width):
    """Coordinate-derived pixels, including valid zeroes and explicit nodata."""
    y = np.arange(top, top + height, dtype="uint32")[:, None]
    x = np.arange(left, left + width, dtype="uint32")[None, :]
    result = []
    for band in range(3):
        hashed = (y * np.uint32(73856093)) ^ (x * np.uint32(19349663))
        hashed ^= np.uint32((band + 1) * 83492791)
        pixels = (hashed % 10000).astype("uint16")
        pixels[(x + y + band) % 997 == 0] = 65535
        result.append(pixels)
    return np.stack(result)


def memory():
    import psutil

    result = {
        "rss_mib": psutil.Process().memory_info().rss / 2**20,
        "peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(("RssAnon:", "RssFile:")):
                key, size, _ = line.split()
                result[key.rstrip(":") + "_mib"] = int(size) / 1024
    return result


def build_cache(args):
    before = memory()
    start = perf_counter()
    with rasterio.Env(GDAL_CACHEMAX=CACHE), rasterio.open(args.file) as source:
        cache = np.lib.format.open_memmap(
            args.cache,
            mode="w+",
            dtype="float32",
            shape=(source.count, source.height, source.width),
        )
        for top in range(0, source.height, BLOCK):
            for left in range(0, source.width, BLOCK):
                height = min(BLOCK, source.height - top)
                width = min(BLOCK, source.width - left)
                cache[:, top : top + height, left : left + width] = source.read(
                    window=Window(left, top, width, height), out_dtype="float32"
                )
        cache.flush()
        del cache
        with open(args.cache, "rb") as file:
            os.fsync(file.fileno())
        metadata = {
            "crs": str(source.crs),
            "transform": list(source.transform)[:6],
            "shape": [source.count, source.height, source.width],
            "nodata": source.nodata,
            "dtype": "float32",
            "band_order": list(source.indexes),
        }
        Path(args.cache + ".json").write_text(json.dumps(metadata))
    elapsed = perf_counter() - start
    after = memory()
    # Verify stored pixels across every region, outside cache-creation timing.
    cache = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    for top in range(0, cache.shape[1], BLOCK):
        for left in range(0, cache.shape[2], BLOCK):
            actual = cache[:, top : top + BLOCK, left : left + BLOCK]
            np.testing.assert_array_equal(
                actual, values(top, left, actual.shape[1], actual.shape[2])
            )
    return {
        "method": "cache_create",
        "seconds": elapsed,
        "bytes": Path(args.cache).stat().st_size,
        "before": before,
        "after": after,
    }


def worker(args):
    import dask
    from tiler import Tiler
    from geosave.geodata import Chips, io

    before = memory()
    start = perf_counter()
    with ExitStack() as opened:
        opened.enter_context(rasterio.Env(GDAL_CACHEMAX=CACHE))
        opened.enter_context(dask.config.set(scheduler="threads", num_workers=4))
        if args.method in ("chips", "blocks"):
            source = opened.enter_context(
                io.open_raster(args.file, chunks={"y": 512, "x": 512})
            )
            height, width = source.sizes["y"], source.sizes["x"]
            if args.method == "chips":
                chips = Chips(
                    [source],
                    size=256,
                    overlap=args.overlap,
                    mode="reflect",
                    dtype="float32",
                )
        elif args.method == "rasterio":
            source = opened.enter_context(rasterio.open(args.file))
            height, width = source.height, source.width
        else:
            source = np.load(args.cache, mmap_mode="r", allow_pickle=False)
            opened.callback(source._mmap.close)
            height, width = source.shape[-2:]
        tiler = Tiler(
            (3, height, width),
            (3, 256, 256),
            channel_dimension=0,
            overlap=args.overlap,
            mode="reflect",
        )
        indices = range(len(tiler))
        if args.workload == "sparse":
            rng = np.random.default_rng(42)
            indices = sorted(
                rng.choice(len(tiler), size=min(16, len(tiler)), replace=False)
            )
        groups = {}
        windows = {}
        for index in indices:
            low, high = tiler.get_tile_bbox(int(index))
            top, left = map(int, low)
            bottom = min(int(high[0]), height)
            right = min(int(high[1]), width)
            windows[int(index)] = (top, left, bottom, right)
            groups.setdefault((top // BLOCK, left // BLOCK), []).append(int(index))
        # Same spatially grouped order for every method. Output keeps global IDs.
        setup = perf_counter() - start
        read = 0.0
        first = None
        digest = hashlib.sha256()
        blocks = 0
        for indices in groups.values():
            block_time = 0.0
            if args.method == "blocks":
                top = min(windows[index][0] for index in indices)
                left = min(windows[index][1] for index in indices)
                bottom = max(windows[index][2] for index in indices)
                right = max(windows[index][3] for index in indices)
                started = perf_counter()
                pixels = source.isel(
                    y=slice(top, bottom), x=slice(left, right)
                ).gs.to_numpy(dtype="float32")
                block_time = perf_counter() - started
                read += block_time
                blocks += 1
            for index in indices:
                started = perf_counter()
                if args.method == "chips":
                    chip = chips[0, index]
                else:

                    def reader(_c, y, x, _nc, h, w):
                        if args.method == "blocks":
                            return pixels[
                                :, y - top : y - top + h, x - left : x - left + w
                            ]
                        if args.method == "rasterio":
                            return source.read(
                                window=Window(x, y, w, h), out_dtype="float32"
                            )
                        return source[:, y : y + h, x : x + w]

                    chip = tiler.get_tile(reader, index)
                duration = perf_counter() - started
                read += duration
                if first is None:
                    first = setup + block_time + duration
                # Verify nonconstant pixels, bands, nodata, overlap and boundary padding.
                y, x, bottom, right = windows[index]
                expected = values(y, x, bottom - y, right - x).astype("float32")
                expected = np.pad(
                    expected,
                    (
                        (0, 0),
                        (0, 256 - expected.shape[1]),
                        (0, 256 - expected.shape[2]),
                    ),
                    mode="reflect",
                )
                assert chip.dtype == np.dtype("float32")
                np.testing.assert_array_equal(chip, expected)
                digest.update(str(index).encode())
                digest.update(chip.tobytes())
                del chip, expected
            if args.method == "blocks":
                del pixels
        after = memory()
        started = perf_counter()
    close = perf_counter() - started
    return {
        "method": args.method,
        "workload": args.workload,
        "overlap": args.overlap,
        "shape": [3, height, width],
        "chips": len(windows),
        "blocks": blocks,
        "setup_s": setup,
        "read_s": read,
        "close_s": close,
        "total_s": setup + read + close,
        "first_chip_s": first,
        "checksum": digest.hexdigest(),
        "before": before,
        "after": after,
    }


def main(args):
    rows = []
    with TemporaryDirectory(
        prefix=".chip-reader-study-", dir=args.output.parent
    ) as temporary:
        folder = Path(temporary)
        path = folder / "source.tif"
        cache = folder / "prepared.npy"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=args.size + 13,
            height=args.size,
            count=3,
            dtype="uint16",
            crs="EPSG:32633",
            transform=Affine(10, 0, 500000, 0, -10, 5000000),
            nodata=65535,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
        ) as file:
            for _, window in file.block_windows(1):
                file.write(
                    values(
                        int(window.row_off),
                        int(window.col_off),
                        int(window.height),
                        int(window.width),
                    ),
                    window=window,
                )
        env = {**os.environ, "MPLCONFIGDIR": "/tmp/geosave-mpl"}
        for repeat in range(args.repeats):
            for workload in ("sparse", "full"):
                for overlap in (0, 64):
                    order = list(METHODS[repeat % 4 :]) + list(METHODS[: repeat % 4])
                    # Cache creation is measured in a separate process before use.
                    if workload == "sparse" and overlap == 0:
                        order = ["cache_create", *order]
                    for method in order:
                        output = subprocess.run(
                            [
                                sys.executable,
                                __file__,
                                "--worker",
                                "--file",
                                str(path),
                                "--cache",
                                str(cache),
                                "--method",
                                method,
                                "--workload",
                                workload,
                                "--overlap",
                                str(overlap),
                            ],
                            env=env,
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=120,
                        )
                        row = json.loads(output.stdout)
                        row["repeat"] = repeat
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        args.output.write_text(json.dumps({"results": rows}, indent=2))
        source_bytes = path.stat().st_size
        cache_bytes = cache.stat().st_size
    # Equal digest for all methods/repeats within each workload verifies ordering.
    for workload in ("sparse", "full"):
        for overlap in (0, 64):
            checksums = set()
            for row in rows:
                if row.get("workload") == workload and row.get("overlap") == overlap:
                    checksums.add(row["checksum"])
            assert len(checksums) == 1
    packages = {}
    for name in ("numpy", "xarray", "dask", "rasterio", "tiler", "psutil"):
        packages[name] = version(name)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": packages,
        "size": args.size,
        "repeats": args.repeats,
        "source_bytes": source_bytes,
        "cache_bytes": cache_bytes,
        "gdal_cache_mib": CACHE / 2**20,
        "dask_chunks": [512, 512],
        "block_size": BLOCK,
        "workers": 4,
        "cache": "OS caches not cleared",
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=[*METHODS, "cache_create"])
    parser.add_argument("--file")
    parser.add_argument("--cache")
    parser.add_argument("--workload", choices=["sparse", "full"])
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results.json")
    )
    args = parser.parse_args()
    if args.worker:
        print(
            json.dumps(
                build_cache(args) if args.method == "cache_create" else worker(args)
            )
        )
    else:
        main(args)
