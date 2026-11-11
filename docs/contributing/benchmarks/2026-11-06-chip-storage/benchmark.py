"""Compare lazy, cached-source and persistent-chip training reads."""

import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import rasterio
from affine import Affine

study = runpy.run_path(
    str(Path(__file__).parents[1] / "2026-11-06-chip-readers/benchmark.py")
)
values = study["values"]
memory = study["memory"]


def worker(args):
    import dask
    from torch.utils.data import DataLoader
    from geosave.geodata import Chips, io

    before = memory()
    with ExitStack() as stack:
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=32 * 2**20))
        stack.enter_context(
            dask.config.set(scheduler="threads", num_workers=args.threads)
        )
        folder = Path(
            stack.enter_context(
                TemporaryDirectory(prefix="store-", dir=Path(args.file).parent)
            )
        )
        start = perf_counter()
        source = stack.enter_context(
            io.open_raster(args.file, chunks={"y": 512, "x": 512})
        )
        prepared = (source.astype("float32") * np.float32(0.5) + np.float32(3)).where(
            source != 65535, np.float32(-9999)
        )
        chips = Chips([prepared], size=256, overlap=64, mode="reflect")
        setup = perf_counter() - start
        start = perf_counter()
        if args.method == "cache":
            reader = stack.enter_context(
                chips.cache(directory=folder, workers=args.threads)
            )
        elif args.method == "saved":
            path = chips.save(folder / "saved", workers=args.threads)
            reader = stack.enter_context(Chips.open(path))
        else:
            reader = chips
        preparation = perf_counter() - start
        after_prepare = memory()
        disk_bytes = sum(
            path.stat().st_size for path in folder.rglob("*") if path.is_file()
        )
        if args.method != "lazy":
            source.close()
        elapsed = []
        digest = hashlib.sha256()
        for epoch in range(args.epochs):
            indices = np.arange(len(reader))
            np.random.default_rng(42 + epoch).shuffle(indices)
            loader = DataLoader(
                reader,
                batch_size=16,
                sampler=indices.tolist(),
                num_workers=args.loader_workers,
            )
            start = perf_counter()
            iterator = iter(loader)
            seconds = perf_counter() - start
            offset = 0
            while True:
                start = perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    seconds += perf_counter() - start
                    break
                seconds += perf_counter() - start
                for index, chip in zip(
                    indices[offset : offset + len(batch)], batch.numpy(), strict=True
                ):
                    low, high = chips._tilers[0].get_tile_bbox(int(index))
                    top, left = map(int, low)
                    bottom = min(int(high[0]), source.sizes["y"])
                    right = min(int(high[1]), source.sizes["x"])
                    raw = values(top, left, bottom - top, right - left)
                    expected = raw.astype("float32") * np.float32(0.5) + np.float32(3)
                    expected[raw == 65535] = -9999
                    expected = np.pad(
                        expected,
                        (
                            (0, 0),
                            (0, 256 - expected.shape[1]),
                            (0, 256 - expected.shape[2]),
                        ),
                        mode="reflect",
                    )
                    np.testing.assert_array_equal(chip, expected)
                    assert chip.dtype == np.float32
                    digest.update(chip.tobytes())
                offset += len(batch)
                del batch, chip, expected, raw
            assert offset == len(reader)
            elapsed.append(seconds)
        after = memory()
    return {
        "method": args.method,
        "threads": args.threads,
        "loader_workers": args.loader_workers,
        "chips": len(chips),
        "setup_s": setup,
        "prepare_s": preparation,
        "epochs_s": elapsed,
        "total_s": setup + preparation + sum(elapsed),
        "disk_mib": disk_bytes / 2**20,
        "before": before,
        "after_prepare": after_prepare,
        "after": after,
        "checksum": digest.hexdigest(),
    }


def main(args):
    rows = []
    with TemporaryDirectory(prefix=".chip-storage-", dir=args.output.parent) as folder:
        path = Path(folder) / "source.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=args.size + 13,
            height=args.size,
            count=3,
            dtype="uint16",
            crs="EPSG:32633",
            nodata=65535,
            transform=Affine(10, 0, 500000, 0, -10, 5000000),
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
        for repeat in range(args.repeats):
            for threads in args.thread_counts:
                methods = (
                    ["lazy", "cache", "saved"] if threads == 4 else ["cache", "saved"]
                )
                if repeat % 2:
                    methods.reverse()
                for method in methods:
                    result = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            "--file",
                            str(path),
                            "--method",
                            method,
                            "--threads",
                            str(threads),
                            "--epochs",
                            str(args.epochs),
                            "--loader-workers",
                            str(args.loader_workers),
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=240,
                        env={
                            **os.environ,
                            "MPLCONFIGDIR": "/tmp/geosave-mpl",
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                        },
                    )
                    row = json.loads(result.stdout)
                    row["repeat"] = repeat
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(
                        json.dumps(
                            {
                                "size": args.size,
                                "epochs": args.epochs,
                                "repeats": args.repeats,
                                "thread_counts": args.thread_counts,
                                "loader_workers": args.loader_workers,
                                "python": platform.python_version(),
                                "platform": platform.platform(),
                                "versions": {
                                    name: version(name)
                                    for name in [
                                        "numpy",
                                        "dask",
                                        "xarray",
                                        "rasterio",
                                        "torch",
                                    ]
                                },
                                "results": rows,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
    assert len({row["checksum"] for row in rows}) == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--method", choices=["lazy", "cache", "saved"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[4])
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results.json")
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
    else:
        main(args)
