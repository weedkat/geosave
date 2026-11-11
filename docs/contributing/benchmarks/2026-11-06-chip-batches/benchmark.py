"""Compare individual and batched Chips reads in isolated processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import rasterio
from affine import Affine

# Reuse the previous study's coordinate-derived pixel oracle and RSS sampler.
study = runpy.run_path(
    str(Path(__file__).parents[1] / "2026-11-06-chip-readers/benchmark.py")
)
values = study["values"]
memory = study["memory"]


def worker(args):
    import dask
    from geosave.geodata import Chips, io

    before = memory()
    start = perf_counter()
    with (
        rasterio.Env(GDAL_CACHEMAX=32 * 2**20),
        dask.config.set(scheduler="threads", num_workers=4),
    ):
        with io.open_raster(args.file, chunks={"y": 512, "x": 512}) as source:
            chips = Chips(
                [source], size=256, overlap=64, mode="reflect", dtype="float32"
            )
            indices = np.arange(len(chips))
            if args.order == "shuffled":
                np.random.default_rng(42).shuffle(indices)
            setup = perf_counter() - start
            elapsed = 0.0
            digest = hashlib.sha256()
            for offset in range(0, len(indices), 16):
                batch = indices[offset : offset + 16].tolist()
                start = perf_counter()
                if args.method == "batch":
                    pixels = chips.__getitems__(batch)
                else:
                    pixels = [chips[index] for index in batch]
                elapsed += perf_counter() - start
                for index, chip in zip(batch, pixels, strict=True):
                    low, high = chips._tilers[0].get_tile_bbox(index)
                    top, left = map(int, low)
                    bottom = min(int(high[0]), source.sizes["y"])
                    right = min(int(high[1]), source.sizes["x"])
                    expected = values(top, left, bottom - top, right - left).astype(
                        "float32"
                    )
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
                    digest.update(chip.tobytes())
                del pixels, chip, expected
            after = memory()
    return {
        "method": args.method,
        "order": args.order,
        "chips": len(chips),
        "setup_s": setup,
        "read_s": elapsed,
        "total_s": setup + elapsed,
        "before": before,
        "after": after,
        "checksum": digest.hexdigest(),
    }


def main(args):
    rows = []
    with TemporaryDirectory(prefix=".chip-batches-", dir=args.output.parent) as folder:
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
            for order in ("sequential", "shuffled"):
                methods = (
                    ("individual", "batch")
                    if repeat % 2 == 0
                    else ("batch", "individual")
                )
                for method in methods:
                    output = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            "--file",
                            str(path),
                            "--method",
                            method,
                            "--order",
                            order,
                        ],
                        env={**os.environ, "MPLCONFIGDIR": "/tmp/geosave-mpl"},
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=120,
                    )
                    row = json.loads(output.stdout)
                    row["repeat"] = repeat
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    for order in ("sequential", "shuffled"):
        assert len({row["checksum"] for row in rows if row["order"] == order}) == 1
    args.output.write_text(
        json.dumps(
            {
                "size": args.size,
                "repeats": args.repeats,
                "batch_size": 16,
                "chip_size": 256,
                "overlap": 64,
                "max_block_bytes": 32 * 2**20,
                "dask_chunks": [512, 512],
                "threads": 4,
                "gdal_cache_mib": 32,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--method")
    parser.add_argument("--order")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results.json")
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
    else:
        main(args)
