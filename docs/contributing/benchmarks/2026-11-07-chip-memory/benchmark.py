"""Compare historical mapping policies with current production behavior."""

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import psutil
import rasterio
from affine import Affine

previous = runpy.run_path(
    str(Path(__file__).parents[1] / "2026-11-06-chip-readers/benchmark.py")
)
values = previous["values"]
memory = previous["memory"]


def snapshot():
    result = memory()
    processes = [psutil.Process(), *psutil.Process().children(recursive=True)]
    result["processes"] = []
    for process in processes:
        try:
            info = process.memory_full_info()
            result["processes"].append(
                {
                    "pid": process.pid,
                    "rss_mib": info.rss / 2**20,
                    "pss_mib": info.pss / 2**20,
                    "uss_mib": info.uss / 2**20,
                }
            )
        except psutil.NoSuchProcess:
            pass
    result["tree_pss_mib"] = sum(row["pss_mib"] for row in result["processes"])
    return result


class ReleasingMap:
    """Prototype writer shim; retain the production writer's actual block plan."""

    def __init__(self, array, path, limit):
        self.array = array
        self.path = path
        self.limit = limit
        self.written = 0
        self.base = array.base

    def __setitem__(self, key, value):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r+", allow_pickle=False)
            self.base = self.array.base
        self.array[key] = value
        self.written += value.nbytes
        if self.written >= self.limit:
            self.flush()
            self.base.close()
            self.array = None
            self.written = 0

    def flush(self):
        if self.array is not None:
            self.array.flush()


def retain_single(reader, key):
    return reader._read_tile(*reader._key(key))


def retain_batch(reader, indices):
    keys = [reader.key(index) for index in indices]
    return [retain_single(reader, key) for key in keys]


def install_reader_retain():
    from geosave.geodata.io.chips import ChipStore

    ChipStore.__getitem__ = retain_single
    ChipStore.__getitems__ = retain_batch


def init_retain_worker(_index):
    install_reader_retain()


def release_batch(reader, indices):
    """Use real indexing and retain copies, then release their source maps."""
    try:
        keys = [reader.key(index) for index in indices]
        return [reader[key] for key in keys]
    finally:
        reader._close_maps()


def install_reader_release():
    from geosave.geodata.io.chips import ChipStore

    ChipStore.__getitem__ = retain_single
    ChipStore.__getitems__ = release_batch


def init_worker(_index):
    # Spawned workers import this module fresh; install only the reader policy.
    install_reader_release()


def worker(args):
    import dask.config
    from torch.utils.data import DataLoader
    from geosave.geodata import Chips, io

    points = {"before": snapshot()}
    with ExitStack() as stack:
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=32 * 2**20))
        stack.enter_context(dask.config.set(scheduler="threads", num_workers=4))
        folder = Path(
            stack.enter_context(TemporaryDirectory(dir=Path(args.file).parent))
        )
        source = stack.enter_context(
            io.open_raster(args.file, chunks={"y": 512, "x": 512})
        )
        prepared = (source.astype("float32") * np.float32(0.5) + np.float32(3)).where(
            source != 65535, np.float32(-9999)
        )
        chips = Chips([prepared], size=256, overlap=64, mode="reflect")
        if args.policy in ("retain", "release"):
            from baseline import _write

            stack.enter_context(patch("geosave.geodata.io.chips._write", _write))
            install_reader_retain()
        if args.policy == "release":
            install_reader_release()
            original = np.lib.format.open_memmap

            def open_map(path, *positional, **kwargs):
                array = original(path, *positional, **kwargs)
                if kwargs.get("mode") != "w+":
                    return array
                # Source writes are already bounded windows. Final-chip writes
                # release after the same bounded batch as production computes.
                tile_bytes = 3 * 256 * 256 * 4
                limit = (
                    1
                    if args.layout == "cache"
                    else (32 * 2**20 // tile_bytes) * tile_bytes
                )
                return ReleasingMap(array, path, limit)

            stack.enter_context(patch("numpy.lib.format.open_memmap", open_map))
        start = perf_counter()
        if args.layout == "cache":
            reader = stack.enter_context(chips.cache(directory=folder, workers=4))
        else:
            reader = stack.enter_context(
                Chips.open(chips.save(folder / "saved", workers=4))
            )
        preparation = perf_counter() - start
        points["prepared"] = snapshot()
        source.close()
        indices = np.arange(len(reader))
        np.random.default_rng(42).shuffle(indices)
        options = {}
        if args.loader_workers:
            options = {
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "timeout": 60,
            }
            if args.policy == "release":
                options["worker_init_fn"] = init_worker
            elif args.policy == "retain":
                options["worker_init_fn"] = init_retain_worker
        loader = DataLoader(
            reader,
            batch_size=16,
            sampler=indices.tolist(),
            num_workers=args.loader_workers,
            **options,
        )
        times = []
        digest = hashlib.sha256()
        for epoch in range(args.epochs):
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
                del batch, chip, raw, expected
            assert offset == len(reader)
            times.append(seconds)
            points[f"epoch_{epoch + 1}"] = snapshot()
        # Persistent workers must stop before closing/removing their files.
        del iterator, loader
        reader.close()
        points["closed"] = snapshot()
    del reader, chips, prepared, source
    points["scope_exited"] = snapshot()
    gc.collect()  # Diagnostic only; never added to the hot read/write path.
    points["collected"] = snapshot()
    return {
        "policy": args.policy,
        "layout": args.layout,
        "loader_workers": args.loader_workers,
        "prepare_s": preparation,
        "epochs_s": times,
        "points": points,
        "checksum": digest.hexdigest(),
    }


def main(args):
    rows = []
    for size in args.sizes:
        with TemporaryDirectory(
            prefix=".chip-memory-", dir=args.output.parent
        ) as folder:
            path = Path(folder) / "source.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=size + 13,
                height=size,
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
            for count in args.worker_counts:
                for repeat in range(args.repeats):
                    policies = list(args.policies)
                    if repeat % 2:
                        policies.reverse()
                    for layout in args.layouts:
                        for policy in policies:
                            result = subprocess.run(
                                [
                                    sys.executable,
                                    __file__,
                                    "--worker",
                                    "--file",
                                    str(path),
                                    "--policy",
                                    policy,
                                    "--layout",
                                    layout,
                                    "--loader-workers",
                                    str(count),
                                    "--epochs",
                                    str(args.epochs),
                                ],
                                capture_output=True,
                                text=True,
                                timeout=180,
                                env={
                                    **os.environ,
                                    "OMP_NUM_THREADS": "1",
                                    "MKL_NUM_THREADS": "1",
                                    "MPLCONFIGDIR": "/tmp/geosave-mpl",
                                },
                            )
                            if result.returncode:
                                raise RuntimeError(result.stderr)
                            row = json.loads(result.stdout)
                            row.update(size=size, repeat=repeat)
                            rows.append(row)
                            print(
                                json.dumps(
                                    {
                                        "size": size,
                                        "workers": count,
                                        "repeat": repeat,
                                        "layout": layout,
                                        "policy": policy,
                                        "prepare_s": row["prepare_s"],
                                        "epochs_s": row["epochs_s"],
                                    }
                                ),
                                flush=True,
                            )
                            args.output.write_text(
                                json.dumps(
                                    {
                                        "sizes": args.sizes,
                                        "epochs": args.epochs,
                                        "repeats": args.repeats,
                                        "results": rows,
                                    },
                                    indent=2,
                                )
                                + "\n"
                            )
    for size in args.sizes:
        assert len({row["checksum"] for row in rows if row["size"] == size}) == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--policy", choices=["retain", "release", "production"])
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["retain", "release", "production"],
        default=["retain", "release"],
    )
    parser.add_argument(
        "--layouts", nargs="+", choices=["cache", "saved"], default=["cache", "saved"]
    )
    parser.add_argument("--layout", choices=["cache", "saved"])
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
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
