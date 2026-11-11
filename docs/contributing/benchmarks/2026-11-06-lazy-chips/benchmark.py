"""Isolated-process eager/lazy Chips speed and peak RSS comparison."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


def worker(args):
    import dask
    import numpy as np
    import psutil

    from geosave.geodata import Chips, io

    if args.method == "eager":
        from eager_baseline import Chips

    process = psutil.Process()
    with dask.config.set(scheduler="threads", num_workers=4):
        with io.open_raster(args.file, chunks={"x": 512, "y": 512}) as source:
            rss_before = process.memory_info().rss / 2**20
            peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            start = perf_counter()
            chips = Chips([source], size=256, dtype="float32")
            setup = perf_counter() - start
            rss_constructed = process.memory_info().rss / 2**20
            if args.workload == "sparse":
                indices = np.linspace(0, len(chips) - 1, 16, dtype=int)
            else:
                indices = range(len(chips))
            elapsed = 0
            first = None
            checksum = 0.0
            for index in indices:
                start = perf_counter()
                tile = chips[0, int(index)]
                duration = perf_counter() - start
                elapsed += duration
                if first is None:
                    first = duration
                assert tile.dtype == np.float32
                assert tile.shape == (3, 256, 256)
                for channel in range(3):
                    assert np.all(tile[channel] == channel + 1)
                checksum += float(tile.sum())
                del tile
            print(
                json.dumps(
                    {
                        "method": args.method,
                        "workload": args.workload,
                        "size": source.sizes["x"],
                        "tiles": len(indices),
                        "setup_s": setup,
                        "first_tile_s": first,
                        "read_s": elapsed,
                        "total_s": setup + elapsed,
                        "rss_before_mib": rss_before,
                        "rss_constructed_mib": rss_constructed,
                        "rss_after_mib": process.memory_info().rss / 2**20,
                        "peak_rss_mib": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        / 1024,
                        "peak_before_mib": peak_before,
                        "checksum": checksum,
                    }
                )
            )


def main(args):
    import numpy as np
    import rasterio
    from affine import Affine

    results = []
    with TemporaryDirectory(prefix="chips-perf-") as folder:
        for size in (2048, 4096):
            path = Path(folder) / f"{size}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=size,
                height=size,
                count=3,
                dtype="uint16",
                crs="EPSG:32633",
                transform=Affine(10, 0, 500000, 0, -10, 5000000),
                tiled=True,
                blockxsize=256,
                blockysize=256,
            ) as file:
                for _, window in file.block_windows(1):
                    for band in range(1, 4):
                        file.write(
                            np.full((256, 256), band, dtype="uint16"),
                            band,
                            window=window,
                        )
            for workload in ("sparse", "full"):
                for repeat in range(args.repeats):
                    methods = (
                        ("eager", "lazy") if repeat % 2 == 0 else ("lazy", "eager")
                    )
                    for method in methods:
                        command = [
                            sys.executable,
                            __file__,
                            "--worker",
                            "--file",
                            str(path),
                            "--method",
                            method,
                            "--workload",
                            workload,
                        ]
                        env = {**os.environ, "MPLCONFIGDIR": "/tmp/geosave-mpl"}
                        output = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=90,
                            env=env,
                        )
                        result = json.loads(output.stdout)
                        result["repeat"] = repeat
                        results.append(result)
                        print(json.dumps(result), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "repeats": args.repeats,
                "chunks": [512, 512],
                "chip_size": 256,
                "workers": 4,
                "cache": "OS caches not cleared; newly written local TIFF",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--method", choices=["eager", "lazy"])
    parser.add_argument("--workload", choices=["sparse", "full"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results.json")
    )
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
