# ruff: noqa
# Throwaway spike (first iteration); see README.md.
import sys, time
import numpy as np, xarray as xr
from litdata import optimize, StreamingDataset, StreamingDataLoader

root = sys.argv[1]
store = xr.open_datatree(f"{root}/prepared.zarr", engine="zarr", chunks=None)
arrays = {name: store[name]["pixels"].values for name in ("s2", "s1", "labels")}


def sample(index):
    return {name: arrays[name][index] for name in arrays} | {"index": index}


if __name__ == "__main__":
    began = time.perf_counter()
    optimize(
        fn=sample,
        inputs=list(range(arrays["s2"].shape[0])),
        output_dir=f"{root}/lit",
        chunk_bytes="64MB",
        num_workers=1,
        mode="overwrite",
    )
    print(f"optimize {time.perf_counter() - began:.1f}s")
    for workers in (0, 4):
        ds = StreamingDataset(f"{root}/lit")
        loader = StreamingDataLoader(ds, batch_size=8, num_workers=workers)
        began = time.perf_counter()
        n = 0
        for batch in loader:
            n += batch["s2"].shape[0]
        print(
            f"litdata workers={workers}: {n / (time.perf_counter() - began):6.1f} samples/s",
            tuple(batch["s2"].shape),
        )
