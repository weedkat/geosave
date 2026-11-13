# ruff: noqa
# Throwaway prototype run; see README.md.
"""Exercise the virtual cube: laziness, pairing, chips, pickling, workers, materialization."""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import odc.stac
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave as gs
from cube import open_cube, open_tree

root = Path(sys.argv[1])
catalog = gs.open_vector(root / "catalog.parquet")


def clock(label, work):
    began = time.perf_counter()
    result = work()
    print(f"{label:<56} {time.perf_counter() - began:8.3f}s")
    return result


grid = GeoBox.from_bbox(
    (500000, 4000000, 520480, 4020480), crs="EPSG:32633", resolution=10
)
tree = clock(
    "open_tree (3 collections, whole 2048² grid)", lambda: open_tree(catalog, grid)
)
print(tree)
s2 = tree["s2"].to_dataset()
s1 = tree["s1"].to_dataset()
labels = tree["labels"].to_dataset()
print("dask?", s2["B04"].chunks, "| pickle bytes:", len(pickle.dumps(tree)))

# Pair S1 to S2 time steps: native xarray, still lazy.
paired = clock(
    "s1.sel(time=s2.time, nearest, 4D)",
    lambda: s1.sel(time=s2.time, method="nearest", tolerance=pd.Timedelta("4D")),
)
print("paired s1 times:", paired.time.values[:4])
print("lazy after pairing:", type(paired["VV"].variable._data).__name__)

# A training window: 4 consecutive S2 steps, one 256² chip.
window = {"time": slice(2, 6), "y": slice(512, 768), "x": slice(1024, 1280)}
chip_s2 = s2.isel(window)
chip_s1 = paired.isel(window)
chip_lb = labels.isel(y=window["y"], x=window["x"])
print("lazy chip:", type(chip_s2["B04"].variable._data).__name__, dict(chip_s2.sizes))
values = clock(
    "read chip: s2 (5 bands) + s1 (2) + labels",
    lambda: (chip_s2.load(), chip_s1.load(), chip_lb.load()),
)

# Same pixels as a direct ODC load onto the chip grid?
box = grid[512:768, 1024:1280]
s2_items = (
    catalog[catalog["collection"] == "s2"]
    .sort_values("datetime")
    .iloc[2:6]
    .gs.to_items()
)
direct = odc.stac.load(s2_items, geobox=box, bands=["B04"], groupby="id", chunks=None)
print(
    "equals direct ODC load:",
    np.array_equal(values[0]["B04"].values, direct["B04"].values),
)

# geosave accessors on the lazy cube.
print("gs.geobox == chip box:", chip_s2.gs.geobox == box)
print(
    "crop_to_grid lazy:", type(s2.gs.crop_to_grid(box)["B04"].variable._data).__name__
)
from geosave.geodata.core.raster.chips import Chips

chips = Chips([s2.isel(time=slice(0, 4))], 256, mode="drop")
clock(
    f"Chips over the cube: {len(chips)} chips, read chips[7]", lambda: chips[7].load()
)

# Composite: monthly median via native resample, read only one chip's window.
monthly = clock(
    "s2.isel(chip).resample(time='MS').median()",
    lambda: s2.isel(y=window["y"], x=window["x"]).resample(time="MS").median().load(),
)
print("monthly composite:", dict(monthly.sizes))

# Materialize: dask-chunk the lazy cube and write Zarr; one task per chunk.
small = s2.isel(time=slice(0, 3), y=slice(0, 1024), x=slice(0, 1024)).chunk(
    {"time": 1, "y": 512, "x": 512}
)
print("dask tasks for 3×1024² × 5 bands:", len(dict(small.__dask_graph__())))
clock("materialize to Zarr", lambda: small.to_zarr(root / "virtual.zarr", mode="w"))

# DataLoader workers: the tree pickles to each worker.
import torch
from torch.utils.data import DataLoader, Dataset


class CubeChips(Dataset):
    def __init__(self, s2, s1, labels, count):
        self.s2, self.s1, self.labels, self.count = s2, s1, labels, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        y = int(rng.integers(0, 1700))
        x = int(rng.integers(0, 1700))
        t = int(rng.integers(0, 8))
        where = {"y": slice(y, y + 256), "x": slice(x, x + 256)}
        s2 = self.s2.isel({**where, "time": slice(t, t + 4)}).to_array().values
        s1 = self.s1.isel({**where, "time": slice(t, t + 4)}).to_array().values
        lb = self.labels.isel(where).to_array().values
        return (
            torch.from_numpy(s2.astype("float32")),
            torch.from_numpy(s1),
            torch.from_numpy(lb.astype("int64")),
        )


if __name__ == "__main__":
    print("ODC reads for one chip read:", s2["B02"].variable._data.array.reader.reads)
    for workers in (0, 4):
        loader = DataLoader(
            CubeChips(s2, paired, labels, 48),
            batch_size=8,
            num_workers=workers,
            multiprocessing_context="fork" if workers else None,
        )
        began = time.perf_counter()
        count = 0
        for a, b, c in loader:
            count += a.shape[0]
        print(
            f"DataLoader over virtual cube, workers={workers}: {count / (time.perf_counter() - began):5.1f} samples/s",
            tuple(a.shape),
        )
