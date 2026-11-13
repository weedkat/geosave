# ruff: noqa
# Throwaway spike (first iteration); see README.md.
import sys, time
import numpy as np, xarray as xr, zarr

root = sys.argv[1]
cube = xr.open_datatree(f"{root}/cube.zarr", engine="zarr", chunks=None)
nodes = {}
for name in ("s2", "s1", "labels"):
    stacked = (
        cube[name].to_dataset().to_array("band").transpose(..., "band", "y", "x")
        if "time" in cube[name].dims
        else cube[name].to_dataset().to_array("band")
    )
    stacked = stacked.drop_vars([c for c in stacked.coords if c not in stacked.dims])
    data = stacked.to_dataset(name="pixels")
    data["pixels"].encoding = {"chunks": (*stacked.shape[:-2], 256, 256)}
    nodes[name] = data
xr.DataTree.from_dict(nodes).to_zarr(
    f"{root}/cube_stacked.zarr", mode="w", consolidated=True
)
arrays = {}
group = zarr.open_group(f"{root}/cube_stacked.zarr", mode="r")
for name in ("s2", "s1", "labels"):
    arrays[name] = group[name]["pixels"]
    print(name, arrays[name].shape, arrays[name].chunks)
rng = np.random.default_rng(0)
began = time.perf_counter()
for _ in range(200):
    y = int(rng.integers(0, 1024 - 256))
    x = int(rng.integers(0, 1024 - 256))
    for name in arrays:
        arrays[name][..., y : y + 256, x : x + 256]
print(
    f"random crops, stacked zarr, raw zarr API: {200 / (time.perf_counter() - began):6.1f} samples/s"
)
began = time.perf_counter()
for _ in range(200):
    y = int(rng.integers(0, 4)) * 256
    x = int(rng.integers(0, 4)) * 256
    for name in arrays:
        arrays[name][..., y : y + 256, x : x + 256]
print(f"chunk-aligned crops: {200 / (time.perf_counter() - began):6.1f} samples/s")
