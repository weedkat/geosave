# ruff: noqa
# Throwaway spike (first iteration); see README.md.
"""D4: prepare one aligned region cube (all collections on one block GeoBox) into Zarr,
then read random 256 crops; compare with litdata if available."""

import sys
import time
from pathlib import Path

import numpy as np
import odc.stac
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave as gs

root = Path(sys.argv[1])
catalog = gs.open_vector(root / "catalog.parquet")
items = {}
for entry in catalog.gs.to_items():
    items[entry.id] = entry


def clock(label, work):
    began = time.perf_counter()
    result = work()
    print(f"{label:<52} {time.perf_counter() - began:8.3f}s")
    return result


block = GeoBox.from_bbox(
    (504100, 4004100, 504100 + 10240, 4004100 + 10240), crs="EPSG:32633", resolution=10
)
s2_ids = ["S2_20240103", "S2_20240108", "S2_20240113", "S2_20240118"]
s1_ids = ["S1_20240103", "S1_20240109", "S1_20240115", "S1_20240121"]


def prepare():
    nodes = {}
    for name, ids in (("s2", s2_ids), ("s1", s1_ids), ("labels", ["landcover"])):
        selected = []
        for identity in ids:
            selected.append(items[identity])
        nodes[name] = odc.stac.load(
            selected, geobox=block, groupby="id", chunks={"x": 256, "y": 256}, pool=6
        )
    tree = xr.DataTree.from_dict(nodes)
    for node in tree.subtree:
        for variable in node.ds.data_vars.values():
            variable.encoding.pop("chunks", None)
    tree.to_zarr(root / "cube.zarr", mode="w", consolidated=True)
    return tree


clock(f"prepare one {block.shape.y}x{block.shape.x} block (3 collections)", prepare)
cube = xr.open_datatree(root / "cube.zarr", engine="zarr", chunks=None)
rng = np.random.default_rng(0)


def random_chips(count):
    for _ in range(count):
        y = int(rng.integers(0, block.shape.y - 256))
        x = int(rng.integers(0, block.shape.x - 256))
        window = {"y": slice(y, y + 256), "x": slice(x, x + 256)}
        for name in ("s2", "s1", "labels"):
            cube[name].to_dataset().isel(window).to_array().values


began = time.perf_counter()
random_chips(64)
print(
    f"random 256 crops from region cube: {64 / (time.perf_counter() - began):6.1f} samples/s (1 process)"
)
