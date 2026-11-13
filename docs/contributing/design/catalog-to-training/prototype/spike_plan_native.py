# ruff: noqa
# Throwaway spike (first iteration); see README.md.
"""Plan samples with native pandas/geopandas only, load each onto its chip GeoBox,
feed a torch DataLoader with workers, and materialize a prepared Zarr store."""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles

import geosave as gs

root = Path(sys.argv[1])
catalog = gs.open_vector(root / "catalog.parquet")


def clock(label, work):
    began = time.perf_counter()
    result = work()
    print(f"{label:<52} {time.perf_counter() - began:8.3f}s")
    return result


# 1. Chip grid over the labelled region: native odc-geo tiles -> GeoDataFrame.
labels = catalog[catalog["collection"] == "labels"]
region = labels.to_crs("EPSG:32633").geometry.union_all()
grid = GeoBox.from_bbox(region.bounds, crs="EPSG:32633", resolution=10)
tiles = GeoboxTiles(grid, (256, 256))
chips = []
for index in np.ndindex(tiles.shape.yx):
    box = tiles[index]
    if box.shape == (256, 256):
        chips.append({"chip": len(chips), "geometry": box.extent.geom})
chips = gpd.GeoDataFrame(chips, crs="EPSG:32633")
print(len(chips), "chips")


# 2. Which Items cover each chip: native spatial join.
items = catalog.to_crs("EPSG:32633")[["id", "collection", "datetime", "geometry"]]
pairs = clock(
    "sjoin chips x items (covered_by)",
    lambda: chips.sjoin(items, predicate="covered_by").drop(columns="index_right"),
)

# 3. Time series: 4 S2 dates per window, stepping 2; S1 paired to each S2 date
#    by nearest acquisition within 4 days (merge_asof).
s2 = pairs[pairs["collection"] == "s2"].sort_values("datetime")
s1 = pairs[pairs["collection"] == "s1"].sort_values("datetime")
paired = pd.merge_asof(
    s2[["chip", "id", "datetime"]],
    s1[["chip", "id", "datetime"]].rename(
        columns={"id": "s1", "datetime": "s1_datetime"}
    ),
    left_on="datetime",
    right_on="s1_datetime",
    by="chip",
    direction="nearest",
    tolerance=pd.Timedelta("4D"),
).rename(columns={"id": "s2"})
label_ids = pairs[pairs["collection"] == "labels"].groupby("chip")["id"].first()

window = 4
step = 2
samples = []
for chip, group in paired.sort_values("datetime").groupby("chip"):
    if chip not in label_ids:
        continue
    for first in range(0, len(group) - window + 1, step):
        frame = group.iloc[first : first + window]
        samples.append(
            {
                "chip": chip,
                "s2": list(frame["s2"]),
                "s1": list(frame["s1"].fillna("")),
                "labels": [label_ids[chip]],
                "start_datetime": frame["datetime"].min(),
                "end_datetime": frame["datetime"].max(),
            }
        )
samples = gpd.GeoDataFrame(
    pd.DataFrame(samples).merge(chips, on="chip"), crs="EPSG:32633"
)
print(samples.drop(columns="geometry").head(3).to_string())
print(len(samples), "samples")
samples.to_parquet(root / "samples.parquet")
print("sample table round trip:", len(gpd.read_parquet(root / "samples.parquet")))

# 4. Load one sample: every collection onto the chip's GeoBox -> aligned tree.
pystac_items = {}
for entry in catalog.gs.to_items():
    pystac_items[entry.id] = entry


def load_sample(row) -> xr.DataTree:
    minx, miny, maxx, maxy = row.geometry.bounds
    box = GeoBox.from_bbox((minx, miny, maxx, maxy), crs="EPSG:32633", resolution=10)
    nodes = {}
    for collection in ("s2", "s1", "labels"):
        selected = []
        for identity in row[collection]:
            if identity:
                selected.append(pystac_items[identity])
        nodes[collection] = odc.stac.load(
            selected, geobox=box, groupby="id", chunks=None
        )
    return xr.DataTree.from_dict(nodes)


tree = clock(
    "load_sample (3 collections on chip GeoBox)", lambda: load_sample(samples.iloc[0])
)
print(tree)

# 5. torch DataLoader with worker processes.
import torch
from torch.utils.data import DataLoader, Dataset


class Planned(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tree = load_sample(self.samples.iloc[index])
        out = {}
        for name, node in tree.children.items():
            out[name] = torch.from_numpy(
                node.to_dataset().to_array().values.astype("float32")
            )
        return out


if __name__ == "__main__":
    for workers in (0, 4):
        loader = DataLoader(
            Planned(samples.iloc[:64]),
            batch_size=8,
            num_workers=workers,
            multiprocessing_context="fork" if workers else None,
        )
        began = time.perf_counter()
        count = 0
        for batch in loader:
            count += batch["s2"].shape[0]
        spent = time.perf_counter() - began
        print(
            f"DataLoader workers={workers}: {count / spent:6.1f} samples/s",
            {k: tuple(v.shape) for k, v in batch.items()},
        )

    # 6. Prepared store: stack samples along `sample`, one Zarr chunk per sample.
    rows = samples.iloc[:64]
    began = time.perf_counter()
    loaded = []
    for _, row in rows.iterrows():
        loaded.append(load_sample(row))
    nodes = {}
    for name in ("s2", "s1", "labels"):
        arrays = []
        for tree in loaded:
            arrays.append(
                tree[name]
                .to_dataset()
                .drop_vars(["time", "spatial_ref", "x", "y"], errors="ignore")
                .to_array("band")
            )
        nodes[name] = xr.concat(arrays, "sample").to_dataset(name="pixels")
    store = xr.DataTree.from_dict(nodes)
    for node in store.subtree:
        if "pixels" in node.ds:
            node.ds["pixels"].encoding["chunks"] = (1, *node.ds["pixels"].shape[1:])
    store.to_zarr(root / "prepared.zarr", mode="w", consolidated=True)
    print(f"prepared 64 samples into zarr in {time.perf_counter() - began:.1f}s")

    class Prepared(Dataset):
        def __init__(self, path):
            self.path = path
            self.tree = None

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            if self.tree is None:
                self.tree = xr.open_datatree(self.path, engine="zarr", chunks=None)
            out = {}
            for name in ("s2", "s1", "labels"):
                out[name] = torch.from_numpy(
                    self.tree[name]["pixels"][index].values.astype("float32")
                )
            return out

    for workers in (0, 4):
        loader = DataLoader(
            Prepared(root / "prepared.zarr"),
            batch_size=8,
            num_workers=workers,
            multiprocessing_context="fork" if workers else None,
        )
        began = time.perf_counter()
        count = 0
        for batch in loader:
            count += batch["s2"].shape[0]
        spent = time.perf_counter() - began
        print(f"Prepared zarr workers={workers}: {count / spent:6.1f} samples/s")
