# ruff: noqa
# Throwaway spike (first iteration); see README.md.
"""Compare tree-first and plan-first loading of one multi-sensor time-series chip."""

import sys
import time
from pathlib import Path

import dask
import numpy as np
import odc.stac
from odc.geo.geobox import GeoBox

import geosave as gs
from geosave.geodata.stac import load_dataset

root = Path(sys.argv[1])
catalog = gs.open_vector(root / "catalog.parquet")
print(type(catalog).__name__, len(catalog), catalog.crs)

s2_rows = catalog[catalog["collection"] == "s2"].sort_values("datetime").iloc[:6]
s1_rows = catalog[catalog["collection"] == "s1"].sort_values("datetime").iloc[:6]
label_rows = catalog[catalog["collection"] == "labels"]


def clock(label, work):
    began = time.perf_counter()
    result = work()
    print(f"{label:<48} {time.perf_counter() - began:8.3f}s")
    return result


# ---------------- A: tree-first ----------------
rows = gs.GeoDataFrame if False else None
selected = catalog[
    catalog.index.isin([*s2_rows.index, *s1_rows.index, *label_rows.index])
]
tree = clock("A1 to_datatree (lazy, native grids)", lambda: selected.gs.to_datatree())
print("   nodes:", len(tree.gs.datasets))
target = tree["labels/landcover/class"].to_dataset().gs.geobox
print("   target grid", target.shape, target.crs)
aligned = clock("A2 tree.gs.reproject(label grid)", lambda: tree.gs.reproject(target))
s2 = clock("A3 combine_stac s2", lambda: aligned["s2"].gs.combine_stac())
s1 = clock("A3 combine_stac s1", lambda: aligned["s1"].gs.combine_stac())
labels = aligned["labels/landcover/class"].to_dataset()
print(s2)
graph = len(dict(s2.__dask_graph__())) + len(dict(s1.__dask_graph__()))
print("   dask tasks s2+s1:", graph)
chips = clock(
    "A4 Chips([s2, s1, labels], 256)",
    lambda: (
        gs.Chips([s2, s1, labels], 256, mode="drop") if hasattr(gs, "Chips") else None
    ),
)
from geosave.geodata.core.raster.chips import Chips

chips_s2 = Chips([s2], 256, mode="drop")
geobox = chips_s2.geobox((0, 5))


def read_tree_chip():
    with dask.config.set(scheduler="synchronous"):
        a = s2.gs.crop_to_grid(geobox).compute()
        b = s1.gs.crop_to_grid(geobox).compute()
        c = labels.gs.crop_to_grid(geobox).compute()
    return a, b, c


clock("A5 compute one chip (sync scheduler)", read_tree_chip)
clock("A5 compute one chip again", read_tree_chip)

# ---------------- B: plan-first, per-chip ODC load ----------------
s2_items = s2_rows.gs.to_items()
s1_items = s1_rows.gs.to_items()
label_items = label_rows.gs.to_items()


def read_planned_chip(box: GeoBox):
    a = odc.stac.load(
        s2_items,
        geobox=box,
        groupby="id",
        chunks=None,
        bands=["B02", "B03", "B04", "B08", "SCL"],
    )
    b = odc.stac.load(
        s1_items, geobox=box, groupby="id", chunks=None, resampling="bilinear"
    )
    c = odc.stac.load(label_items, geobox=box, groupby="id", chunks=None)
    return a, b, c


got = clock(
    "B1 odc.stac.load 3 collections onto chip", lambda: read_planned_chip(geobox)
)
clock("B1 again (OS cache warm)", lambda: read_planned_chip(geobox))
print(got[0].sizes, got[1].sizes, got[2].sizes)
clock("B2 to_items for 13 rows", lambda: (s2_rows.gs.to_items(), s1_rows.gs.to_items()))
clock(
    "B3 geosave load_dataset s2 onto chip",
    lambda: load_dataset(s2_items, geobox, groupby="id", chunks=None),
)

# same answer?
a, b, c = read_tree_chip()
print("A vs B s2 equal:", np.array_equal(a["B04"].values, got[0]["B04"].values))
print(
    "A vs B s1 close:", np.allclose(b["VV"].values, got[1]["VV"].values, equal_nan=True)
)
print("A vs B label equal:", np.array_equal(c["class"].values, got[2]["class"].values))

print(
    "A vs B label equal (squeezed):",
    np.array_equal(c["class"].values.squeeze(), got[2]["class"].values.squeeze()),
)
near = odc.stac.load(s1_items, geobox=geobox, groupby="id", chunks=None)
print(
    "A vs B s1 equal (both nearest):",
    np.allclose(b["VV"].values, near["VV"].values, equal_nan=True),
)

# ---------------- A': tree-first with chip-sized dask chunks ----------------
small = selected.gs.to_datatree(chunks={"x": 256, "y": 256})
aligned_small = small.gs.reproject(target)
s2s = aligned_small["s2"].gs.combine_stac()
s1s = aligned_small["s1"].gs.combine_stac()
lbs = aligned_small["labels/landcover/class"].to_dataset()
print(
    "   dask tasks s2+s1 (256 chunks):",
    len(dict(s2s.__dask_graph__())) + len(dict(s1s.__dask_graph__())),
)


def read_small():
    with dask.config.set(scheduler="synchronous"):
        return (
            s2s.gs.crop_to_grid(geobox).compute(),
            s1s.gs.crop_to_grid(geobox).compute(),
            lbs.gs.crop_to_grid(geobox).compute(),
        )


clock("A'5 compute one chip, 256 chunks", read_small)
clock("A'5 again", read_small)
same = np.isclose(b["VV"].values, near["VV"].values, equal_nan=True)
print("A vs B s1 nearest: fraction of equal pixels", same.mean())
