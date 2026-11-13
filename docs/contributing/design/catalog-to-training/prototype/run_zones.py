# ruff: noqa
# Throwaway prototype run; see README.md.
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from odc.geo.geobox import GeoBox
import geosave as gs
from cube import open_tree
from plan import tile, read_geobox

if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    catalog = gs.open_vector(root / "catalog.parquet")
    g33 = GeoBox.from_bbox(
        (505000, 4005000, 507560, 4007560), crs="EPSG:32633", resolution=10
    )
    g34 = GeoBox.from_geopolygon(
        g33.extent.to_crs("EPSG:32634"), resolution=10, crs="EPSG:32634"
    )[0:256, 0:256]
    plan = pd.concat([tile(g33, 256), tile(g34, 256)], ignore_index=True)
    print(plan[["chip", "proj:code", "proj:shape"]].to_string())
    for _, row in plan.iterrows():
        tree = open_tree(catalog[catalog["collection"] != "labels"], read_geobox(row))
        s2 = tree["s2"].to_dataset().isel(time=0).compute()
        print(
            row["proj:code"],
            "s2 B04 chip",
            s2["B04"].shape,
            "nonzero",
            float((s2["B04"] > 0).mean()),
        )
    # pickling cost of a large catalog for worker processes
    big = pd.concat([catalog] * 8000, ignore_index=True)
    began = time.perf_counter()
    blob = pickle.dumps(big)
    spent = time.perf_counter() - began
    print(f"pickle {len(big)} Item rows: {len(blob) / 1e6:.0f} MB in {spent:.2f}s")
    began = time.perf_counter()
    items = big.iloc[:2000].gs.to_items()
    spent = time.perf_counter() - began
    print(f"to_items 2000 rows: {spent:.2f}s ({spent / 2000 * 1e3:.2f} ms/row)")
