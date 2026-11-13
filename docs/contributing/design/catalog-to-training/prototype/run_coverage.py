# ruff: noqa
# Throwaway prototype run; see README.md.
"""Design A planning: coverage per chip × step from the footprint coordinate, then window starts."""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from odc.geo.geobox import GeoBox

import geosave as gs
from cube import coverage, open_tree

if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    catalog = gs.open_vector(root / "catalog.parquet")
    # A grid larger than the S2 tile, so edge chips are partly or not covered.
    grid = GeoBox.from_bbox(
        (495000, 3995000, 525720, 4025720), crs="EPSG:32633", resolution=10
    )
    s2 = open_tree(catalog, grid)["s2"].to_dataset()
    boxes = []
    for y in range(0, grid.shape.y - 255, 256):
        for x in range(0, grid.shape.x - 255, 256):
            boxes.append(grid[y : y + 256, x : x + 256].extent.geom)
    chips = gpd.GeoDataFrame(geometry=boxes, crs="EPSG:32633")
    began = time.perf_counter()
    fraction = coverage(s2, chips)
    full = (fraction >= 0.99).astype(int)
    window = full.rolling(time=4).min().shift(time=-3).fillna(0).astype(bool)
    starts = np.argwhere(window.values)
    print(
        f"coverage {fraction.shape} + window starts in {time.perf_counter() - began:.3f}s"
    )
    print(
        f"{len(chips)} chips; fully covered chips: {(full.sum('time') == s2.sizes['time']).sum().item()}; partly: {((fraction > 0) & (fraction < 0.99)).any('time').sum().item()}; windows: {len(starts)}"
    )
    print("lazy after coverage:", type(s2["B04"].variable._data).__name__)
