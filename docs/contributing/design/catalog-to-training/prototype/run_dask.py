# ruff: noqa
# Throwaway prototype run; see README.md.
import sys, time
from pathlib import Path
from odc.geo.geobox import GeoBox
import geosave as gs
from cube import open_tree

if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    catalog = gs.open_vector(root / "catalog.parquet")
    grid = GeoBox.from_bbox(
        (500000, 4000000, 520480, 4020480), crs="EPSG:32633", resolution=10
    )
    s2 = open_tree(catalog, grid)["s2"].to_dataset().drop_vars("ids")
    small = s2.isel(time=slice(0, 3), y=slice(0, 1024), x=slice(0, 1024)).chunk(
        {"time": 1, "y": 512, "x": 512}
    )
    began = time.perf_counter()
    small.to_zarr(root / "virtual.zarr", mode="w")
    reader = s2["B02"].variable._data.array.reader
    print(
        f"dask to_zarr 3×1024²×5 bands: {time.perf_counter() - began:.2f}s, ODC reads {reader.reads} (12 windows)"
    )
