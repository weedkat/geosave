# ruff: noqa
# Throwaway prototype run; see README.md.
import sys, time
from pathlib import Path
import numpy as np, xarray as xr
from odc.geo import xy_
from odc.geo.gridspec import GridSpec
from odc.geo.geobox import GeoBox
import geosave as gs
from cube import open_tree
from geosave.geodata.core.raster.chips import Chips


def clock(label, work):
    began = time.perf_counter()
    result = work()
    print(f"{label:<60} {time.perf_counter() - began:8.3f}s")
    return result


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    ard = gs.open_vector(root / "ard/catalog.parquet")
    spec = GridSpec(
        "EPSG:32633", tile_shape=(1024, 1024), resolution=10, origin=xy_(0, 0)
    )
    tile = spec.tile_geobox((49, 391))
    tree = clock("open ARD tree once on the tile grid", lambda: open_tree(ard, tile))
    s2 = tree["s2-monthly"].to_dataset()
    s1 = tree["s1-monthly"].to_dataset()
    rng = np.random.default_rng(0)
    began = time.perf_counter()
    for _ in range(30):
        y = int(rng.integers(0, 768))
        x = int(rng.integers(0, 768))
        w = {"y": slice(y, y + 256), "x": slice(x, x + 256)}
        s2.isel(w).load()
        s1.isel(w).load()
    print(
        f"ARD chips via one opened tile cube: {30 / (time.perf_counter() - began):.1f} samples/s"
    )

    # Inference: load the block once, tile in memory, vs tile over the lazy cube.
    catalog = gs.open_vector(root / "catalog.parquet")
    block = GeoBox.from_bbox(
        (500000, 4000000, 520480, 4020480), crs="EPSG:32633", resolution=10
    )[0:1024, 0:1024]
    raw = (
        open_tree(catalog[catalog["collection"] == "s2"], block)["s2"]
        .to_dataset()
        .isel(time=slice(0, 4))
    )
    loaded = clock("inference: load 1024² block, 4 dates, once", lambda: raw.load())
    chips = Chips([loaded], 256, overlap=32, mode="reflect")
    clock(
        f"inference: {len(chips)} in-memory chips → arrays",
        lambda: [chips[k].to_array().values for k in chips],
    )
    lazy = Chips([raw], 256, overlap=32, mode="reflect")
    clock(
        f"inference: {len(lazy)} chips read one by one from the cube",
        lambda: [lazy[k].to_array().values for k in lazy],
    )

    # Materialize a region cube with a windowed writer: one read per window, all bands.
    region = open_tree(catalog, block)

    def materialize():
        nodes = {}
        for name, node in region.children.items():
            cube = node.to_dataset()
            template = cube.drop_vars("ids")
            parts = []
            for y in range(0, 1024, 512):
                row = []
                for x in range(0, 1024, 512):
                    row.append(
                        template.isel(y=slice(y, y + 512), x=slice(x, x + 512)).load()
                    )
                parts.append(xr.concat(row, "x"))
            nodes[name] = xr.concat(parts, "y")
        out = xr.DataTree.from_dict(nodes)
        for node in out.subtree:
            for var in node.ds.data_vars.values():
                var.encoding["chunks"] = tuple(
                    min(s, c) for s, c in zip(var.shape, (1, 256, 256))
                )
        out.to_zarr(root / "region.zarr", mode="w")

    clock("materialize 1024² region, 3 collections, all dates → Zarr", materialize)
    cube = xr.open_datatree(root / "region.zarr", engine="zarr", chunks=None)
    began = time.perf_counter()
    for _ in range(30):
        y = int(rng.integers(0, 768))
        x = int(rng.integers(0, 768))
        t = int(rng.integers(0, 8))
        w = {"y": slice(y, y + 256), "x": slice(x, x + 256)}
        cube["s2"].to_dataset().isel(w, time=slice(t, t + 4)).load() if False else cube[
            "s2"
        ].to_dataset().isel({**w, "time": slice(t, t + 4)}).load()
        cube["s1"].to_dataset().isel({**w, "time": slice(t, t + 4)}).load()
        cube["labels"].to_dataset().isel(w).load()
    print(
        f"random 4-step chips from the materialized region Zarr: {30 / (time.perf_counter() - began):.1f} samples/s"
    )
