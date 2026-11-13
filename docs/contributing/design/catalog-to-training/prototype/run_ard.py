# ruff: noqa
# Throwaway prototype run; see README.md.
"""Design C: GridSpec tiles → monthly masked composites → COG ARD catalog → chip reads."""

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from odc.geo import xy_
from odc.geo.gridspec import GridSpec

import geosave as gs
from cube import open_cube, open_tree
from publish import to_catalog


def clock(label, work):
    began = time.perf_counter()
    result = work()
    print(f"{label:<60} {time.perf_counter() - began:8.3f}s")
    return result


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    catalog = gs.open_vector(root / "catalog.parquet")
    spec = GridSpec(
        "EPSG:32633", tile_shape=(1024, 1024), resolution=10, origin=xy_(0, 0)
    )
    labels = catalog[catalog["collection"] == "labels"]
    from odc.geo.geom import Geometry

    aoi = Geometry(labels.geometry.union_all(), crs="EPSG:4326")
    tiles = list(spec.tiles_from_geopolygon(aoi))
    print(
        len(tiles), "GridSpec tiles over the label AOI:", [index for index, _ in tiles]
    )

    def composite(tile):
        tree = open_tree(catalog[catalog["collection"] != "labels"], tile)
        s2 = tree["s2"].to_dataset()
        clear = ~s2["SCL"].isin([0, 3, 8, 9, 10])
        bands = s2[["B02", "B03", "B04", "B08"]].where(clear)
        monthly = bands.resample(time="MS").median().round().fillna(0).astype("uint16")
        for name in monthly.data_vars:
            monthly[name].attrs = {"_FillValue": 0, "long_name": name}
        months = monthly.time.to_index()
        monthly = monthly.assign_coords(
            start_datetime=("time", months),
            end_datetime=(
                "time",
                months
                + months.to_series()
                .dt.days_in_month.map(lambda d: np.timedelta64(int(d), "D"))
                .to_numpy(),
            ),
        )
        s1 = tree["s1"].to_dataset().resample(time="MS").mean().sel(time=monthly.time)
        s1 = s1.assign_coords(
            start_datetime=("time", monthly.start_datetime.values),
            end_datetime=("time", monthly.end_datetime.values),
        )
        for name in s1.data_vars:
            s1[name].attrs = {"_FillValue": np.nan, "long_name": name}
        return monthly, s1

    index, tile = tiles[0]
    monthly, s1 = clock(
        f"read 12 dates × 2 sensors + composite one 1024² tile {index}",
        lambda: composite(tile),
    )
    print(monthly)
    tag = f"{index[0]}_{index[1]}"
    s2_rows = clock(
        "publish S2 monthly as COG + Items",
        lambda: to_catalog(
            monthly,
            root / "ard/s2-monthly",
            collection_id="s2-monthly",
            format_id="s2m_" + tag + "_{time:%Y%m}",
        ),
    )
    s1_rows = clock(
        "publish S1 monthly as COG + Items",
        lambda: to_catalog(
            s1,
            root / "ard/s1-monthly",
            collection_id="s1-monthly",
            format_id="s1m_" + tag + "_{time:%Y%m}",
        ),
    )
    ard = gs.GeoDataFrame.gs.concat([s2_rows, s1_rows]) if False else None
    import geopandas as gpd

    from geosave.geodata.core.vector.catalog import GeoCatalog

    ard = GeoCatalog.concat([s2_rows, s1_rows])
    path = ard.gs.to_geoparquet(root / "ard/catalog.parquet", overwrite=True)
    ard = gs.open_vector(path)
    print(ard[["id", "collection", "start_datetime", "end_datetime"]].to_string())

    rng = np.random.default_rng(0)

    def read_chip():
        y = int(rng.integers(0, 768))
        x = int(rng.integers(0, 768))
        chip = tile[y : y + 256, x : x + 256]
        tree = open_tree(ard, chip)
        return tree["s2-monthly"].to_dataset().load(), tree[
            "s1-monthly"
        ].to_dataset().load()

    clock("random 256² chip from the ARD COGs (no warp)", read_chip)
    began = time.perf_counter()
    for _ in range(20):
        read_chip()
    print(
        f"ARD chip reads: {20 / (time.perf_counter() - began):.1f} samples/s (1 process)"
    )
