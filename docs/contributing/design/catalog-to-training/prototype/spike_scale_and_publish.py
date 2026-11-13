# ruff: noqa
# Throwaway spike (first iteration); see README.md.
"""(1) Planning cost on a 200k-row catalog with native pandas; (2) publish round trip."""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from odc.geo.geobox import GeoBox, GeoboxTiles

import geosave as gs
from geosave.geodata.stac import create_catalog, create_stac_item

root = Path(sys.argv[1])


def clock(label, work):
    began = time.perf_counter()
    result = work()
    print(f"{label:<52} {time.perf_counter() - began:8.3f}s")
    return result


# ---------- (1) planning at scale ----------
rng = np.random.default_rng(1)
tiles = 400  # 400 MGRS-like tiles of 100 km, 500 dates each -> 200k rows
xs = rng.uniform(0, 4_000_000, tiles)
ys = rng.uniform(0, 4_000_000, tiles)
boxes = shapely.box(xs, ys, xs + 100_000, ys + 100_000)
dates = pd.date_range("2018-01-01", periods=500, freq="5D", tz="UTC")
catalog = gpd.GeoDataFrame(
    {
        "id": [f"t{t}_{d}" for t in range(tiles) for d in range(len(dates))],
        "collection": "s2",
        "datetime": np.tile(dates, tiles),
        "eo:cloud_cover": rng.uniform(0, 100, tiles * len(dates)),
    },
    geometry=np.repeat(boxes, len(dates)),
    crs="EPSG:3857",
)
print(len(catalog), "rows")
footprints = clock(
    "group by unique geometry (tiles)",
    lambda: catalog.groupby(catalog.geometry.to_wkb()).size(),
)
print("   unique footprints:", len(footprints))
# 50 random chips of 2.56 km per tile.
chip_x = np.repeat(xs, 50) + rng.uniform(0, 97_000, tiles * 50)
chip_y = np.repeat(ys, 50) + rng.uniform(0, 97_000, tiles * 50)
chips = gpd.GeoDataFrame(
    {"chip": range(tiles * 50)},
    geometry=shapely.box(chip_x, chip_y, chip_x + 2560, chip_y + 2560),
    crs="EPSG:3857",
)
unique = catalog.drop_duplicates(subset="geometry")[["geometry"]].reset_index(
    names="tile_row"
)
covered = clock(
    "sjoin 20k chips x unique footprints",
    lambda: chips.sjoin(unique, predicate="covered_by"),
)
pairs = clock(
    "expand to chip x item rows (merge on footprint)",
    lambda: covered.merge(
        catalog.assign(
            tile_row=catalog.groupby(catalog.geometry.to_wkb())
            .ngroup()
            .map(dict(enumerate(unique["tile_row"])))
        )[["tile_row", "id", "datetime", "eo:cloud_cover"]],
        on="tile_row",
    ),
)
print("   chip x item rows:", len(pairs))
monthly = clock(
    "monthly least-cloud pick per chip (groupby+idxmin)",
    lambda: pairs.loc[
        pairs.groupby(["chip", pairs["datetime"].dt.to_period("M")])[
            "eo:cloud_cover"
        ].idxmin()
    ],
)
print("   chip-month rows:", len(monthly))


# ---------- (2) publish round trip ----------
def publish_round_trip():
    source = gs.open_vector(root / "catalog.parquet")
    s2 = source[source["collection"] == "s2"].iloc[:2]
    box = GeoBox.from_bbox(
        (502000, 4002000, 504560, 4004560), crs="EPSG:32633", resolution=10
    )
    scene = s2.gs.to_dataset(box, chunks=None, bands=["B02", "B04"], groupby="id")
    print(scene)
    tree = scene.gs.expand_stac(format_id="S2_{time:%Y%m%d}")
    paths = tree.gs.to_cog(root / "published", overwrite=True)
    print([str(p) for p in paths])
    items = []
    for node in tree.children.values():
        sources = {}
        for leaf in node.children:
            sources[leaf] = root / "published" / node.name / f"{leaf}.tif"
        items.append(
            create_stac_item(
                sources,
                id=node.name,
                input_datetime=pd.Timestamp(node["time"].values).to_pydatetime(),
                collection="s2-chip",
                with_proj=True,
                with_raster=True,
                with_eo=True,
            )
        )
    table = create_catalog(items)
    path = table.gs.to_geoparquet(root / "published/catalog.parquet", overwrite=True)
    again = gs.open_vector(path).gs.to_dataset(box, chunks=None, groupby="id")
    print(again)
    print(
        "round trip equal B04:",
        np.array_equal(again["B04"].values, scene["B04"].values),
    )
    print("attrs kept B04:", scene["B04"].attrs, "->", again["B04"].attrs)


publish_round_trip()
