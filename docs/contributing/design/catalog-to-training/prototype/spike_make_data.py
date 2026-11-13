# ruff: noqa
# Throwaway spike (first iteration); see README.md.
"""Synthetic multi-sensor COG catalog: s2 (10 m, 4 single-band assets + 20 m SCL),
s1 (20 m, one 2-band asset, shifted grid, other UTM zone), labels (static, 10 m)."""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pystac
import rasterio
from odc.geo.geobox import GeoBox

import geosave  # noqa: F401
from geosave.geodata.stac import create_catalog, create_stac_item

root = Path(sys.argv[1])
size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
dates = int(sys.argv[3]) if len(sys.argv) > 3 else 12
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def write_cog(path: Path, grid: GeoBox, pixels: np.ndarray, nodata, names) -> None:
    profile = dict(
        driver="COG",
        width=grid.width,
        height=grid.height,
        count=pixels.shape[0],
        dtype=pixels.dtype,
        crs=str(grid.crs),
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
        blocksize=512,
        overview_resampling="nearest",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(pixels)
        for index, name in enumerate(names, start=1):
            dst.set_band_description(index, name)


s2_grid = GeoBox.from_bbox(
    (500000, 4000000, 500000 + size * 10, 4000000 + size * 10),
    crs="EPSG:32633",
    resolution=10,
)
scl_grid = GeoBox.from_bbox(s2_grid.boundingbox, crs="EPSG:32633", resolution=20)
# S1 sits in the neighbouring zone on a 20 m grid, so aligning needs reprojection.
s1_extent = s2_grid.extent.to_crs("EPSG:32634").buffer(2000)
s1_grid = GeoBox.from_geopolygon(s1_extent, resolution=20, crs="EPSG:32634")
label_grid = GeoBox.from_bbox(
    (
        500000 + size * 2,
        4000000 + size * 2,
        500000 + size * 8,
        4000000 + size * 8,
    ),
    crs="EPSG:32633",
    resolution=10,
)

items = []
start = datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc)
began = time.perf_counter()
for day in range(dates):
    when = start + timedelta(days=5 * day)
    sources = {}
    for name in ("B02", "B03", "B04", "B08"):
        path = root / f"s2/{when:%Y%m%d}/{name}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 10000, (1, *s2_grid.shape), dtype="uint16")
        write_cog(path, s2_grid, pixels, 0, [name])
        sources[name] = path
    path = root / f"s2/{when:%Y%m%d}/SCL.tif"
    write_cog(
        path,
        scl_grid,
        rng.integers(0, 12, (1, *scl_grid.shape), dtype="uint8"),
        0,
        ["SCL"],
    )
    sources["SCL"] = path
    items.append(
        create_stac_item(
            sources,
            id=f"S2_{when:%Y%m%d}",
            input_datetime=when,
            collection="s2",
            asset_roles=["data"],
            with_proj=True,
            with_raster=True,
            with_eo=True,
        )
    )

for day in range(dates):
    when = start + timedelta(days=6 * day, hours=7)
    path = root / f"s1/{when:%Y%m%d}/backscatter.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rng.normal(-15, 3, (2, *s1_grid.shape)).astype("float32")
    write_cog(path, s1_grid, pixels, np.nan, ["VV", "VH"])
    items.append(
        create_stac_item(
            {"backscatter": path},
            id=f"S1_{when:%Y%m%d}",
            input_datetime=when,
            collection="s1",
            asset_roles=["data"],
            with_proj=True,
            with_raster=True,
            with_eo=True,
        )
    )

path = root / "labels/landcover.tif"
path.parent.mkdir(parents=True, exist_ok=True)
write_cog(
    path,
    label_grid,
    rng.integers(1, 6, (1, *label_grid.shape), dtype="uint8"),
    0,
    ["class"],
)
items.append(
    create_stac_item(
        {"class": path},
        id="landcover",
        properties={
            "start_datetime": "2024-01-01T00:00:00Z",
            "end_datetime": "2024-12-31T23:59:59Z",
        },
        collection="labels",
        asset_roles=["data"],
        with_proj=True,
        with_raster=True,
        with_eo=True,
    )
)
print(f"wrote COGs + items in {time.perf_counter() - began:.1f}s")

collections = []
for name in ("s2", "s1", "labels"):
    collections.append(
        pystac.Collection(
            name,
            f"synthetic {name}",
            pystac.Extent(
                pystac.SpatialExtent([[-180, -90, 180, 90]]),
                pystac.TemporalExtent([[start, None]]),
            ),
        )
    )
catalog = create_catalog(items, collections=collections)
print(catalog.gs.to_geoparquet(root / "catalog.parquet", overwrite=True))
print(catalog[["id", "collection", "datetime"]].head())
print(catalog.columns.tolist())
