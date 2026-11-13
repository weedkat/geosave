# ruff: noqa
# Throwaway prototype run; see README.md.
"""Inference block over the virtual cube → Chips → merge → publish → reload equal."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr

import geosave as gs
from cube import open_cube, open_tree
from geosave.geodata.core.raster.chips import Chips
from publish import to_catalog

root = Path(sys.argv[1])
catalog = gs.open_vector(root / "catalog.parquet")
grid = GeoBox.from_bbox(
    (500000, 4000000, 520480, 4020480), crs="EPSG:32633", resolution=10
)
block = grid[0:1024, 0:1024]
tree = open_tree(catalog[catalog["collection"] != "labels"], block)
s2 = tree["s2"].to_dataset().isel(time=slice(0, 4))

began = time.perf_counter()
chips = Chips([s2], 256, overlap=32, mode="reflect")


def model(chip: xr.Dataset) -> np.ndarray:
    """Stand-in classifier: class = argmax over the median of 4 bands."""
    pixels = chip[["B02", "B03", "B04", "B08"]].to_array().median("time").values
    return np.eye(4, dtype="float32")[pixels.argmax(0)].transpose(2, 0, 1)


merged = chips.merge((key, model(chips[key])) for key in chips)
scores = merged[0]
print(
    f"inference over {len(chips)} chips: {time.perf_counter() - began:.2f}s",
    scores.shape,
)

labels = wrap_xr(scores.argmax(0).astype("uint8"), block).rename("landcover")
labels.attrs.update(
    flag_values=np.array([0, 1, 2, 3], dtype="uint8"),
    flag_meanings="water forest crop urban",
    long_name="landcover",
)
prediction = labels.to_dataset().expand_dims(time=[pd.Timestamp(s2.time.values[-1])])
prediction = prediction.assign_coords(
    start_datetime=("time", [pd.Timestamp(s2.start_datetime.values[0])]),
    end_datetime=("time", [pd.Timestamp(s2.end_datetime.values[-1])]),
)
print(prediction)

rows = to_catalog(
    prediction,
    root / "predictions",
    collection_id="landcover-pred",
    format_id="pred_{time:%Y%m%d}",
)
print(rows[["id", "collection", "start_datetime", "end_datetime"]])
asset = rows.iloc[0]["assets"]["landcover"]
print("asset fields:", sorted(k for k, v in asset.items() if v is not None))
print(
    "classes:",
    [c["name"] for c in asset["classification:classes"]]
    if asset.get("classification:classes") is not None
    else asset,
)
path = rows.gs.to_geoparquet(root / "predictions/catalog.parquet", overwrite=True)

again = gs.open_vector(path)
reloaded = open_cube(again, block).load()
print(
    "published → reloaded pixels equal:",
    np.array_equal(reloaded["landcover"].values[0], labels.values),
)
print("time cell kept:", reloaded.start_datetime.values, reloaded.end_datetime.values)
import rustac

print("rustac reads published catalog:", len(rustac.read_sync(str(path))["features"]))
