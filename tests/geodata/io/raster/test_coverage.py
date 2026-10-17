"""Composite coverage survives files, path-built Items and catalog persistence."""

from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

import geosave as gs
from geosave.geodata.transform import temporal


@pytest.mark.parametrize("explicit", [False, True], ids=["resampled", "explicit"])
def test_saved_paths_preserve_composite_coverage(tmp_path: Path, explicit: bool):
    grid = GeoBox.from_bbox((0, 0, 20, 20), "EPSG:3857", resolution=10)
    source = gs.raster(
        {"red": (("time", "y", "x"), da.full((2, 2, 2), 5.0, chunks=(1, 2, 2)))},
        grid,
        time=pd.to_datetime(["2024-02-05", "2024-02-20"]),
        nodata=np.nan,
    )
    monthly = temporal.reduce(temporal.resample(source, "MS"), "median")
    if explicit:
        monthly.time.attrs = {}
        monthly = monthly.assign_coords(
            start_datetime=np.datetime64("2024-02-01T00:00:00", "us"),
            end_datetime=np.datetime64("2024-02-29T23:59:59.999999", "us"),
        )
    monthly.gs.to_cog(tmp_path / "{time:%Y%m}.tif")
    collection = gs.stac.create_collection("image", description="Monthly imagery")
    items = list(
        gs.stac.create_items({"image": tmp_path / "202402.tif"}, collection=collection)
    )
    assert items[0].properties["start_datetime"] == "2024-02-01T00:00:00Z"
    assert items[0].properties["end_datetime"] == "2024-02-29T23:59:59.999999Z"
    table = gs.stac.create_catalog(items, collections=[collection])
    restored = gs.open_vector(table.gs.to_geoparquet(tmp_path / "catalog.parquet"))
    raster = restored.iloc[0].gs.to_xarray()
    assert raster.to_array().chunks is not None
    assert raster.start_datetime.values == np.datetime64("2024-02-01")
    assert raster.end_datetime.values == np.datetime64("2024-02-29T23:59:59.999999")
    assert raster.gs.timespan == (
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-02-29T23:59:59.999999"),
    )
    np.testing.assert_allclose(raster.to_array().values, 5)
