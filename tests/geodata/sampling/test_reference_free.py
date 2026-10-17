"""Public spatial grouping and common-clock regression cases."""

import pandas as pd
import pytest

from geosave.geodata.core import GeoSamples
from geosave.geodata.warnings import DroppedRowsWarning


def test_sample_groups_equal_footprints_and_keeps_containing_scenes_separate(items):
    items = items.copy()
    radar = items.collection == "s1"
    items.loc[radar, "geometry"] = items.loc[radar, "geometry"].translate(0.01)
    samples = items.gs.sample()
    assert isinstance(samples.gs, GeoSamples)
    assert len(samples) == 2
    assert samples.iloc[0].gs.geobox is None
    assert set(samples.iloc[0].gs.items) == {"label", "dem", "s2"}
    assert set(samples.iloc[1].gs.items) == {"s1"}
    assert len(samples.iloc[0].gs.items["s2"]) == 5


def test_gridless_sample_requires_explicit_chip_grid(items, grid):
    samples = items.gs.sample()
    with pytest.raises(ValueError, match="target"):
        samples.gs.chip(32)
    chips = samples.gs.chip(32, target=grid)
    assert all(row.gs.geobox is not None for _, row in chips.iterrows())


def test_frames_select_four_inclusive_history_periods(items):
    images = items[items.collection == "s2"].copy()
    images["datetime"] = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], utc=True
    )
    images["start_datetime"] = None
    images["end_datetime"] = None
    with pytest.warns(DroppedRowsWarning):
        frames = images.gs.sample().gs.frame(
            windows={"s2": {"slots": ["-3D", "-2D", "-1D", "0D"]}}, stride=1
        )
    assert frames.start_datetime.tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-04"),
    ]
    assert len(frames.iloc[0].gs.items["s2"]) == 4


def test_explicit_grid_aligns_acquisitions_before_time_assembly(tmp_path):
    import dask.array as da
    import numpy as np
    from odc.geo.geobox import GeoBox
    import geosave as gs

    fine = GeoBox.from_bbox((0, 0, 40, 40), "EPSG:3857", resolution=10)
    coarse = GeoBox.from_bbox((0, 0, 40, 40), "EPSG:3857", resolution=20)
    entries = []
    for index, box in enumerate([fine, coarse]):
        source = gs.raster(
            {
                "red": (
                    ("y", "x"),
                    da.full(tuple(box.shape), index + 1.0, chunks=tuple(box.shape)),
                )
            },
            box,
            nodata=np.nan,
        )
        source = source.assign_coords(time=np.datetime64(f"2024-01-0{index + 1}", "us"))
        entries.extend(
            gs.stac.create_items(
                source.gs.to_cog(tmp_path / f"{index}.tif"),
                collection="image",
                id_format=str(index),
            )
        )
    row = gs.stac.create_catalog(entries).gs.sample().iloc[0]
    with pytest.raises(ValueError):
        row.gs.to_xarray()
    tree = row.gs.to_xarray(target=fine, resampling={"image/red": "nearest"})
    raster = tree["image"].dataset
    assert raster.sizes == {"time": 2, "y": 4, "x": 4}
    assert raster.red.chunks is not None
    np.testing.assert_allclose(raster.red.isel(time=0).values, 1)
    np.testing.assert_allclose(raster.red.isel(time=1).values, 2)
