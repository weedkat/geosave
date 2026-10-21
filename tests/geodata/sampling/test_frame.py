import warnings

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

import geosave as gs
from geosave.geodata import raster, tree
from geosave.geodata.warnings import DroppedRowsWarning, TimelessItemWarning
from tests.geodata.conftest import item_table, tree_templates

WINDOWS = {"s2": {"slots": ["-1M", "0M"]}, "s1": {"slots": ["-1M", "0M"]}}


def test_frame_pairs_composites_by_coverage(composites):
    with pytest.warns(DroppedRowsWarning):
        frames = composites.gs.sample().gs.frame(windows=WINDOWS)
    assert frames.reference_time.tolist() == [
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2024-05-01"),
    ]
    for _, row in frames.iterrows():
        opened = row.gs.to_xarray()
        assert opened["s2"].dataset.sizes["time"] == 2
        assert opened["s1"].dataset.sizes["time"] == 2
        assert set(opened.gs.groups) == {"s1", "s2"}


@pytest.mark.parametrize("stride,count", [(None, 2), (1, 4), (2, 2)])
def test_stride_steps_in_candidate_timestamps(composites, stride, count):
    with pytest.warns(DroppedRowsWarning):
        frames = composites.gs.sample().gs.frame(windows=WINDOWS, stride=stride)
    assert len(frames) == count


def test_frame_drops_timeless_clock(grid, tmp_path):
    dem = tree({"dem": raster({"height": (("y", "x"), np.zeros(grid.shape))}, grid)})
    with pytest.warns(TimelessItemWarning):
        rows = item_table(dem.gs.to_cog(tree_templates(dem, tmp_path)), prefix="a")
    with pytest.warns(DroppedRowsWarning, match="reference times"):
        result = rows.gs.sample().gs.frame(windows={"dem": {"slots": ["0D"]}})
    assert result.empty


@pytest.mark.parametrize(
    "config",
    [
        {"slots": []},
        {"slots": ["-1Q"]},
        {"slots": ["0D", "-1D"]},
        {"slots": ["0D", "0D"]},
        {"slots": ["0D"], "reduce": "unknown"},
        {"slots": ["0D"], "fill": "forward"},
    ],
)
def test_invalid_window_is_rejected(composites, config):
    with pytest.raises(ValueError):
        composites.gs.sample().gs.frame(windows={"s2": config})


def test_frame_refuses_invalid_stride(composites):
    with pytest.raises(ValueError, match="positive"):
        composites.gs.sample().gs.frame(windows=WINDOWS, stride=0)


def test_frame_and_chip_commute(composites, grid):
    samples = composites.gs.sample()
    with pytest.warns(DroppedRowsWarning):
        first = samples.gs.frame(windows=WINDOWS, stride=1).gs.chip(32, target=grid)
    with pytest.warns(DroppedRowsWarning):
        second = samples.gs.chip(32, target=grid).gs.frame(windows=WINDOWS, stride=1)
    columns = ["reference_time", "proj:transform", "pad_widths", "items", "frame"]
    assert sorted(map(str, first[columns].values.tolist())) == sorted(
        map(str, second[columns].values.tolist())
    )


def test_reference_time_is_not_rounded_to_calendar_boundary(items):
    labels = items[items.collection == "label"].assign(
        datetime=pd.Timestamp("2024-03-08T10:15:00Z"),
        start_datetime=None,
        end_datetime=None,
    )
    result = labels.gs.sample().gs.frame(windows={"label": {"slots": ["0h"]}})
    assert result.reference_time.tolist() == [pd.Timestamp("2024-03-08T10:15:00")]


def catalog(tmp_path, scenes, *, nodata=np.nan):
    grid = GeoBox.from_bbox((0, 0, 20, 20), "EPSG:3857", resolution=10)
    entries = []
    for index, (name, date, value, bounds) in enumerate(scenes):
        data = gs.raster(
            {"value": (("y", "x"), da.full((2, 2), float(value), chunks=(2, 2)))},
            grid,
            nodata=nodata,
        )
        if date is not None:
            data = data.assign_coords(time=np.datetime64(date, "us"))
        if bounds:
            data = data.assign_coords(
                start_datetime=np.datetime64(bounds[0], "us"),
                end_datetime=np.datetime64(bounds[1], "us"),
            )
        path = tmp_path / f"{index}.tif"
        data.gs.to_cog(path)
        entries.extend(
            gs.stac.create_items({"value": path}, collection=name, id_format=str(index))
        )
    return gs.stac.create_catalog(entries)


def test_weekly_composites_and_reference_survive_parquet(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 2, ("2024-01-01", "2024-01-07T23:59:59.999999")),
            ("image", "2024-01-08", 4, ("2024-01-08", "2024-01-14T23:59:59.999999")),
            ("label", "2024-01-15", 9, None),
        ],
    )
    frames = rows.gs.sample().gs.frame(
        windows={"image": {"slots": ["-2W", "-1W"]}}, reference="label"
    )
    assert len(frames) == 1
    restored = gs.open_vector(frames.gs.to_geoparquet(tmp_path / "frames.parquet"))
    tree = restored.iloc[0].gs.to_xarray()
    image = tree["image"].dataset
    assert image.sizes["time"] == 2
    assert image.value.chunks is not None
    np.testing.assert_allclose(image.value[:, 0, 0].values, [2, 4])
    np.testing.assert_allclose(tree["label"].dataset.value.values, 9)
    np.testing.assert_array_equal(
        image.start_datetime.values,
        np.array(["2024-01-01", "2024-01-08"], dtype="datetime64[us]"),
    )


def test_reduce_first_selects_whole_acquisition_and_interpolates_missing_bin(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 2, None),
            ("image", "2024-01-02", 99, None),
            ("image", "2024-01-15", 6, None),
            ("label", "2024-01-22", 9, None),
        ],
    )
    frames = rows.gs.sample().gs.frame(
        windows={
            "image": {
                "slots": ["-3W", "-2W", "-1W"],
                "reduce": "first",
                "fill": "interpolate",
            }
        },
        reference="label",
    )
    image = frames.iloc[0].gs.to_xarray()["image"].dataset
    assert image.value.chunks is not None
    np.testing.assert_allclose(image.value[:, 0, 0].values, [2, 4, 6])


def test_crossing_composite_is_not_a_weekly_observation(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 2, ("2024-01-01", "2024-01-31T23:59:59.999999")),
            ("label", "2024-01-15", 9, None),
        ],
    )
    with pytest.warns(DroppedRowsWarning):
        frames = rows.gs.sample().gs.frame(
            windows={"image": {"slots": ["-1W"]}}, reference="label"
        )
    assert frames.empty


def test_default_reference_is_newest_first_and_stride_counts_candidates(tmp_path):
    rows = catalog(tmp_path, [("image", f"2024-01-0{i}", i, None) for i in range(1, 6)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DroppedRowsWarning)
        frames = rows.gs.sample().gs.frame(windows={"image": {"slots": ["-2D", "-1D"]}})
    assert frames.reference_time.tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-03"),
    ]
    np.testing.assert_allclose(
        frames.iloc[0].gs.to_xarray()["image"].dataset.value[:, 0, 0], [3, 4]
    )


def test_multiple_observations_require_reducer(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 1, None),
            ("image", "2024-01-02", 2, None),
            ("label", "2024-01-08", 9, None),
        ],
    )
    with pytest.raises(ValueError, match="reduce"):
        rows.gs.sample().gs.frame(
            windows={"image": {"slots": ["-1W"]}}, reference="label"
        )


def test_interpolation_does_not_blend_nodata_sentinels(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", -9999, None),
            ("image", "2024-01-15", 5, None),
            ("label", "2024-01-22", 9, None),
        ],
        nodata=-9999,
    )
    samples = rows.gs.sample()
    frames = samples.gs.frame(
        windows={"image": {"slots": ["-3W", "-2W", "-1W"], "fill": "interpolate"}},
        reference="label",
    )
    image = frames.iloc[0].gs.to_xarray()["image"].dataset
    assert np.isnan(image.value.isel(time=1).values).all()


def test_timeless_auxiliary_survives_framing(tmp_path):
    rows = catalog(
        tmp_path, [("image", "2024-01-01", 1, None), ("dem", None, 10, None)]
    )
    frame = rows.gs.sample().gs.frame(windows={"image": {"slots": ["0D"]}}).iloc[0]
    dem = frame.gs.to_xarray()["dem"].dataset
    assert "time" not in dem.dims
    np.testing.assert_allclose(dem.value.values, 10)


@pytest.mark.parametrize("method,expected", [("first", 2), ("last", 6), ("mean", 4)])
def test_same_timestamp_acquisitions_reduce_as_separate_observations(
    tmp_path, method, expected
):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 2, None),
            ("image", "2024-01-01", 6, None),
            ("label", "2024-01-08", 9, None),
        ],
    )
    frame = (
        rows.gs.sample()
        .gs.frame(
            windows={"image": {"slots": ["-1W"], "reduce": method}}, reference="label"
        )
        .iloc[0]
    )
    np.testing.assert_allclose(
        frame.gs.to_xarray()["image"].dataset.value.values, expected
    )


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
