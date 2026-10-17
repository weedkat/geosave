import numpy as np
import pandas as pd
import pytest

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
