"""A window row reads itself as an aligned lazy stack."""

from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest

from geosave_engine.geodata import cuts


def test_a_window_reads_every_collection_on_its_grid(items, sample, grid) -> None:
    window = cuts.stacks(items, on="label").iloc[0]

    tree = window.gs.to_stack()

    assert sorted(tree.gs.groups) == ["dem", "label", "s1", "s2"]
    for raster in tree.gs.rasters.values():
        assert raster.gs.geobox == grid
    assert dask.is_dask_collection(tree["s2"].dataset.red)
    np.testing.assert_array_equal(
        tree["label"].dataset.label.values, sample["label"].dataset.label.values
    )


def test_a_row_states_the_grid_its_window_reads_onto(items, grid) -> None:
    windows = cuts.stacks(items, on="label")
    chip = cuts.chips(windows, (20, 30)).iloc[1]

    assert windows.iloc[0].gs.geobox == grid
    assert chip.gs.geobox == grid[0:20, 30:60]


def test_a_row_without_sources_has_no_sample_accessor() -> None:
    with pytest.raises(AttributeError, match="cuts table"):
        pd.Series({"a": 1}).gs


def test_a_reflect_halo_fills_only_past_the_parent(items, sample) -> None:
    chip = cuts.chips(cuts.stacks(items, on="label"), 32, overlap=8).iloc[0]

    label = chip.gs.to_stack()["label"].dataset.label.values

    (top, _), (left, _) = chip["pad"]
    source = sample["label"].dataset.label.values
    np.testing.assert_array_equal(label[top:, left:], source[: 32 - top, : 32 - left])
    np.testing.assert_array_equal(label[top - 1, left:], source[1, : 32 - left])


def test_a_saved_windows_table_reads_the_same_pixels(items, tmp_path) -> None:
    from geosave_engine.geodata import open_vector

    cut = cuts.chips(cuts.stacks(items, on="label"), 32)
    path = cut.gs.to_geoparquet(tmp_path / "windows.parquet")

    back = open_vector(path)

    a = cut.iloc[3].gs.to_stack()["s2"].dataset.red.values
    b = back.iloc[3].gs.to_stack()["s2"].dataset.red.values
    np.testing.assert_array_equal(a, b)
