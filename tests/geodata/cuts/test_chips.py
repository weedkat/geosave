"""Windows are cut in space into chips of one size."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import orjson
import pandas as pd
import pytest

from geosave_engine.geodata import cuts


def test_chips_without_overlap_tile_the_window(items) -> None:
    windows = cuts.stacks(items, on="label")

    cut = cuts.chips(windows, 32)

    # 50 by 70 pixels in 32-pixel chips: two rows of three.
    assert cut.index.tolist() == list(range(6))
    assert cut["chip"].tolist() == list(range(6))
    assert cut["parent"].tolist() == [0] * 6
    assert {tuple(shape) for shape in cut["proj:shape"]} == {(32, 32)}
    assert cut.iloc[0]["halo"] == [[0, 0], [0, 0]]
    # The last column and row reach past the window's 70 and 50 pixels.
    assert cut.iloc[2]["pad"] == [[0, 0], [0, 26]]
    assert cut.iloc[3]["pad"] == [[0, 14], [0, 0]]
    assert cut.iloc[0]["pad"] == [[0, 0], [0, 0]]


def test_overlapping_chips_are_laid_over_a_halo(items) -> None:
    cut = cuts.chips(cuts.stacks(items, on="label"), 32, overlap=8)

    tiler, halo = cuts.layout((50, 70), (32, 32), overlap=8)
    assert len(cut) == len(tiler)
    assert cut.iloc[0]["halo"] == [list(widths) for widths in halo]
    assert cut.iloc[0]["pad"][0][0] == halo[0][0]
    assert cut.iloc[0]["pad"][1][0] == halo[1][0]
    assert cut.iloc[0]["mode"] == "reflect"


def test_a_chip_states_its_own_grid_and_footprint(items, grid) -> None:
    cut = cuts.chips(cuts.stacks(items, on="label"), 32)

    chip = grid.translate_pix(32, 32).crop((32, 32))
    row = cut.iloc[4]
    assert list(row["proj:transform"]) == pytest.approx(list(chip.transform)[:6])
    assert row.geometry.equals_exact(chip.extent.to_crs("EPSG:4326").geom, 1e-9)


def test_chips_of_a_frame_keep_its_sources_and_name_it(items) -> None:
    framed = cuts.frames(cuts.stacks(items, on="label"), 2, tolerance="10D")

    cut = cuts.chips(framed, 32)

    assert cut["parent"].tolist() == [0] * 6 + [1] * 6
    assert orjson.loads(cut.iloc[0]["sources"]) == orjson.loads(
        framed.iloc[0]["sources"]
    )


def test_a_size_states_height_then_width(items) -> None:
    cut = cuts.chips(cuts.stacks(items, on="label"), (25, 35))

    assert {tuple(shape) for shape in cut["proj:shape"]} == {(25, 35)}
    assert len(cut) == 4


def test_a_layout_is_the_same_every_time() -> None:
    first, halo = cuts.layout((300, 500), (128, 128), overlap=32)
    second, again = cuts.layout((300, 500), (128, 128), overlap=32)

    assert halo == again
    assert len(first) == len(second)
    assert all(
        np.array_equal(first.get_tile_bbox(n)[0], second.get_tile_bbox(n)[0])
        for n in range(len(first))
    )


def test_a_chip_larger_than_its_window_pads_past_it(items) -> None:
    cut = cuts.chips(cuts.stacks(items, on="label"), 128)

    (chip,) = cut.to_dict("records")
    assert list(chip["proj:shape"]) == [128, 128]
    # 50 by 70 pixels inside, the rest filled past the bottom and right edges.
    assert chip["pad"] == [[0, 78], [0, 58]]


def test_a_chip_the_size_of_its_window_needs_no_pad(items) -> None:
    cut = cuts.chips(cuts.stacks(items, on="label"), (50, 70))

    (chip,) = cut.to_dict("records")
    assert chip["pad"] == [[0, 0], [0, 0]]
    assert list(chip["proj:transform"]) == list(cut.iloc[0]["proj:transform"])


def test_no_windows_cut_no_chips_but_keep_the_columns(items) -> None:
    windows = cuts.stacks(items, on="label")

    cut = cuts.chips(windows.iloc[0:0], 32)

    assert cut.empty
    assert list(cut.columns) == [
        *windows.columns,
        "chip",
        "halo",
        "overlap",
        "mode",
        "pad",
    ]


@pytest.mark.parametrize("overlap", [32, 40, -4])
def test_an_overlap_outside_the_chip_refuses(items, overlap) -> None:
    with pytest.raises(ValueError, match="overlap"):
        cuts.chips(cuts.stacks(items, on="label"), 32, overlap=overlap)


def test_chips_name_the_window_each_was_cut_from(items) -> None:
    windows = cuts.stacks(items, on="label")
    twice = gpd.GeoDataFrame(
        pd.concat([windows, windows], ignore_index=True),
        geometry="geometry",
        crs=windows.crs,
    )

    cut = cuts.chips(twice, 32)

    assert cut["parent"].tolist() == [0] * 6 + [1] * 6
    assert cut["chip"].tolist() == list(range(6)) * 2
