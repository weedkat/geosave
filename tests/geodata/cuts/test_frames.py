"""Windows are cut along time into frames, each collection on its own dates."""

from __future__ import annotations

import geopandas as gpd
import orjson
import pandas as pd
import pytest

from geosave.geodata import cuts
from geosave.geodata.warnings import DroppedFramesWarning, DroppedInstantsWarning


def _days(frame, collection: str) -> list[str]:
    members = orjson.loads(frame["sources"])[collection]
    return [item["properties"]["datetime"][:10] for item in members]


def test_frames_take_each_collections_own_nearest_dates(items) -> None:
    cut = cuts.frames(cuts.stacks(items, on="label"), 2, tolerance="10D")

    assert cut.index.tolist() == [0, 1]
    assert cut["parent"].tolist() == [0, 0]
    first, second = cut.iloc[0], cut.iloc[1]
    assert _days(first, "s2") == ["2024-03-08", "2024-03-28"]
    assert _days(first, "s1") == ["2024-03-05", "2024-04-04"]
    # The frame ends on Jul 1, yet optical answers with the scene it took on Jul 3.
    assert _days(second, "s2") == ["2024-04-12", "2024-07-03"]


def test_a_static_collection_joins_every_frame_whole(items) -> None:
    cut = cuts.frames(cuts.stacks(items, on="label"), 2, tolerance="10D")

    for _, frame in cut.iterrows():
        sources = orjson.loads(frame["sources"])
        assert len(sources["dem"]) == 1 and len(sources["label"]) == 1


def test_a_frame_states_what_its_instants_span(items) -> None:
    cut = cuts.frames(cuts.stacks(items, on="label"), 2, tolerance="10D")

    assert str(cut.iloc[0]["start_datetime"])[:10] == "2024-03-05"
    assert str(cut.iloc[0]["end_datetime"])[:10] == "2024-04-04"


def test_a_frame_keeps_its_windows_grid(items, grid) -> None:
    cut = cuts.frames(cuts.stacks(items, on="label"), 2, tolerance="10D")

    assert {tuple(shape) for shape in cut["proj:shape"]} == {tuple(grid.shape)}
    assert cut.crs.to_epsg() == 4326


def test_a_frame_reads_one_scene_per_instant(items) -> None:
    frame = cuts.frames(cuts.stacks(items, on="label"), 2, tolerance="10D").iloc[0]

    tree = frame.gs.to_stack()

    assert tree["s2"].dataset.sizes["time"] == 2
    assert tree["s1"].dataset.sizes["time"] == 2
    assert "time" not in tree["dem"].dataset.dims


def test_an_instant_no_collection_reaches_refuses_when_strict(items) -> None:
    with pytest.raises(ValueError, match="observed nothing within"):
        cuts.frames(cuts.stacks(items, on="label"), 2, tolerance="5D")


def test_incomplete_frames_are_dropped_with_a_warning(items) -> None:
    with pytest.warns(DroppedFramesWarning):
        cut = cuts.frames(
            cuts.stacks(items, on="label"), 2, tolerance="5D", mode="drop"
        )

    assert len(cut) == 0


def test_instants_that_fill_no_frame_warn(items) -> None:
    with pytest.warns(DroppedInstantsWarning):
        cut = cuts.frames(cuts.stacks(items, on="label"), 3, tolerance="10D")

    assert len(cut) == 1


def test_windows_with_no_dated_collection_refuse(items) -> None:
    timeless = cuts.stacks(
        items[items["collection"].isin(["dem", "label"])], on="label"
    )

    with pytest.raises(ValueError, match="states no time labels"):
        cuts.frames(timeless, 2, tolerance="10D")


@pytest.mark.parametrize("tolerance", ["0D", "-1D", "NaT"])
def test_a_tolerance_reaching_nothing_refuses(items, tolerance) -> None:
    with pytest.raises(ValueError, match="reaches no scene"):
        cuts.frames(cuts.stacks(items, on="label"), 2, tolerance=tolerance)


def _twice(windows):
    """The same windows listed twice, as two anchors would list them."""
    return gpd.GeoDataFrame(
        pd.concat([windows, windows], ignore_index=True),
        geometry="geometry",
        crs=windows.crs,
    )


def test_no_windows_cut_no_frames_but_keep_the_columns(items) -> None:
    windows = cuts.stacks(items, on="label")

    cut = cuts.frames(windows.iloc[0:0], 2, tolerance="10D")

    assert cut.empty
    assert list(cut.columns) == list(windows.columns)


def test_frames_name_the_window_each_was_cut_from(items) -> None:
    cut = cuts.frames(_twice(cuts.stacks(items, on="label")), 2, tolerance="10D")

    assert cut["parent"].tolist() == [0, 0, 1, 1]
    assert cut.index.tolist() == [0, 1, 2, 3]


def test_drop_may_leave_no_frame_at_all(items) -> None:
    with pytest.warns(DroppedFramesWarning, match="3 of 3"):
        cut = cuts.frames(
            cuts.stacks(items, on="label"), 2, tolerance="1D", mode="drop"
        )

    assert cut.empty


def test_a_collection_of_one_scene_joins_every_frame_whole(items) -> None:
    s1 = items["collection"] == "s1"
    single = gpd.GeoDataFrame(
        pd.concat([items[~s1], items[s1].head(1)]), geometry="geometry", crs=items.crs
    )

    cut = cuts.frames(cuts.stacks(single, on="label"), 2, tolerance="10D")

    assert [_days(frame, "s1") for _, frame in cut.iterrows()] == [["2024-03-05"]] * 2
