"""Temporal membership is independent of collection order and pixel loading."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box
import geosave.geodata  # noqa: F401


def test_date_selection_of_plain_features_preserves_original_positions():
    rows = catalog(["2025-01-05", "2025-02-05", None]).drop(columns="collection")
    rows.index = [7, 7, 2]

    samples = rows.gs.sample(date_range=("2025-02-01", "2025-03-01"))

    assert samples.indices == [[1, 2]]
    assert samples[0].equals(rows.iloc[[1, 2]])


def test_sampling_validates_only_selected_geometries():
    rows = catalog(["2025-01-05", "2025-01-05"], ["optical", "excluded"])
    rows.loc[1, "geometry"] = None

    assert rows.gs.sample(collections=["optical"]).indices == [[0]]


def test_filtered_slots_keep_source_positions_with_duplicate_labels():
    rows = catalog(["2025-01-05", "2025-02-05", "2025-03-05", "2025-04-05"])
    rows.index = [8, 8, 3, 3]

    samples = rows.gs.sample(
        date_range=("2025-02-01", "2025-04-01"), freq="MS", window=2
    )

    assert samples.indices == [[1, 2]]
    assert samples.slots[0]["indices"].tolist() == [[1], [2]]
    assert samples[0].equals(rows.iloc[[1, 2]])


def test_exact_sampling_preserves_distinct_submicrosecond_instants():
    rows = catalog(["2025-01-01T00:00:00.000000001", "2025-01-01T00:00:00.000000002"])

    samples = rows.gs.sample(collections=["optical"])

    assert samples.indices == [[0], [1]]
    assert [labels[0] for labels in samples.times] == list(
        pd.to_datetime(rows.datetime)
    )


def catalog(dates, collections=None, starts=None, ends=None):
    count = len(dates)
    values = {
        "id": [f"item-{i}" for i in range(count)],
        "datetime": dates,
        "collection": collections or ["optical"] * count,
    }
    if starts is not None:
        values["start_datetime"] = starts
        values["end_datetime"] = ends
    return gpd.GeoDataFrame(values, geometry=[box(0, 0, 1, 1)] * count, crs=4326)


def test_collections_fill_calendar_slots_independently():
    rows = catalog(["2025-01-05", "2025-01-25"], ["time", "footprint"])
    samples = rows.gs.sample(collections=["footprint", "time"], freq="MS")
    assert set(samples.indices[0]) == {0, 1}
    assert samples.times[0].tolist() == [pd.Timestamp("2025-01-01")]


@pytest.mark.parametrize(
    "policy, months", [("first", [1]), ("last", [2]), ("overlap", [1, 2]), ("drop", [])]
)
def test_crossing_interval_assignment(policy, months):
    rows = catalog([None], starts=["2025-01-20"], ends=["2025-02-10"])
    samples = rows.gs.sample(freq="MS", interval=policy)
    assert [times[0].month for times in samples.times] == months
    for indices in samples.indices:
        assert indices == [0]


def test_crossing_interval_raises_with_item_and_remedy():
    rows = catalog([None], starts=["2025-01-20"], ends=["2025-02-01"])
    with pytest.raises(ValueError, match="item-0.*interval"):
        rows.gs.sample(freq="MS")


def test_empty_month_is_not_removed_before_windowing():
    rows = catalog(["2025-01-05", "2025-03-05", "2025-04-05"])
    samples = rows.gs.sample(freq="MS", window=2)
    assert samples.indices == [[1, 2]]
    kept = rows.gs.sample(freq="MS", window=3, missing="keep")
    assert kept.times[0].month.tolist() == [1, 2, 3]
    assert kept.slots[0].iloc[1]["indices"] == []


def test_datetime_assigns_but_does_not_erase_coverage():
    rows = catalog(["2025-02-05"], starts=["2025-01-20"], ends=["2025-02-10"])
    samples = rows.gs.sample(freq="MS")
    assert samples.times[0][0].month == 2
    assert samples[0].iloc[0]["start_datetime"] == "2025-01-20"


def test_exact_groups_preserve_original_positions():
    rows = catalog(["2025-01-05", "2025-01-06", "2025-01-07"])
    rows.loc[1, "geometry"] = box(0, 0, 2, 2)
    rows.index = [9, 9, 4]
    samples = rows.gs.sample(match="exact")
    assert samples.indices == [[0, 2], [1]]


@pytest.mark.parametrize(
    "starts, ends", [(["2025-02-01"], ["2025-01-01"]), (["2025-01-01"], [None])]
)
def test_invalid_coverage_identifies_item(starts, ends):
    rows = catalog([None], starts=starts, ends=ends)
    with pytest.raises(ValueError, match="item-0"):
        rows.gs.sample(freq="MS")


def test_interval_only_requires_calendar_slots():
    rows = catalog([None], starts=["2025-01-01"], ends=["2025-01-31"])
    with pytest.raises(ValueError, match="freq"):
        rows.gs.sample(collections=["optical"])


def test_incomplete_windows_are_explicitly_kept_or_dropped():
    rows = catalog(["2025-01-05", "2025-03-05"])
    assert len(rows.gs.sample(freq="MS", window=3)) == 0
    samples = rows.gs.sample(freq="MS", window=3, missing="keep")
    assert samples.indices == [[0, 1]]
    assert samples.slots[0].iloc[1]["indices"] == []


def test_nominal_times_normalize_timezones_before_assignment():
    rows = catalog(["2025-02-01T00:30:00+02:00"])
    assert rows.gs.sample(freq="MS").times[0][0] == pd.Timestamp("2025-01-01")


def test_date_range_filters_before_calendar_grouping():
    rows = catalog(["2025-01-05", "2025-01-20", "2025-02-15", "2025-03-01"])
    samples = rows.gs.sample(date_range=("2025-01-15", "2025-03-01"), freq="MS")
    assert samples.indices == [[1], [2]]
    assert [time[0] for time in samples.times] == list(
        pd.to_datetime(["2025-01-01", "2025-02-01"])
    )


def test_relative_months_use_explicit_anchor_across_spatial_groups():
    rows = catalog(["2025-01-20", "2025-02-16", "2025-02-18"])
    rows.loc[2, "geometry"] = box(5, 5, 6, 6)
    samples = rows.gs.sample(
        date_range=("2025-01-15", None),
        freq=pd.DateOffset(months=1),
        window=2,
        missing="keep",
    )
    assert samples.indices == [[0, 1], [2]]
    for times in samples.times:
        assert times.tolist() == list(pd.to_datetime(["2025-01-15", "2025-02-15"]))


def test_unbounded_range_filters_interval_coverage_without_rewriting_rows():
    rows = catalog(
        [None, None],
        starts=["2025-01-10", "2025-03-01"],
        ends=["2025-02-10", "2025-04-01"],
    )
    samples = rows.gs.sample(date_range=(None, "2025-02-01"))
    assert samples.indices == [[0]]
    assert samples[0].iloc[0]["end_datetime"] == "2025-02-10"


def test_bounded_interval_assignment_uses_only_selected_periods():
    rows = catalog([None], starts=["2025-01-10"], ends=["2025-04-10"])
    samples = rows.gs.sample(date_range=("2025-02-01", "2025-03-01"), freq="MS")
    assert samples.indices == [[0]]
    assert samples.times[0].tolist() == [pd.Timestamp("2025-02-01")]


def test_relative_months_require_anchor():
    with pytest.raises(ValueError, match="date_range"):
        catalog(["2025-01-20"]).gs.sample(freq=pd.DateOffset(months=1))


def test_samples_preserve_membership_after_source_sort():
    rows = catalog(["2025-02-01", "2025-01-01"])
    samples = rows.gs.sample(freq="MS")
    before = samples[0]["id"].tolist()
    rows.sort_values("datetime", inplace=True)
    assert samples[0]["id"].tolist() == before


def test_fixed_duration_anchor_preserves_empty_leading_period():
    rows = catalog(["2025-01-26", "2025-02-05"])
    samples = rows.gs.sample(
        date_range=("2025-01-15", "2025-02-14"),
        freq="10D",
        window=3,
        missing="keep",
    )
    assert samples.indices == [[0, 1]]
    assert samples.times[0].tolist() == list(
        pd.to_datetime(["2025-01-15", "2025-01-25", "2025-02-04"])
    )
    assert samples.slots[0].iloc[0]["indices"] == []


def test_range_timezone_and_empty_selection():
    rows = catalog(["2025-01-14T23:30:00Z", "2025-01-15T00:30:00Z"])
    samples = rows.gs.sample(
        date_range=("2025-01-15T01:00:00+02:00", "2025-01-15T02:00:00+02:00")
    )
    assert samples.indices == [[0]]
    assert len(rows.gs.sample(date_range=("2026-01-01", None), freq="MS")) == 0


@pytest.mark.parametrize(
    "bounds", [("2025-02-01", "2025-01-01"), ("2025-01-01", "2025-01-01")]
)
def test_inverted_or_empty_range_is_rejected(bounds):
    with pytest.raises(ValueError, match="date_range"):
        catalog(["2025-01-20"]).gs.sample(date_range=bounds)
