"""Time labels group into windows of consecutive slots without reading pixels."""

from collections.abc import Iterable, Iterator, Sequence
from typing import assert_type

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from geosave.geodata.sampling import ChipLocation, RasterChipper, iter_samples
from geosave.geodata.sampling.temporal import (
    choose_bucket_positions,
    find_filled_buckets,
    select_window,
)
from tests.geodata.conftest import _refuse, build_raster
from tests.geodata.sampling.conftest import DATES, build_scene, build_tree, format_times


def test_sequences_and_single_raster_or_stream_preserve_the_element_type():
    scene = build_scene()
    sequence: Sequence[xr.Dataset] = (scene,)
    assert_type(iter_samples(sequence, "D"), Iterator[xr.Dataset])

    # A DataArray also implements Iterable[Any]. The direct branch must keep its type.
    def first(source: xr.DataArray | Iterable[xr.Dataset]):
        windows = assert_type(
            iter_samples(source, "D"), Iterator[xr.DataArray | xr.Dataset]
        )
        chips = RasterChipper(windows, size=4)
        return assert_type(
            next(iter(chips)), tuple[ChipLocation, xr.DataArray | xr.Dataset]
        )

    _, band_chip = first(scene.red)
    _, dataset_chip = first(iter(sequence))
    xr.testing.assert_identical(band_chip, dataset_chip.red)


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", [["06-03", "07-08"], ["09-11", "10-01"]]),
        ("last", [["06-28", "07-23"], ["09-11", "10-21"]]),
    ],
)
def test_a_slot_keeps_one_observation_and_an_empty_slot_ends_a_window(keep, expected):
    windows = iter_samples([build_tree()], "MS", size=2, keep=keep)

    assert [format_times(w["image"]) for w in windows] == expected


def test_stride_spaces_the_windows():
    single = iter_samples([build_tree()], "MS")
    assert [format_times(w["image"]) for w in single] == [
        ["06-28"],
        ["07-23"],
        ["09-11"],
        ["10-21"],
    ]

    pairs = iter_samples([build_tree()], "MS", size=2, stride=2)
    assert [format_times(w["image"]) for w in pairs] == [["06-28", "07-23"]]


def test_a_dataset_and_a_dataarray_split_into_lazy_windows():
    scene = build_scene()
    scene = scene.assign_coords(
        start_datetime=("time", DATES), end_datetime=("time", DATES)
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        windows = list(iter_samples([scene], "MS", size=2, keep="last"))
        band_windows = list(iter_samples([scene["red"]], "MS", size=2, keep="first"))

    assert tasks == []
    assert [format_times(window) for window in windows] == [
        ["06-28", "07-23"],
        ["09-11", "10-21"],
    ]
    assert windows[0]["red"].chunks is not None
    # The period coordinates follow their time labels.
    assert (
        windows[1]["start_datetime"].values.tolist()
        == scene["start_datetime"].values[[5, 7]].tolist()
    )
    assert (windows[0]["red"].values == scene["red"].values[[2, 4]]).all()
    assert [format_times(window) for window in band_windows] == [
        ["06-03", "07-08"],
        ["09-11", "10-01"],
    ]


def test_a_tree_windows_its_dated_nodes_and_keeps_static_ones_whole():
    tree = build_tree()

    windows = list(iter_samples([tree], "MS", size=2))

    assert len(windows) == 2
    assert format_times(windows[1]["image"]) == ["09-11", "10-21"]
    assert dict(windows[1]["label"].sizes) == {"y": 10, "x": 10}


def test_windows_need_a_shared_datetime_time_dimension():
    with pytest.raises(ValueError, match="time"):
        next(iter_samples([build_scene(times=None)], "MS"))
    with pytest.raises(ValueError, match="time"):
        next(iter_samples([build_scene().isel(time=0)], "MS"))
    with pytest.raises(ValueError, match="time"):
        next(iter_samples([build_tree(times=None)], "MS"))

    scene = build_scene()
    observations = scene.rename_dims(time="observation")
    with pytest.raises(ValueError, match="datetime `time` dimension"):
        next(iter_samples(observations, "MS"))


def test_stopping_after_one_window_slices_only_one_tree(monkeypatch):
    tree = build_tree()
    slices = []
    original = xr.DataTree.map_over_datasets

    def record_slice(self, *args, **kwargs):
        slices.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataTree, "map_over_datasets", record_slice)
    windows = iter_samples([tree], "MS", size=2)
    assert iter(windows) is windows
    assert not slices
    assert format_times(next(windows)["image"]) == ["06-28", "07-23"]
    assert len(slices) == 1


def test_windowing_requires_a_datetime_dimension_but_keeps_scalar_dated_leaves():
    scene = build_scene()
    static = scene.isel(time=0)
    tree = xr.DataTree.from_dict({"image": scene, "label": static})
    window = next(iter_samples([tree], "MS", size=2))
    xr.testing.assert_identical(window["label"].to_dataset(), static)

    with pytest.raises(ValueError, match="datetime"):
        next(iter_samples([xr.DataTree(static)], "MS", size=2))
    with pytest.raises(TypeError, match="not datetimes"):
        next(iter_samples([scene.assign_coords(time=range(len(DATES)))], "MS"))


def test_each_input_keeps_its_own_timeline_and_is_requested_on_demand():
    consumed = []

    def rasters():
        for month in (1, 2):
            consumed.append(month)
            start = pd.Timestamp(2025, month, 1)
            yield build_scene(times=pd.date_range(start, periods=2))

    windows = iter_samples(rasters(), "D", size=2)
    assert consumed == []
    assert format_times(next(windows)) == ["01-01", "01-02"]
    assert consumed == [1]
    assert [format_times(window) for window in windows] == [["02-01", "02-02"]]


def test_sampling_keeps_source_positions_on_a_shared_unsorted_timeline():
    source = build_raster(times=2).isel(time=[1, 0, 0]).chunk()
    tree = xr.DataTree.from_dict({"image": source, "label": build_raster()})

    with dask.config.set(scheduler=_refuse):
        windows = list(iter_samples([tree], "D", keep="last"))

    assert len(windows) == 2
    for window, position in zip(windows, [2, 0], strict=True):
        xr.testing.assert_identical(
            window["image"].gs.to_dataset(), source.isel(time=[position])
        )
        xr.testing.assert_identical(
            window["label"].gs.to_dataset(), tree["label"].gs.to_dataset()
        )


def test_independent_timelines_meet_by_slot_without_reading_pixels():
    source = build_raster(times=2).chunk()
    tree = xr.DataTree.from_dict({"a": source, "b": source.isel(time=[1, 0])})

    with dask.config.set(scheduler=_refuse):
        windows = list(iter_samples([tree], "D"))

    for window, day in zip(windows, [0, 1], strict=True):
        xr.testing.assert_identical(window["a"].to_dataset(), source.isel(time=[day]))
        xr.testing.assert_identical(window["b"].to_dataset(), source.isel(time=[day]))


def build_sensors() -> xr.DataTree:
    """Sentinel-2 and Sentinel-1 on their own dates, and a static label."""
    s2 = build_scene(times=pd.DatetimeIndex(["2025-06-03", "2025-06-28", "2025-07-23"]))
    s1 = build_scene(times=pd.DatetimeIndex(["2025-06-10", "2025-07-04", "2025-07-16"]))
    return xr.DataTree.from_dict(
        {
            "s2": s2,
            "s1": s1[["red"]].rename(red="vv"),
            "label": build_tree()["label"].to_dataset(),
        }
    )


def test_sensors_on_their_own_dates_share_monthly_slots():
    windows = list(iter_samples(build_sensors(), "MS", size=2))

    assert len(windows) == 1
    assert format_times(windows[0]["s2"]) == ["06-28", "07-23"]
    assert format_times(windows[0]["s1"]) == ["06-10", "07-16"]
    assert dict(windows[0]["label"].to_dataset().sizes) == {"y": 10, "x": 10}


def test_a_slot_empty_in_one_sensor_skips_or_refuses_the_window():
    tree = build_sensors()
    tree["s1"] = tree["s1"].to_dataset().isel(time=[1, 2])  # July only

    assert [format_times(w["s2"]) for w in iter_samples(tree, "MS")] == [["07-23"]]
    with pytest.raises(ValueError, match=r"empty MS buckets \['2025-06-01'\]"):
        next(iter_samples(tree, "MS", missing="raise"))


def test_a_group_holding_time_selects_its_nodes_together():
    scene = build_scene()
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"title": "monthly"}),
            "s2": xr.Dataset(coords={"time": scene.time}),
            "s2/image": scene.drop_vars("time"),
            "label": build_tree()["label"].to_dataset(),
        }
    )

    window = next(iter_samples(tree, "MS", size=2))

    assert format_times(window["s2/image"].to_dataset()) == ["06-28", "07-23"]
    assert window.attrs == {"title": "monthly"}


def test_a_window_with_an_empty_slot_raises_before_any_window_is_yielded():
    # August holds no observation; June-July is complete but comes first.
    windows = iter_samples([build_tree()], "MS", size=2, missing="raise")

    with pytest.raises(ValueError, match=r"empty MS buckets \['2025-08-01'\]"):
        next(windows)


def test_a_composite_fills_the_slots_a_raw_series_skips():
    scene = build_scene()
    monthly = scene.gs.composite("MS", "median", interpolation="linear")

    raw = list(iter_samples([scene], "MS", size=5))
    filled = list(iter_samples([monthly], "MS", size=5, missing="raise"))

    assert raw == []
    assert [format_times(window) for window in filled] == [
        ["06-01", "07-01", "08-01", "09-01", "10-01"]
    ]


@pytest.mark.parametrize(
    ("freq", "filled_labels"),
    [
        # Month starts label their buckets by the first day.
        ("MS", ["2025-06-01", "2025-07-01", "2025-08-01"]),
        # Month ends and weeks close on the right and label by the last day.
        ("ME", ["2025-06-30", "2025-07-31", "2025-08-31"]),
        ("W", ["2025-06-08", "2025-07-06", "2025-08-03", "2025-08-10"]),
    ],
)
def test_buckets_follow_the_pandas_labels_of_each_frequency(freq, filled_labels):
    # Monday, Monday (last of June), Thursday (last of July), Monday.
    times = pd.DatetimeIndex(["2025-06-02", "2025-06-30", "2025-07-31", "2025-08-04"])

    positions = choose_bucket_positions(times, freq, "last")

    filled = pd.DatetimeIndex(filled_labels)
    assert positions.dropna().index.equals(filled)
    assert positions.index[0] == filled[0]
    assert positions.index[-1] == filled[-1]


def test_a_bucket_keeps_its_first_or_last_observation_and_marks_empty_ones():
    times = pd.DatetimeIndex(["2025-06-03", "2025-06-28", "2025-06-28", "2025-08-12"])

    last = choose_bucket_positions(times, "MS", "last")
    first = choose_bucket_positions(times, "MS", "first")

    # Of two observations on one date, "last" keeps the later position.
    assert last.tolist()[0] == 2
    assert first.tolist()[0] == 0
    assert np.isnan(last.tolist()[1])
    assert last.tolist()[2] == 3


def test_a_bucket_is_filled_only_when_every_dated_raster_has_an_observation():
    spring = build_raster(times=2).assign_coords(
        time=pd.DatetimeIndex(["2025-03-10", "2025-04-10"])
    )
    summer = build_raster(times=2).assign_coords(
        time=pd.DatetimeIndex(["2025-04-20", "2025-06-01"])
    )
    static = build_raster()

    filled = find_filled_buckets([spring, summer, static], "MS")

    # Rasters that never overlap in March, May or June leave those buckets empty.
    assert filled.index.equals(pd.date_range("2025-03-01", periods=4, freq="MS"))
    assert filled.tolist() == [False, True, False, False]


def test_an_empty_or_missing_timeline_fills_no_bucket():
    raster = build_raster(times=2)

    assert not find_filled_buckets([raster, raster.isel(time=[])], "D").any()
    with pytest.raises(ValueError, match="datetime `time` dimension"):
        find_filled_buckets([build_raster()], "D")


def test_a_window_selects_lazily_and_leaves_static_rasters_alone():
    times = pd.DatetimeIndex(["2025-06-03", "2025-06-28", "2025-07-23"])
    raster = build_raster(times=3).assign_coords(time=times).chunk()
    buckets = pd.DatetimeIndex(["2025-06-01", "2025-07-01"])

    with dask.config.set(scheduler=_refuse):
        window = select_window(raster, "MS", buckets, "last")
        band = select_window(raster["red"], "MS", buckets, "first")

    assert format_times(window) == ["06-28", "07-23"]
    assert format_times(band) == ["06-03", "07-23"]
    static = build_raster()
    assert select_window(static, "MS", buckets, "last") is static


def test_weekly_buckets_close_on_sunday_in_every_sensor():
    s2 = build_raster(times=2).assign_coords(
        time=pd.DatetimeIndex(["2025-06-02", "2025-06-09"])  # Mondays
    )
    s1 = build_raster(times=2).assign_coords(
        time=pd.DatetimeIndex(["2025-06-08", "2025-06-12"])  # Sunday, Thursday
    )
    tree = xr.DataTree.from_dict({"s2": s2, "s1": s1})

    windows = list(iter_samples(tree, "W"))

    assert [format_times(window["s2"]) for window in windows] == [["06-02"], ["06-09"]]
    assert [format_times(window["s1"]) for window in windows] == [["06-08"], ["06-12"]]
