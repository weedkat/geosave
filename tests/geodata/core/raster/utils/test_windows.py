"""Time labels group into windows of consecutive slots without reading pixels."""

import pytest
from dask.callbacks import Callback

from geosave.geodata.core.raster.utils.windows import window_times
from tests.geodata.sampling.conftest import DATES, build_scene, build_tree, format_times


def format_positions(windows):
    """Show each window's kept observations as month-day text."""
    shown = []
    for positions in windows:
        shown.append([f"{DATES[position]:%m-%d}" for position in positions])
    return shown


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", [["06-03", "07-08"], ["09-11", "10-01"]]),
        ("last", [["06-28", "07-23"], ["09-11", "10-21"]]),
    ],
)
def test_a_slot_keeps_one_observation_and_an_empty_slot_ends_a_window(keep, expected):
    windows = window_times(DATES, "MS", size=2, keep=keep)

    assert format_positions(windows) == expected


def test_stride_spaces_the_windows():
    single = window_times(DATES, "MS")
    assert format_positions(single) == [["06-28"], ["07-23"], ["09-11"], ["10-21"]]

    pairs = window_times(DATES, "MS", size=2, stride=2)
    assert format_positions(pairs) == [["06-28", "07-23"]]


def test_windows_refuse_an_unknown_policy_or_an_empty_size():
    with pytest.raises(ValueError, match="keep"):
        window_times(DATES, "MS", keep="nearest")
    with pytest.raises(ValueError, match="at least 1"):
        window_times(DATES, "MS", size=0)


def test_a_dataset_and_a_dataarray_split_into_lazy_windows():
    scene = build_scene()
    scene = scene.assign_coords(
        start_datetime=("time", DATES), end_datetime=("time", DATES)
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        windows = scene.gs.sample("MS", size=2, keep="last")
        band_windows = scene["red"].gs.sample("MS", size=2, keep="first")

    assert tasks == []
    assert [format_times(window) for window in windows] == [
        ["06-28", "07-23"],
        ["09-11", "10-21"],
    ]
    assert windows[0]["red"].chunks is not None
    # The period coordinates follow their time labels.
    assert windows[1]["start_datetime"].values.tolist() == scene["start_datetime"].values[[5, 7]].tolist()
    assert (windows[0]["red"].values == scene["red"].values[[2, 4]]).all()
    assert [format_times(window) for window in band_windows] == [
        ["06-03", "07-08"],
        ["09-11", "10-01"],
    ]


def test_a_tree_windows_its_dated_nodes_and_keeps_static_ones_whole():
    tree = build_tree()

    windows = tree.gs.sample("MS", size=2)

    assert len(windows) == 2
    assert format_times(windows[1]["image"]) == ["09-11", "10-21"]
    assert dict(windows[1]["label"].sizes) == {"y": 10, "x": 10}


def test_windows_need_a_shared_datetime_time_dimension():
    import xarray as xr

    with pytest.raises(ValueError, match="time"):
        build_scene(times=None).gs.sample("MS")
    with pytest.raises(ValueError, match="time"):
        build_scene().isel(time=0).gs.sample("MS")
    with pytest.raises(ValueError, match="time"):
        build_tree(times=None).gs.sample("MS")

    scene = build_scene()
    mixed = xr.DataTree.from_dict({"image": scene, "other": scene.isel(time=slice(0, 3))})
    with pytest.raises(ValueError, match="own time axis"):
        mixed.gs.sample("MS")
