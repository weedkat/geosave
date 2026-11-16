"""A RasterChipper lays fixed-size chips over prepared rasters and cuts them lazily."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from odc.geo.geobox import GeoBox

from geosave.geodata.errors import RasterValidationError
from geosave.geodata.sampling import (
    ChipLayout,
    ChipLocation,
    RasterChipper,
    find_chip_span,
    lay_chip_offsets,
)
from tests.geodata.sampling.conftest import GRID, build_tree, format_times

SMALL_GRID = GeoBox.from_bbox(
    (600_000.0, 9_100_000.0, 600_080.0, 9_100_060.0),
    crs="EPSG:32749",
    shape=(6, 8),
    tight=True,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("shift", [0, 4, 6]), ("drop", [0, 4]), ("reflect", [0, 4, 8]), ("constant", [0, 4, 8])],
)
def test_each_mode_meets_the_edge_its_own_way(mode, expected):
    assert lay_chip_offsets(10, size=4, stride=4, mode=mode) == expected


def test_offsets_that_fit_exactly_are_the_same_in_every_mode():
    for mode in ("shift", "drop", "edge", "wrap"):
        assert lay_chip_offsets(8, size=4, stride=4, mode=mode) == [0, 4]
        assert lay_chip_offsets(10, size=4, stride=3, mode=mode) == [0, 3, 6]


def test_an_axis_shorter_than_a_chip_needs_padding():
    with pytest.raises(ValueError, match="fewer than one chip"):
        lay_chip_offsets(3, size=4, stride=4, mode="shift")
    assert lay_chip_offsets(3, size=4, stride=4, mode="drop") == []
    assert lay_chip_offsets(3, size=4, stride=4, mode="reflect") == [0]


def test_chips_are_cut_lazily_on_their_own_grid():
    tree = build_tree(times=None)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        chipper = RasterChipper([tree], size=4)
        chip = chipper[1]

    assert tasks == []
    assert len(chipper) == 9
    assert chipper.layout.geobox(1) == GRID[0:4, 4:8]
    assert chip.gs.geobox == GRID[0:4, 4:8]
    assert dict(chip["image"].sizes) == {"y": 4, "x": 4}
    assert chip["image"]["red"].chunks is not None
    np.testing.assert_array_equal(
        chip["image"]["red"].values, tree["image"]["red"].values[0:4, 4:8]
    )
    np.testing.assert_array_equal(
        chip["label"]["cover"].values, tree["label"]["cover"].values[0:4, 4:8]
    )


def test_shift_puts_the_last_chip_flush_with_the_corner():
    chipper = RasterChipper([build_tree(times=None)], size=4, mode="shift")

    assert chipper.layout.geobox(8) == GRID[6:10, 6:10]
    assert chipper.layout.geobox(-1) == GRID[6:10, 6:10]


def test_a_padded_chip_reaches_past_the_edge_and_is_filled():
    tree = build_tree(times=None)
    chipper = RasterChipper([tree], size=4, mode="constant", constant_value=7)

    chip = chipper[8]

    assert len(chipper) == 9
    assert chip.gs.geobox == chipper.layout.geobox(8)
    assert chip.gs.geobox.extent.boundingbox.right == GRID.extent.boundingbox.right + 20
    red = chip["image"]["red"].values
    np.testing.assert_array_equal(red[:2, :2], tree["image"]["red"].values[8:, 8:])
    assert (red[2:, :] == 7).all()
    assert (red[:, 2:] == 7).all()
    assert chip["label"]["cover"].dtype == "uint8"


def test_drop_leaves_incomplete_chips_out():
    chipper = RasterChipper([build_tree(times=None)], size=4, mode="drop")

    assert len(chipper) == 4
    assert chipper.layout.geobox(3) == GRID[4:8, 4:8]


def test_chips_number_through_several_rasters_on_their_own_grids():
    scene = build_tree(times=None)
    other = build_tree(SMALL_GRID, times=None)
    chipper = RasterChipper([scene, other], size=4, mode="shift")

    # 3 x 3 chips of the 10 x 10 scene, then 2 x 2 of the 6 x 8 one.
    assert len(chipper) == 9 + 4
    assert chipper.layout.locate(0) == ChipLocation(raster_index=0, top=0, left=0)
    assert chipper.layout.locate(8) == ChipLocation(raster_index=0, top=6, left=6)
    assert chipper.layout.locate(9) == ChipLocation(raster_index=1, top=0, left=0)
    assert chipper.layout.locate(12) == ChipLocation(raster_index=1, top=2, left=4)
    assert chipper.layout.geobox(9) == SMALL_GRID[0:4, 0:4]
    assert chipper[12].gs.geobox == SMALL_GRID[2:6, 4:8]
    assert len(list(chipper)) == 13
    with pytest.raises(IndexError):
        chipper[13]
    with pytest.raises(IndexError):
        chipper.layout.locate(-14)


def test_time_windows_are_rasters_like_any_other():
    tree = build_tree()
    windows = tree.gs.sample("MS", size=2, keep="last")
    chipper = RasterChipper(windows, size=5)

    assert len(chipper) == 2 * 4
    assert format_times(chipper[0]["image"]) == ["06-28", "07-23"]
    assert format_times(chipper[4]["image"]) == ["09-11", "10-21"]
    assert dict(chipper[4]["label"].sizes) == {"y": 5, "x": 5}


def test_a_chip_converts_to_tensors():
    torch = pytest.importorskip("torch")
    windows = build_tree().gs.sample("MS", size=2)
    chipper = RasterChipper(windows, size=4, stride=2, mode="reflect")

    tensors = chipper[len(chipper) - 1].gs.to_tensor(context=["time"])

    assert tensors["image"].shape == (2, 2, 4, 4)
    assert tensors["label"].shape == (1, 4, 4)
    assert tensors["time"].shape == (2,)
    batch = torch.utils.data.default_collate([chipper[0].gs.to_tensor(context=["time"]), tensors])
    assert batch["image"].shape == (2, 2, 2, 4, 4)


def test_chipper_refuses_what_it_cannot_lay_out():
    tree = build_tree(times=None)
    with pytest.raises(ValueError, match="fewer than one chip"):
        RasterChipper([tree], size=16, mode="shift")
    assert len(RasterChipper([tree], size=16)) == 1
    with pytest.raises(ValueError, match="mode must be"):
        RasterChipper([tree], size=4, mode="mirror")
    with pytest.raises(ValueError, match="at least 1"):
        RasterChipper([tree], size=4, stride=0)
    assert len(RasterChipper([], size=4)) == 0

    coarse = tree["label"].to_dataset().coarsen(x=2, y=2).max()
    two_grids = xr.DataTree.from_dict({"image": tree["image"].to_dataset(), "coarse": coarse})
    with pytest.raises(RasterValidationError, match="GeoBoxes"):
        RasterChipper([two_grids], size=4)


def test_reflect_is_the_default_and_mirrors_the_raster_edge():
    tree = build_tree(times=None)
    chipper = RasterChipper([tree], size=4)

    assert chipper.layout.mode == "reflect"
    assert chipper.layout.locate(8) == ChipLocation(raster_index=0, top=8, left=8)
    red = chipper[8]["image"]["red"].values
    whole = np.pad(tree["image"]["red"].values, ((0, 2), (0, 2)), mode="reflect")
    np.testing.assert_array_equal(red, whole[8:12, 8:12])


def test_a_layout_places_chips_from_grids_alone():
    chipper = RasterChipper([build_tree(times=None)], size=4, stride=2)

    layout = ChipLayout([GRID], size=4, stride=2)

    assert len(layout) == len(chipper)
    assert layout.locate(7) == chipper.layout.locate(7)
    assert layout.geobox(7) == chipper[7].gs.geobox


@pytest.mark.parametrize(("size", "stride"), [(4, 3), (4, 4), (5, 2), (8, 3)])
def test_a_span_with_its_halo_holds_every_chip_that_touches_it(size, stride):
    length = 29
    offsets = lay_chip_offsets(length, size=size, stride=stride, mode="reflect")

    for start in range(0, length, 2 * stride):
        stop = min(start + 2 * stride, length)
        first, last = find_chip_span(start, stop, length, size=size, stride=stride)

        touching = [o for o in offsets if o < stop and o + size > start]
        assert first <= touching[0]
        assert first % stride == 0
        assert last == min(touching[-1] + size, length)
        # A chip past the end reflects pixels the span must hold.
        reach = touching[-1] + size - length
        if reach > 0:
            assert first <= max(0, length - 1 - reach)
