"""Cutting sample trees into chips, and merging chip outputs back."""

import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

from geosave.geodata import Chips
from tests.conftest import build_dataset, build_tree
from geosave.geodata.core.raster.chips import lay_grid

CRS = "EPSG:32633"


def build_sample(left: int, width: int, height: int, crs: str = CRS):
    """Return a sample tree: s2 at 10 m with random pixels, s1 at 20 m, a label."""
    rng = np.random.default_rng(left)
    bounds = (left, 5000000, left + width * 10, 5000000 + height * 10)
    fine = GeoBox.from_bbox(bounds, crs, resolution=10)
    coarse = GeoBox.from_bbox(bounds, crs, resolution=20)
    times = pd.to_datetime(["2024-01-05", "2024-02-04"])
    s2 = build_dataset(
        {"red": (("time", "y", "x"), rng.random((2, *fine.shape), dtype="float32"))},
        fine,
        time=times,
    )
    s1 = build_dataset(
        {"vv": (("time", "y", "x"), rng.random((2, *coarse.shape), dtype="float32"))},
        coarse,
        time=times,
    )
    label = build_dataset(
        {"class": (("y", "x"), rng.integers(0, 3, fine.shape).astype("uint8"))}, fine
    )
    s2 = s2.gs.write_nodata(np.nan).chunk({"x": 16, "y": 16})
    s1 = s1.gs.write_nodata(np.nan).chunk({"x": 16, "y": 16})
    return build_tree(
        {"s2": s2, "s1": s1, "label": label.chunk({"x": 16, "y": 16})}
    )


@pytest.fixture
def trees():
    """A tree that chips divide evenly, and one they do not."""
    return [build_sample(300000, 64, 64), build_sample(320000, 70, 50)]


def test_chips_count_over_every_tree(trees):
    chips = Chips(trees, size=32, resolution=10)

    assert len(chips) == 2 * 2 + 2 * 3


def test_a_chip_names_its_source_tree(trees):
    chips = Chips(trees, size=32, resolution=10)

    assert chips[0][0] == 0
    assert chips[len(chips) - 1][0] == 1


def test_a_chip_puts_every_group_on_its_grid_lazily(trees):
    chips = Chips(trees, size=32, resolution=10, overlap=8)

    _, chip = chips[0]

    for raster in chip.gs.rasters.values():
        assert raster.gs.geobox == chips.grid(0)
        for variable in raster.data_vars.values():
            assert variable.chunks is not None
    assert chip["s1"].gs.to_dataset().sizes["time"] == 2


@pytest.mark.parametrize("overlap", [0, 8])
def test_merging_each_chips_own_pixels_rebuilds_the_tree(trees, overlap):
    chips = Chips(trees, size=32, resolution=10, overlap=overlap)

    for index in range(len(chips)):
        _, chip = chips[index]
        chips.add(index, chip["s2"].gs.to_dataset()["red"].isel(time=0).values)
    rasters = chips.merge()

    for tree, raster in zip(trees, rasters, strict=True):
        source = tree["s2"].gs.to_dataset()
        assert raster.gs.geobox == source.gs.geobox
        np.testing.assert_allclose(
            raster.isel(band=0), source["red"].isel(time=0), atol=1e-6
        )


def test_merging_keeps_output_channels(trees):
    chips = Chips(trees[:1], size=32, resolution=10)

    for index in range(len(chips)):
        chips.add(index, np.ones((3, 32, 32), "float32"))

    assert chips.merge()[0].sizes["band"] == 3


def test_merging_needs_outputs_for_every_tree(trees):
    chips = Chips(trees, size=32, resolution=10)
    chips.add(0, np.ones((32, 32), "float32"))

    with pytest.raises(ValueError, match="tree 1 received no outputs"):
        chips.merge()


def test_a_tree_in_two_crs_cannot_be_chipped():
    tree = build_sample(300000, 32, 32)
    other = build_sample(300000, 32, 32, crs="EPSG:32634")
    mixed = build_tree({"s2": tree["s2"].gs.to_dataset(), "s1": other["s1"].gs.to_dataset()})

    with pytest.raises(ValueError, match="one CRS"):
        Chips([mixed], size=16, resolution=10)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (32, 32), (32, -1)])
def test_chip_sizes_are_checked(trees, size, overlap):
    with pytest.raises(ValueError, match="chips need"):
        Chips(trees, size=size, resolution=10, overlap=overlap)


def test_lay_grid_covers_groups_on_their_own_grids():
    optical_grid = GeoBox.from_bbox((0, 0, 640, 640), "EPSG:32633", resolution=10)
    radar_grid = GeoBox.from_bbox((320, 0, 960, 640), "EPSG:32633", resolution=20)
    optical = build_dataset(
        {"red": (("y", "x"), np.zeros(optical_grid.shape, "float32"))}, optical_grid
    )
    radar = build_dataset(
        {"vv": (("y", "x"), np.zeros(radar_grid.shape, "float32"))}, radar_grid
    )

    grid = lay_grid(build_tree({"optical": optical, "radar": radar}), 10)

    assert tuple(grid.boundingbox) == (0, 0, 960, 640)
    assert grid.shape == (64, 96)


def test_lay_grid_needs_one_crs():
    rasters = {}
    for name, crs in [("north", "EPSG:32633"), ("east", "EPSG:32634")]:
        grid = GeoBox.from_bbox((0, 0, 160, 160), crs, resolution=10)
        rasters[name] = build_dataset(
            {"red": (("y", "x"), np.zeros(grid.shape, "float32"))}, grid
        )

    with pytest.raises(ValueError, match="one grid needs one CRS"):
        lay_grid(build_tree(rasters), 10)
