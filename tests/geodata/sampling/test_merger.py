"""Chip predictions merge back onto the rasters their chipper cut them from."""

import pickle

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from shapely import Polygon

import geosave as gs
from geosave.geodata.sampling import (
    ChipLocation,
    RasterChipper,
    RasterMerger,
    create_chip_weights,
)
from tests.geodata.sampling.conftest import GRID, build_scene, build_tree
from tests.geodata.sampling.test_chipper import SMALL_GRID


def chip_scores(location: ChipLocation, size: int) -> np.ndarray:
    """Stand in for a model: three bands that differ by chip and by pixel."""
    rows, columns = np.mgrid[0:size, 0:size]
    pixels = (location.top + rows) * 100 + location.left + columns
    return np.stack([pixels, pixels + location.top, -pixels]).astype("float32")


def merge_scores(chips: RasterChipper, locations=None, window="boxcar", store=None):
    """Merge `chip_scores` of the given locations, or every chip, with one merger."""
    if locations is None:
        locations = [location for location, _ in chips]
    merger = RasterMerger(chips, window=window, store=store)
    for location in locations:
        merger.add(location, chip_scores(location, chips.size))
    return merger.merge()


def read_last_date(chip):
    """Stand in for a model: return the bands of the chip's last date."""
    pixels = chip["image"].to_dataset().isel(time=-1).gs.to_numpy()
    return pixels


@pytest.mark.parametrize(
    "mode", ["shift", "drop", "constant", "edge", "reflect", "wrap"]
)
@pytest.mark.parametrize("window", ["boxcar", "hann", "triang", "blackmanharris"])
def test_shuffled_chips_merge_back_to_each_raster_in_input_order(mode, window):
    # 8 x 8 and 6 x 8 rasters, so "drop" still covers every pixel with 2 px overlap.
    even_grid = GRID[0:8, 0:8]
    rasters = [
        *gs.iter_samples([build_tree(even_grid)], "MS", size=2),
        build_tree(SMALL_GRID),
    ]
    chipper = RasterChipper(rasters, size=4, overlap=2, mode=mode)
    chips = list(chipper)

    # Chips of several rasters, interleaved, into one merger.
    merger = RasterMerger(chipper, window=window)
    for index in np.random.default_rng(0).permutation(len(chips)):
        location, chip = chips[index]
        merger.add(location, read_last_date(chip))
    merged = merger.merge()

    assert isinstance(merged, tuple)
    assert len(merged) == len(rasters)
    for raster, scores in zip(rasters, merged, strict=True):
        expected = raster["image"].to_dataset().isel(time=-1).gs.to_numpy()
        assert scores.dims == ("band", "y", "x")
        assert scores.dtype == "float32"
        assert scores.gs.geobox == raster.gs.geobox
        np.testing.assert_allclose(scores.values, expected, rtol=1e-5)


def test_a_taper_never_zeroes_a_pixel_only_rims_cover():
    assert create_chip_weights("boxcar", 4).tolist() == [[1.0] * 4] * 4
    weights = create_chip_weights("hann", 4)
    assert weights.min() > 0
    assert weights[1, 1] == pytest.approx(weights.max())
    assert weights[0, 0] == pytest.approx(weights.min())

    chips = RasterChipper(build_scene(times=None), size=4, overlap=2)
    merger = RasterMerger(chips, window="hann")
    for location, _ in chips:
        merger.add(location, np.full((1, 4, 4), 7.0))

    (merged,) = merger.merge()
    np.testing.assert_allclose(merged.values, 7.0)


def test_overlaps_average_and_uncovered_pixels_are_nan():
    chips = RasterChipper(build_scene(times=None), size=4, overlap=2)
    first, second, *_ = [location for location, _ in chips]
    merger = RasterMerger(chips)
    merger.add(first, np.full((1, 4, 4), 2.0))
    merger.add(second, np.full((1, 4, 4), 4.0))

    (merged,) = merger.merge()

    assert merged.shape == (1, *GRID.shape)
    assert merged.values[0, 0, 0] == 2.0
    assert merged.values[0, 0, 3] == 3.0
    assert merged.values[0, 0, 5] == 4.0
    assert np.isnan(merged.values[0, 9, 9])


@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_overlapping_logits_use_the_chosen_window_before_decoding(window):
    chips = RasterChipper([build_tree(times=None)], size=4, overlap=2)
    merger = RasterMerger(chips, window=window)
    locations = iter(chips)
    for logit in [2.0, 4.0]:
        location, _ = next(locations)
        merger.add(location, np.full((1, 4, 4), logit))

    weights = create_chip_weights(window, 4)
    expected = np.average([2.0, 4.0], weights=[weights[1, 2], weights[1, 0]])
    (merged,) = merger.merge()
    assert merged.values[0, 1, 2] == pytest.approx(expected)


def test_merger_refuses_predictions_it_cannot_place():
    chips = RasterChipper(build_scene(times=None), size=4)
    other = RasterChipper(
        [build_scene(times=None), build_scene(SMALL_GRID, times=None)], size=4
    )
    location, _ = next(iter(chips))
    merger = RasterMerger(chips)

    with pytest.raises(ValueError, match="no chip was added"):
        merger.merge()
    with pytest.raises(ValueError, match=r"\(band, 4, 4\); got \(4, 4\)"):
        merger.add(location, np.zeros((4, 4)))
    with pytest.raises(ValueError, match=r"\(band, 4, 4\); got \(2, 5, 5\)"):
        merger.add(location, np.zeros((2, 5, 5)))
    with pytest.raises(ValueError, match="did not cut"):
        merger.add(ChipLocation(other.sources[1], 0, 0), np.zeros((2, 4, 4)))
    with pytest.raises(ValueError):
        RasterMerger(chips, window="mirror").add(  # pyright: ignore[reportArgumentType]
            location, np.zeros((1, 4, 4))
        )


def test_padding_past_the_raster_is_dropped():
    # 5-pixel chips leave the last chip of each row and column reaching past the edge.
    chips = RasterChipper(build_scene(GRID[:9, :9], times=None), size=5)
    merger = RasterMerger(chips, window="hann")
    for location, _ in chips:
        merger.add(location, np.stack([np.zeros((5, 5)), np.ones((5, 5))]))

    (merged,) = merger.merge()

    assert merged.shape == (2, 9, 9)
    assert (merged.argmax("band") == 1).all()


def test_a_raster_without_chips_merges_to_nan_in_its_place():
    tiny = build_scene(GRID[:3, :3], times=None)
    chips = RasterChipper(
        [build_scene(times=None), tiny], size=4, overlap=2, mode="drop"
    )

    merged = merge_scores(chips)

    assert len(merged) == 2
    assert merged[1].shape == (3, 3, 3)
    assert merged[1].gs.geobox == tiny.gs.geobox
    assert merged[1].isnull().all()


def test_a_copied_location_still_names_its_raster():
    chips = RasterChipper(build_scene(times=None), size=4)
    merger = RasterMerger(chips)
    for location, _ in chips:
        # A DataLoader worker or another process returns a copy of each location.
        merger.add(pickle.loads(pickle.dumps(location)), np.ones((1, 4, 4)))

    (merged,) = merger.merge()
    np.testing.assert_allclose(merged.values, 1)


def test_an_iterator_input_is_held_so_the_chips_can_be_read_again():
    rasters = [build_scene(times=None).red, build_scene(SMALL_GRID, times=None)]
    chips = RasterChipper(iter(rasters), size=4, overlap=2)

    first = [location for location, _ in chips]
    second = [location for location, _ in chips]

    assert first == second
    assert [source.geobox for source in chips.sources] == [GRID, SMALL_GRID]


@pytest.mark.parametrize("mode", ["shift", "drop", "reflect"])
@pytest.mark.parametrize("overlap", [0, 1, 3])
def test_a_staged_merge_equals_the_merge_in_memory_and_stays_lazy(
    tmp_path, mode, overlap
):
    rasters = [build_scene(times=None), build_scene(SMALL_GRID, times=None)]
    chips = RasterChipper(rasters, size=4, overlap=overlap, mode=mode)

    in_memory = merge_scores(chips, window="hann")
    staged = merge_scores(chips, window="hann", store=tmp_path / "merged")

    assert sorted(path.name for path in (tmp_path / "merged").iterdir()) == [
        "0.zarr",
        "1.zarr",
    ]
    for lazy, expected in zip(staged, in_memory, strict=True):
        assert isinstance(lazy.data, da.Array)
        xr.testing.assert_allclose(lazy.compute(), expected)


def test_a_staged_merge_takes_chips_in_any_order(tmp_path):
    chips = RasterChipper(build_scene(times=None), size=4, overlap=2)
    locations = [location for location, _ in chips]
    shuffled = []
    for index in np.random.default_rng(0).permutation(len(locations)):
        shuffled.append(locations[index])

    (staged,) = merge_scores(chips, shuffled, store=tmp_path / "merged")

    (expected,) = merge_scores(chips)
    xr.testing.assert_allclose(staged.compute(), expected)


def test_a_staged_merge_takes_chips_in_the_chippers_block_order(tmp_path):
    from tests.conftest import build_dataset

    grid = GeoBox.from_bbox((0, 0, 1280, 640), crs="EPSG:32633", resolution=10)
    pixels = np.random.default_rng(0).random(grid.shape).astype("float32")
    # 16-pixel chunks and 8-pixel chips: each strip is read in several column blocks.
    scene = build_dataset({"b": (("y", "x"), pixels)}, grid).chunk({"y": 16, "x": 16})
    chips = RasterChipper(scene, size=8, overlap=2)
    outputs = []
    for batch in chips.iter_batches(1):
        for location, chip in zip(batch.locations, batch.chips, strict=True):
            outputs.append((location, chip.gs.to_numpy()))
    tops = [location.top for location, _ in outputs]
    assert tops != sorted(tops)

    merger = RasterMerger(chips, store=tmp_path / "merged")
    for location, output in outputs:
        merger.add(location, output)

    (merged,) = merger.merge()
    np.testing.assert_allclose(merged.values[0], pixels, rtol=1e-6)


def test_a_staged_merger_refuses_an_existing_store(tmp_path):
    chips = RasterChipper(build_scene(times=None), size=4)
    location, _ = next(iter(chips))
    (tmp_path / "0.zarr").mkdir()

    with pytest.raises(FileExistsError):
        RasterMerger(chips, store=tmp_path).add(location, chip_scores(location, 4))


def test_the_merged_raster_declares_its_source_extent():
    triangle = Geometry(Polygon(GRID.extent.exterior.coords[:3]), GRID.crs)
    tree = build_tree(times=None)
    tree["image"] = tree["image"].to_dataset().gs.write_extent(triangle)
    chips = RasterChipper(tree, size=4)

    (merged,) = merge_scores(chips)

    assert chips.sources[0].extent == triangle
    assert merged.gs.extent == triangle
