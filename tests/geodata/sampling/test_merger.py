"""A RasterMerger places predictions back by chip index, through its RasterChipper."""

import numpy as np
import pytest

from geosave.geodata.sampling import RasterMerger, RasterChipper, create_chip_weights
from tests.geodata.sampling.conftest import GRID, build_tree
from tests.geodata.sampling.test_chipper import SMALL_GRID


def read_last_date(chip):
    """Stand in for a model: return the bands of the chip's last date."""
    return chip["image"].to_dataset().isel(time=-1).gs.to_numpy()


@pytest.mark.parametrize("mode", ["shift", "drop", "constant", "edge", "reflect", "wrap"])
@pytest.mark.parametrize("window", ["boxcar", "hann", "triang", "blackmanharris"])
def test_shuffled_chips_merge_back_to_each_raster(mode, window):
    # 8 x 8 and 6 x 8 rasters, so "drop" still covers every pixel at stride 2.
    even_grid = GRID[0:8, 0:8]
    rasters = build_tree(even_grid).gs.sample("MS", size=2) + [build_tree(SMALL_GRID)]
    chipper = RasterChipper(rasters, size=4, stride=2, mode=mode)
    merger = RasterMerger(chipper.layout, window=window)

    for index in np.random.default_rng(0).permutation(len(chipper)):
        merger.add(int(index), read_last_date(chipper[int(index)]))

    for raster_index, raster in enumerate(rasters):
        merged = merger.merge(raster_index)
        expected = raster["image"].to_dataset().isel(time=-1).gs.to_numpy()
        assert merged.dims == ("band", "y", "x")
        assert merged.dtype == "float32"
        assert merged.odc.geobox == raster.gs.geobox
        np.testing.assert_allclose(merged.values, expected, rtol=1e-5)


def test_a_taper_never_zeroes_a_pixel_only_rims_cover():
    assert create_chip_weights("boxcar", 4).tolist() == [[1.0] * 4] * 4
    weights = create_chip_weights("hann", 4)
    assert weights.min() > 0
    assert weights[1, 1] == pytest.approx(weights.max())
    assert weights[0, 0] == pytest.approx(weights.min())

    chipper = RasterChipper([build_tree(times=None)], size=4, stride=2)
    merger = RasterMerger(chipper.layout, window="hann")
    for index in range(len(chipper)):
        merger.add(index, np.full((1, 4, 4), 7.0))

    np.testing.assert_allclose(merger.merge(0).values, 7.0)


def test_overlaps_average_and_uncovered_pixels_are_nan():
    chipper = RasterChipper([build_tree(times=None)], size=4, stride=2)
    merger = RasterMerger(chipper.layout)
    merger.add(0, np.full((1, 4, 4), 2.0))
    merger.add(1, np.full((1, 4, 4), 4.0))

    merged = merger.merge(0)

    assert merged.values[0, 0, 0] == 2.0
    assert merged.values[0, 0, 3] == 3.0
    assert merged.values[0, 0, 5] == 4.0
    assert np.isnan(merged.values[0, 9, 9])


def test_merger_refuses_predictions_it_cannot_place():
    chipper = RasterChipper(
        [build_tree(times=None), build_tree(SMALL_GRID, times=None)], size=4, mode="shift"
    )
    merger = RasterMerger(chipper.layout)

    with pytest.raises(ValueError, match="no prediction was added for raster 1"):
        merger.merge(1)
    with pytest.raises(ValueError, match=r"\(band, 4, 4\)"):
        merger.add(0, np.zeros((4, 4)))
    with pytest.raises(IndexError):
        merger.add(len(chipper), np.zeros((1, 4, 4)))

    merger.add(0, np.zeros((2, 4, 4)))
    with pytest.raises(ValueError, match="bands"):
        merger.add(1, np.zeros((3, 4, 4)))
    # Another raster may have another band count.
    merger.add(9, np.zeros((3, 4, 4)))
    assert merger.merge(1).sizes["band"] == 3
    with pytest.raises(ValueError):
        RasterMerger(chipper.layout, window="mirror")


def test_merged_logits_decode_with_native_xarray():
    chipper = RasterChipper([build_tree(times=None)], size=5)
    merger = RasterMerger(chipper.layout, window="hann")
    for index in range(len(chipper)):
        merger.add(index, np.stack([np.zeros((5, 5)), np.ones((5, 5))]))

    classes = merger.merge(0).argmax("band")

    assert classes.odc.geobox == GRID
    assert int(classes.min()) == 1


def test_a_raster_completes_with_its_last_chip_and_is_freed_on_discard():
    chipper = RasterChipper(
        [build_tree(times=None), build_tree(times=None)], size=4, stride=3
    )
    merger = RasterMerger(chipper.layout)
    first_raster = chipper.layout.count_chips(0)

    for index in range(first_raster - 1):
        merger.add(index, np.ones((1, 4, 4), dtype="float32"))
    assert not merger.is_complete(0)

    merger.add(first_raster - 1, np.ones((1, 4, 4), dtype="float32"))
    assert merger.is_complete(0)
    assert not merger.is_complete(1)
    assert float(merger.merge(0).min()) == 1.0

    merger.discard(0)
    assert not merger.is_complete(0)
    with pytest.raises(ValueError, match="no prediction was added"):
        merger.merge(0)
