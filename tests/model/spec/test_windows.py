from __future__ import annotations

import pytest

from geosave.geodata.warnings import DroppedRowsWarning
from geosave.model.spec import FramesSpec, ChipsSpec


def test_a_square_size_names_both_sides():
    assert ChipsSpec(size=224).shape == (224, 224)
    assert ChipsSpec(size=[224, 256]).shape == (224, 256)


@pytest.mark.parametrize("size", [0, [224], [224, 0], [1, 2, 3]])
def test_a_size_that_names_no_tile_is_refused(size):
    with pytest.raises(ValueError):
        ChipsSpec(size=size)


def test_a_window_needs_overlapping_tiles():
    with pytest.raises(ValueError, match="overlap"):
        ChipsSpec(size=4, window="hann")


@pytest.mark.parametrize(
    "window", ["hann", "bartlett", "barthann", "bohman", "blackman", "overlap-tile"]
)
def test_taper_rejects_single_pixel_overlap_that_leaves_seams(window):
    with pytest.raises(ValueError, match="at least 2"):
        ChipsSpec(size=8, overlap=1, window=window)


@pytest.mark.parametrize(
    "window", ["boxcar", "hamming", "triang", "parzen", "nuttall", "blackmanharris"]
)
def test_positive_edge_windows_allow_single_pixel_overlap(window):
    assert ChipsSpec(size=8, overlap=1, window=window).overlap == 1


def test_a_chip_spec_cannot_wrap():
    with pytest.raises(ValueError, match="wrap"):
        ChipsSpec(size=8, overlap=2, mode="wrap")


def test_frames_cut_as_the_spec_declares(composites):
    samples = composites.gs.sample()

    with pytest.warns(DroppedRowsWarning):
        cut = FramesSpec(
            windows={"s2": {"slots": ["-1M", "0M"]}, "s1": {"slots": ["-1M", "0M"]}},
            stride=1,
        ).cut(samples)
    with pytest.warns(DroppedRowsWarning):
        expected = samples.gs.frame(
            windows={"s2": {"slots": ["-1M", "0M"]}, "s1": {"slots": ["-1M", "0M"]}},
            stride=1,
        )

    assert cut.equals(expected)
    assert len(cut) == 4


def test_chips_cut_as_the_spec_declares(items, grid):
    samples = items.gs.sample()

    cut = ChipsSpec(size=[25, 35], overlap=4, mode="edge").cut(samples, target=grid)

    assert cut.equals(
        samples.gs.chip((25, 35), overlap=4, pad_mode="edge", target=grid)
    )
