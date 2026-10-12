from __future__ import annotations

import geopandas as gpd
import orjson
import pytest
from shapely.geometry import box

from geosave.geodata import cuts
from geosave.model.spec import FramesSpec, ChipsSpec


def _windows() -> gpd.GeoDataFrame:
    """One 50 by 70 window, as `cuts.stacks` lists it: s2 on four dates, a static label."""
    dates = ["2024-03-01", "2024-03-11", "2024-03-21", "2024-03-31"]
    sources = {
        "label": [{"id": "label", "properties": {}}],
        "s2": [
            {"id": f"s2_{day}", "properties": {"datetime": f"{day}T00:00:00Z"}}
            for day in dates
        ],
    }
    return gpd.GeoDataFrame(
        {
            "anchor": ["label"],
            "parent": [None],
            "sources": [orjson.dumps(sources).decode()],
            "proj:code": ["EPSG:32633"],
            "proj:transform": [[10.0, 0.0, 300000.0, 0.0, -10.0, 5000500.0]],
            "proj:shape": [[50, 70]],
        },
        geometry=[box(12.0, 45.0, 12.01, 45.01)],
        crs="EPSG:4326",
    )


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


def test_frames_cut_as_the_spec_declares():
    spec = FramesSpec(length=2, stride=1, tolerance="1D", mode="drop")

    cut = spec.cut(_windows())

    expected = cuts.frames(_windows(), 2, tolerance="1D", stride=1, mode="drop")
    assert cut.equals(expected)
    assert cut["parent"].tolist() == [0, 0, 0]


def test_chips_cut_and_lay_out_as_the_spec_declares():
    spec = ChipsSpec(size=[25, 35], overlap=4, mode="edge")

    cut = spec.cut(_windows())
    tiler, halo = spec.layout((50, 70))

    expected = cuts.chips(_windows(), (25, 35), overlap=4, mode="edge")
    assert cut.equals(expected)
    assert len(tiler) == len(cut)
    assert halo == cuts.layout((50, 70), (25, 35), overlap=4, mode="edge")[1]
