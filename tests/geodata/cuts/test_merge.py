"""Chip outputs merge back onto the grid of the window they were cut from."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import cuts, open_raster

BANDS = ["plus", "minus"]


def _x(grid: GeoBox) -> np.ndarray:
    """Each pixel's x on a grid."""
    return np.tile(np.asarray(grid.coords["x"].values, "float64"), (grid.shape[0], 1))


def _predict(chips: gpd.GeoDataFrame, *, drop=()) -> dict[int, np.ndarray]:
    """Give each chip two outputs equal to plus and minus each pixel's x."""
    outputs = {}
    for position, chip in chips.iterrows():
        if position in drop:
            continue
        h, w = (int(edge) for edge in chip["proj:shape"])
        grid = GeoBox((h, w), Affine(*chip["proj:transform"][:6]), chip["proj:code"])
        outputs[int(position)] = np.stack([_x(grid), -_x(grid)])
    return outputs


@pytest.mark.parametrize("taper", [None, "hann"])
def test_chip_outputs_merge_onto_the_parent_grid(items, grid, taper) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32, overlap=8)

    merged = cuts.merge(windows, chips, _predict(chips), bands=BANDS, taper=taper)

    (result,) = merged.values()
    assert result.gs.geobox == grid
    assert list(result.data_vars) == BANDS
    np.testing.assert_allclose(result["plus"].values, _x(grid))
    np.testing.assert_allclose(result["minus"].values, -_x(grid))


def test_a_missing_chip_refuses(items) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32)

    with pytest.raises(ValueError, match="no output"):
        cuts.merge(windows, chips, _predict(chips, drop={0}), bands=BANDS)


def test_outputs_naming_other_bands_refuse(items) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32)

    with pytest.raises(ValueError, match="bands"):
        cuts.merge(windows, chips, _predict(chips), bands=["only"])


def test_chip_outputs_merge_in_any_order(items, grid) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32, overlap=8)
    outputs = _predict(chips)
    shuffled = {key: outputs[key] for key in reversed(list(outputs))}

    (result,) = cuts.merge(
        windows, chips, shuffled, bands=BANDS, taper="hann"
    ).values()

    np.testing.assert_allclose(result["plus"].values, _x(grid))


def test_a_chip_larger_than_its_window_merges_back_onto_it(items, grid) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 128)

    (result,) = cuts.merge(windows, chips, _predict(chips), bands=BANDS).values()

    assert result.gs.geobox == grid
    np.testing.assert_allclose(result["plus"].values, _x(grid))


def test_each_window_merges_onto_its_own_grid(items, grid) -> None:
    windows = cuts.stacks(items, on="label")
    twice = gpd.GeoDataFrame(
        pd.concat([windows, windows], ignore_index=True),
        geometry="geometry",
        crs=windows.crs,
    )
    chips = cuts.chips(twice, 32)
    outputs = _predict(chips)

    merged = cuts.merge(twice, chips, outputs, bands=BANDS)
    only_second = cuts.merge(
        twice, chips[chips["parent"] == 1], outputs, bands=BANDS
    )

    assert sorted(merged) == [0, 1]
    for result in merged.values():
        np.testing.assert_allclose(result["plus"].values, _x(grid))
    assert list(only_second) == [1]


def _corner(chip: pd.Series, grid: GeoBox) -> tuple[int, int]:
    """Return the window pixel a chip's first pixel sits on."""
    transform = Affine(*chip["proj:transform"][:6])
    column, row = ~grid.transform * (transform.c, transform.f)
    return round(row), round(column)


def test_an_invalid_pixel_weighs_nothing(items, grid) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32, overlap=8)
    outputs = _predict(chips)
    valid = {key: np.ones(value.shape[1:], bool) for key, value in outputs.items()}
    # The first chip is invalid throughout, and its values would show if weighed.
    outputs[0][:] = 1e9
    valid[0][:] = False
    # Window pixel (10, 10) is invalid in every chip covering it.
    for key, chip in chips.iterrows():
        row, column = _corner(chip, grid)
        if 0 <= 10 - row < 32 and 0 <= 10 - column < 32:
            valid[int(key)][10 - row, 10 - column] = False

    (result,) = cuts.merge(
        windows, chips, outputs, bands=BANDS, valid=valid, taper="hann"
    ).values()

    plus = result["plus"].values
    assert np.isnan(plus[10, 10])
    known = np.isfinite(plus)
    assert known.sum() > plus.size / 2
    np.testing.assert_allclose(plus[known], _x(grid)[known])


def test_a_taper_without_overlap_refuses(items) -> None:
    # A tapered chip edge weighs nearly nothing, and no neighbour covers it.
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32)

    with pytest.raises(ValueError, match="overlap"):
        cuts.merge(windows, chips, _predict(chips), bands=BANDS, taper="hann")


def test_a_merged_window_round_trips_through_a_cog(items, grid, tmp_path) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32, overlap=8)
    (result,) = cuts.merge(windows, chips, _predict(chips), bands=BANDS).values()

    (file,) = result.gs.to_cog(f"{tmp_path}/merged.tif")
    reopened = open_raster(file.href)

    assert reopened.gs.geobox == grid
    assert list(reopened.data_vars) == BANDS
    np.testing.assert_allclose(reopened["plus"].values.squeeze(), _x(grid))
