"""Chip outputs saved as COGs merge back onto the grid they were cut from."""

from __future__ import annotations

from datetime import UTC, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import cuts, raster, stac


def _predict(chips, tmp_path, *, drop=()):
    """Save, per chip, two outputs equal to plus and minus each pixel's x."""
    files, positions = [], []
    for position, (_, chip) in enumerate(chips.iterrows()):
        if position in drop:
            continue
        h, w = (int(edge) for edge in chip["proj:shape"])
        grid = GeoBox((h, w), Affine(*chip["proj:transform"][:6]), chip["proj:code"])
        xs = np.tile(np.asarray(grid.coords["x"].values, "float32"), (h, 1))
        out = raster(
            {"logits_0": (("y", "x"), xs), "logits_1": (("y", "x"), -xs)}, grid
        )
        (file,) = out.gs.to_cog(f"{tmp_path}/chip-{position}.tif")
        files.append(file)
        positions.append(position)
    rows = stac.create_items(
        files,
        collection=stac.create_collection("chips", description="Chip outputs"),
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
    )
    rows["chip"] = positions
    return rows


@pytest.mark.parametrize("taper", [None, "hann"])
def test_saved_chips_merge_onto_the_parent_grid(items, grid, tmp_path, taper) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32, overlap=8)

    merged = cuts.merge(windows, chips, _predict(chips, tmp_path), taper=taper)

    (result,) = merged.values()
    assert result.gs.geobox == grid
    values = result["logits"].values
    expected = np.tile(
        np.asarray(grid.coords["x"].values, "float32"), (grid.shape[0], 1)
    )
    np.testing.assert_allclose(values[0], expected, rtol=1e-5)
    np.testing.assert_allclose(values[1], -expected, rtol=1e-5)


def test_a_missing_chip_refuses(items, tmp_path) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32)

    with pytest.raises(ValueError, match="no output"):
        cuts.merge(windows, chips, _predict(chips, tmp_path, drop={0}))


def _x(grid) -> np.ndarray:
    """Each pixel's x, as `_predict` saves it on the first band."""
    return np.tile(np.asarray(grid.coords["x"].values, "float32"), (grid.shape[0], 1))


def test_chip_outputs_merge_in_any_order(items, grid, tmp_path) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32, overlap=8)
    shuffled = _predict(chips, tmp_path).sample(frac=1, random_state=1)

    (result,) = cuts.merge(windows, chips, shuffled, taper="hann").values()

    np.testing.assert_allclose(result["logits"].values[0], _x(grid), rtol=1e-5)


def test_a_chip_larger_than_its_window_merges_back_onto_it(
    items, grid, tmp_path
) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 128)

    (result,) = cuts.merge(windows, chips, _predict(chips, tmp_path)).values()

    assert result.gs.geobox == grid
    np.testing.assert_allclose(result["logits"].values[0], _x(grid), rtol=1e-5)


def test_each_window_merges_onto_its_own_grid(items, grid, tmp_path) -> None:
    windows = cuts.stacks(items, on="label")
    twice = gpd.GeoDataFrame(
        pd.concat([windows, windows], ignore_index=True),
        geometry="geometry",
        crs=windows.crs,
    )
    chips = cuts.chips(twice, 32)
    # Chips of identical windows share grids, so each window's outputs are one batch.
    batches = []
    for parent in (0, 1):
        members = chips[chips["parent"] == parent]
        batch = _predict(members, tmp_path / str(parent))
        batch["chip"] = members.index.tolist()
        batches.append(batch)
    predictions = gpd.GeoDataFrame(
        pd.concat(batches, ignore_index=True), geometry="geometry", crs=batches[0].crs
    )

    merged = cuts.merge(twice, chips, predictions)
    only_second = cuts.merge(twice, chips[chips["parent"] == 1], predictions)

    assert sorted(merged) == [0, 1]
    for result in merged.values():
        np.testing.assert_allclose(result["logits"].values[0], _x(grid), rtol=1e-5)
    assert list(only_second) == [1]


def test_a_taper_without_overlap_refuses(items, tmp_path) -> None:
    # A tapered chip edge weighs nearly nothing, and no neighbour covers it.
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32)

    with pytest.raises(ValueError, match="overlap"):
        cuts.merge(windows, chips, _predict(chips, tmp_path), taper="hann")
