import numpy as np
import pytest

from geosave.geodata.warnings import DroppedRowsWarning


def _red_outputs(chips) -> dict:
    outputs = {}
    for label, row in chips.iterrows():
        red = row.gs.to_xarray(collections=["s2"])["s2"].dataset.red
        outputs[label] = red.isel(time=0).values[None].astype("float64")
    return outputs


@pytest.mark.parametrize("taper", [None, "hann"])
@pytest.mark.parametrize("overlap", [8, 0.25])
def test_merge_rebuilds_the_sample_from_overlapping_chips(items, taper, grid, overlap):
    samples = items.gs.sample()
    chips = samples.gs.chip(32, overlap=overlap, target=grid)

    merged = samples.gs.merge(chips, _red_outputs(chips), bands=["red"], taper=taper)

    expected = samples.iloc[0].gs.to_xarray()["s2"].dataset.red.isel(time=0)
    np.testing.assert_allclose(merged[0].red.values, expected.values)
    assert merged[0].gs.geobox == grid


def test_merge_leaves_pixels_no_valid_chip_covers_nan(items, grid):
    samples = items.gs.sample()
    chips = samples.gs.chip((25, 35), target=grid)
    outputs = _red_outputs(chips)
    valid = {label: np.ones((25, 35), dtype=bool) for label in chips.index}
    valid[chips.index[0]][:] = False

    merged = samples.gs.merge(chips, outputs, bands=["red"], valid=valid)

    assert np.isnan(merged[0].red.values[:25, :35]).all()
    assert not np.isnan(merged[0].red.values[25:, 35:]).any()


def test_merge_refuses_chips_of_other_samples(items, composites, grid):
    chips = items.gs.sample().gs.chip(25, target=grid)
    others = composites.gs.sample()

    with pytest.raises(ValueError, match="not cut"):
        others.gs.merge(chips, _red_outputs(chips), bands=["red"])


def test_merge_refuses_a_misshapen_output(items, grid):
    samples = items.gs.sample()
    chips = samples.gs.chip(25, target=grid)
    outputs = {label: np.zeros((2, 25, 25)) for label in chips.index}

    with pytest.raises(ValueError, match="shape"):
        samples.gs.merge(chips, outputs, bands=["red"])


def test_merge_refuses_repeated_bands(items, grid):
    samples = items.gs.sample()

    with pytest.raises(ValueError, match="unique"):
        samples.gs.merge(samples.gs.chip(25, target=grid), {}, bands=["a", "a"])


def _constant_outputs(chips):
    return {
        label: np.ones((1, *row.gs.geobox.shape)) for label, row in chips.iterrows()
    }


def test_merge_nested_chips_into_their_own_parents(items, tmp_path, grid):
    from geosave.geodata import open_vector
    from geosave.geodata.core import GeoSamples

    parents = items.gs.sample().gs.chip(32, overlap=8, target=grid)
    children = parents.gs.chip(10, target=grid)
    saved = children.gs.to_geoparquet(tmp_path / "nested.parquet")
    restored = open_vector(saved)
    restored = GeoSamples.concat([restored.iloc[::2], restored.iloc[1::2]])
    parents = parents.iloc[::-1].reset_index(drop=True)

    merged = parents.gs.merge(
        restored, _constant_outputs(restored), bands=["prediction"]
    )

    assert len(merged) == len(parents)
    for label, row in parents.iterrows():
        assert merged[label].gs.geobox.extent == row.gs.region
        np.testing.assert_allclose(merged[label].prediction.values, 1)


@pytest.mark.parametrize("change", ["crs", "scale", "fractional_offset"])
def test_merge_refuses_chips_off_the_parent_pixel_lattice(items, change, grid):
    chips = items.gs.sample().gs.chip(25, target=grid)
    if change == "crs":
        chips["proj:code"] = "EPSG:4326"
    else:
        transform = list(chips.iloc[0]["proj:transform"])
        if change == "scale":
            transform[0] *= 2
        else:
            transform[2] += transform[0] / 2
        chips.at[0, "proj:transform"] = transform

    with pytest.raises(ValueError, match="grid|lattice|CRS"):
        items.gs.sample().gs.merge(
            chips, _constant_outputs(chips), bands=["prediction"]
        )


@pytest.mark.parametrize("duplicate", ["parents", "chips"])
def test_merge_requires_unique_row_labels(items, duplicate, grid):
    import pandas as pd

    parents = items.gs.sample()
    chips = parents.gs.chip(25, target=grid)
    if duplicate == "parents":
        parents = pd.concat([parents, parents])
    else:
        chips = pd.concat([chips, chips])

    with pytest.raises(ValueError, match="unique"):
        parents.gs.merge(chips, _constant_outputs(chips), bands=["prediction"])


def test_merge_reports_a_missing_validity_mask(items, grid):
    parents = items.gs.sample()
    chips = parents.gs.chip(25, target=grid)

    with pytest.raises(ValueError, match="validity"):
        parents.gs.merge(
            chips, _constant_outputs(chips), bands=["prediction"], valid={}
        )


@pytest.mark.parametrize("chip_first", [False, True])
def test_merge_matches_frames_in_either_operation_order(composites, chip_first, grid):
    samples = composites.gs.sample()
    with pytest.warns(DroppedRowsWarning):
        parents = samples.gs.frame(
            windows={"s2": {"slots": ["-1M", "0M"]}, "s1": {"slots": ["-1M", "0M"]}}
        )
    assert len(parents) == 2

    if chip_first:
        with pytest.warns(DroppedRowsWarning):
            chips = samples.gs.chip(25, target=grid).gs.frame(
                windows={"s2": {"slots": ["-1M", "0M"]}, "s1": {"slots": ["-1M", "0M"]}}
            )
    else:
        chips = parents.gs.chip(25, target=grid)
    outputs = {}
    for label, row in chips.iterrows():
        outputs[label] = np.full((1, *row.gs.geobox.shape), row.gs.timespan[0].month)

    merged = parents.gs.merge(chips, outputs, bands=["prediction"])

    for label, row in parents.iterrows():
        np.testing.assert_allclose(
            merged[label].prediction.values, row.gs.timespan[0].month
        )


def test_merge_refuses_ambiguous_parents_even_with_unique_labels(items, grid):
    import pandas as pd

    samples = items.gs.sample()
    chips = samples.gs.chip(25, target=grid)
    parents = pd.concat([samples, samples], ignore_index=True)

    with pytest.raises(ValueError, match="unique spatial identities"):
        parents.gs.merge(chips, _constant_outputs(chips), bands=["prediction"])


def test_merge_preserves_explicit_chip_grid_after_parquet_round_trip(
    items, grid, tmp_path
):
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from geosave.geodata import open_vector

    samples = items.gs.sample()
    target = GeoBox((100, 140), grid.transform * Affine.scale(0.5), grid.crs)
    chips = samples.gs.chip((50, 70), target=target)
    restored = open_vector(chips.gs.to_geoparquet(tmp_path / "chips.parquet"))

    merged = samples.gs.merge(
        restored, _constant_outputs(restored), bands=["prediction"]
    )

    assert merged[0].gs.geobox == target
    np.testing.assert_allclose(merged[0].prediction.values, 1)
    row = restored.iloc[0]
    tree = row.gs.to_xarray(target=row.gs.geobox)
    assert tree["label"].dataset.gs.geobox == target[:50, :70]
    assert tree["s2"].dataset.gs.geobox == target[:50, :70]
    assert tree["s2"].dataset.red.chunks is not None


def test_merge_rejects_mixed_target_layouts_for_one_parent(items, grid):
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from geosave.geodata.core import GeoSamples

    samples = items.gs.sample()
    fine = GeoBox((100, 140), grid.transform * Affine.scale(0.5), grid.crs)
    chips = GeoSamples.concat(
        [
            samples.gs.chip(25, target=grid).iloc[:1],
            samples.gs.chip(25, target=fine).iloc[:1],
        ]
    ).reset_index(drop=True)

    with pytest.raises(ValueError, match="layout grid"):
        samples.gs.merge(chips, _constant_outputs(chips), bands=["prediction"])
