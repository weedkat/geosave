import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geosave.geodata.core import GeoSample


def test_a_sample_row_reads_through_geosample(items):
    assert isinstance(items.gs.sample().iloc[0].gs, GeoSample)


def test_a_sample_reads_every_collection_on_its_native_grid_lazily(items):
    tree = items.gs.sample().iloc[0].gs.to_xarray()

    assert isinstance(tree, xr.DataTree)
    assert set(tree.children) == {"label", "dem", "s1", "s2"}
    assert tree["s2"].dataset.sizes == {"time": 5, "y": 50, "x": 70}
    # Radar keeps 20 m pixels, cropped to the sample's ground.
    assert tree["s1"].dataset.sizes == {"time": 5, "y": 25, "x": 35}
    for name in tree.children:
        for variable in tree[name].dataset.data_vars.values():
            assert isinstance(variable.data, da.Array)


def test_a_single_collection_sample_still_opens_as_a_datatree(items):
    optical = items[items["collection"] == "s2"].gs.sample()

    tree = optical.iloc[0].gs.to_xarray()

    assert isinstance(tree, xr.DataTree)
    assert tree.gs.groups == ("s2",)


def test_a_sample_keeps_dtypes_and_collection_acdd(items):
    tree = items.gs.sample().iloc[0].gs.to_xarray()

    assert tree["label"].dataset.label.dtype == "uint8"
    assert tree["dem"].dataset.height.dtype == "float32"
    assert tree["s2"].dataset.attrs["id"] == "s2"


def test_a_sample_aligns_only_onto_an_explicit_target(items, grid):
    row = items.gs.sample().iloc[0]

    tree = row.gs.to_xarray(target=grid, resampling="bilinear")

    assert tree["s1"].dataset.gs.geobox == grid
    assert isinstance(tree["s1"].dataset.vv.data, da.Array)


def test_alignment_options_need_a_target(items):
    row = items.gs.sample().iloc[0]

    with pytest.raises(ValueError, match="explicit target"):
        row.gs.to_xarray(resampling="bilinear")


@pytest.mark.parametrize("target", ["label", "s1"])
def test_a_collection_can_supply_the_alignment_grid(items, target, grid):
    row = items.gs.sample().gs.chip(32, overlap=8, target=grid).iloc[0]
    native = row.gs.to_xarray()

    aligned = row.gs.to_xarray(target=target)

    grid = native[target].dataset.gs.geobox
    for collection in aligned.children:
        assert aligned[collection].dataset.gs.geobox == grid
        for variable in aligned[collection].dataset.data_vars.values():
            assert isinstance(variable.data, da.Array)


def test_a_target_collection_need_not_be_in_the_output(items, grid):
    row = items.gs.sample().iloc[0]

    aligned = row.gs.to_xarray(collections=["s1"], target="label")

    assert aligned.gs.groups == ("s1",)
    assert aligned["s1"].dataset.gs.geobox == grid


def test_a_frame_reads_only_its_time_steps(items):
    samples = items.gs.sample()
    with pytest.warns(Warning):
        frames = samples.gs.frame(
            windows={
                "s2": {"slots": ["-1M"], "reduce": "first"},
                "s1": {"slots": ["-1M"], "reduce": "first"},
            },
            stride=1,
        )
    frame = frames[frames.reference_time == pd.Timestamp("2024-04-04")].iloc[0]
    opened = frame.gs.to_xarray(collections=["s2", "s1"])
    assert opened["s2"].dataset.sizes["time"] == 1
    assert opened["s1"].dataset.sizes["time"] == 1
    expected = items[items.id == "a/s2_20240308T000000"].iloc[0].gs.to_xarray()
    np.testing.assert_allclose(opened["s2"].dataset.red.isel(time=0), expected.red)


def test_an_inner_chip_reads_the_samples_pixels(items, grid):
    samples = items.gs.sample()
    chip = samples.gs.chip((25, 35), target=grid).iloc[3]

    whole = samples.iloc[0].gs.to_xarray()["s2"].dataset.red
    part = chip.gs.to_xarray()["s2"].dataset.red

    np.testing.assert_array_equal(part.values, whole.values[:, 25:, 35:])


def test_an_edge_chip_pads_each_collection_in_its_own_pixels(items, grid):
    chip = items.gs.sample().gs.chip(32, overlap=8, target=grid).iloc[0]

    native = chip.gs.to_xarray()
    aligned = chip.gs.to_xarray(target=chip.gs.anchor.geobox)

    assert native["s2"].dataset.gs.geobox == chip.gs.anchor.geobox
    assert native["s1"].dataset.sizes["y"] == 16
    assert aligned["s1"].dataset.gs.geobox == chip.gs.anchor.geobox
    assert aligned["label"].dataset.label.dtype == "uint8"


def test_a_sample_selects_collections_in_order(items):
    row = items.gs.sample().iloc[0]

    assert row.gs.to_xarray(collections=["dem", "s2"]).gs.groups == ("dem", "s2")
    with pytest.raises(KeyError):
        row.gs.to_xarray(collections=["landsat"])
    with pytest.raises(TypeError):
        row.gs.to_xarray(collections="s2")


def test_chip_padding_uses_native_ground_extent_across_crs(
    items, grid, tmp_path, monkeypatch
):
    from geosave.geodata import raster
    from geosave.geodata.core import GeoCatalog
    from geosave.geodata.transform import extent
    from tests.geodata.conftest import item_table

    native_grid = grid.buffered(200).to_crs("EPSG:4326", resolution=0.0001)
    native = raster(
        {"value": (("y", "x"), np.ones(native_grid.shape, dtype="uint16"))}, native_grid
    )
    geographic_items = item_table(
        native.gs.to_cog(str(tmp_path / "native.tif")),
        collection="geographic",
        datetime=pd.Timestamp("2024-03-08", tz="UTC").to_pydatetime(),
    )
    samples = GeoCatalog.concat([items, geographic_items]).gs.sample()
    samples = samples[samples["collections"].map(lambda names: "geographic" in names)]
    chip = samples.gs.chip(32, overlap=8, target=grid).iloc[0]
    pad = extent.pad

    def bounded_pad(data, widths, **options):
        # Catch the metres/degrees regression before it allocates huge arrays.
        assert max(max(pair) for pair in widths) < 64
        return pad(data, widths, **options)

    monkeypatch.setattr(extent, "pad", bounded_pad)
    result = chip.gs.to_xarray(collections=["geographic"])["geographic"].dataset
    expected = native_grid.enclosing(chip.gs.anchor.geobox.extent)
    assert result.gs.geobox.shape == expected.shape
    assert result.gs.geobox.crs == expected.crs
    assert result.gs.geobox.transform.almost_equals(expected.transform, precision=1e-10)
    assert isinstance(result.value.data, da.Array)
    np.testing.assert_array_equal(result.value.compute(), 1)


def test_sample_properties_describe_the_anchor_and_padding(items, grid):
    row = items.gs.sample().gs.chip(32, overlap=8, target=grid).iloc[0]
    assert row.gs.geobox == row.gs.anchor.geobox
    assert row.gs.timespan == row.gs.anchor.timespan
    assert row.gs.pad_widths == [(12, 0), (12, 0)]
    assert row.gs.region == row.gs.geobox[12:, 12:].extent


def test_sample_timespan_normalizes_offsets_and_handles_missing_bounds(items):
    row = items.gs.sample().iloc[0].copy()
    row["start_datetime"] = "2024-03-01T07:00:00+07:00"
    row["end_datetime"] = "2024-03-02T00:00:00Z"
    assert row.gs.timespan == (pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-02"))
    row["end_datetime"] = None
    assert row.gs.timespan is None
