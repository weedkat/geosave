import pandas as pd


def test_single_collection_groups_acquisitions_before_framing(items, grid):
    optical = items[items.collection == "s2"]
    samples = optical.gs.sample()
    assert len(samples) == 1
    assert samples.iloc[0].gs.geobox is None
    assert [item.id for item in samples.iloc[0].gs.items["s2"]] == optical.id.tolist()
    raster = samples.iloc[0].gs.to_xarray()["s2"].dataset
    assert raster.red.chunks is not None
    assert len(raster.gs.times) == 5
    frames = samples.gs.frame(windows={"s2": {"slots": ["0D"]}})
    assert len(frames) == 5
    chip = frames.gs.chip(25, target=grid).iloc[0]
    raster = chip.gs.to_xarray()["s2"].dataset
    assert raster.gs.geobox == chip.gs.geobox
    assert raster.red.chunks is not None
    assert len(raster.gs.times) == 1


def test_sample_span_covers_all_observations_without_selecting_a_reference(items):
    labels = items[items.collection == "label"].assign(
        datetime=pd.Timestamp("2024-03-30", tz="UTC"),
        start_datetime=None,
        end_datetime=None,
    )
    catalog = pd.concat([labels, items[items.collection == "s2"]], ignore_index=True)
    samples = catalog.gs.sample()
    assert samples.iloc[0].gs.timespan == (
        pd.Timestamp("2024-01-10"),
        pd.Timestamp("2024-07-03"),
    )
    assert len(samples.iloc[0].gs.items["s2"]) == 5
    frames = samples.gs.frame(
        windows={"s2": {"slots": ["-1M"], "reduce": "first"}}, reference="label"
    )
    assert len(frames) == 1
    assert len(frames.iloc[0].gs.items["s2"]) == 2
