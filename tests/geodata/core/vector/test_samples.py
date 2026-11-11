"""Grouping source rows and explicitly preparing their native raster data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temporal_catalog():
    from datetime import UTC, datetime

    import pystac
    from shapely.geometry import box, mapping

    from geosave.geodata.stac.catalog import create_catalog
    from geosave.geodata.stac.item import TIMELESS

    footprint = box(0, 0, 1, 1)
    entries = []
    for name, collection, instant, properties in [
        ("january", "optical", datetime(2025, 1, 15, tzinfo=UTC), {}),
        ("february", "optical", datetime(2025, 2, 15, tzinfo=UTC), {}),
        (
            "composite",
            "composite",
            None,
            {
                "start_datetime": "2025-01-01T00:00:00Z",
                "end_datetime": "2025-01-31T23:59:59Z",
            },
        ),
        (
            "dem",
            "dem",
            None,
            {
                "start_datetime": TIMELESS[0].isoformat(),
                "end_datetime": TIMELESS[1].isoformat(),
            },
        ),
    ]:
        entry = pystac.Item(
            name,
            mapping(footprint),
            list(footprint.bounds),
            instant,
            properties,
            collection=collection,
        )
        entry.add_asset("data", pystac.Asset(f"/not-opened/{name}.tif", roles=["data"]))
        entries.append(entry)
    return create_catalog(entries)


def test_static_collections_are_explicit_and_do_not_define_dates(temporal_catalog):
    samples = temporal_catalog.gs.sample(
        collections=["dem", "optical"], freq="MS", window=2
    )
    assert set(samples[0]["id"]) == {"dem", "january", "february"}
    assert samples[0]["id"].tolist().count("dem") == 1
    optical = temporal_catalog.gs.sample(collections=["optical"], freq="MS")
    assert set(optical[0]["collection"]) == {"optical"}


def test_static_only_sample_has_no_temporal_axis(temporal_catalog):
    samples = temporal_catalog.gs.sample(collections=["dem"])
    assert samples.indices == [[3]]
    assert len(samples.times[0]) == 0


def test_spatial_only_samples_keep_source_membership(tiles):
    samples = tiles.gs.sample()
    assert samples.footprints["items"].tolist() == [10, 1]
    assert samples[0].equals(tiles.iloc[samples.indices[0]])
    assert samples.footprints.crs == tiles.crs


def test_exact_time_windows_and_stride(tiles):
    samples = tiles.gs.sample(collections=["s2"], window=3, step=2)
    dates = []
    for rows in samples:
        dates.append(pd.to_datetime(rows["datetime"]).dt.strftime("%m%d").tolist())
    assert dates == [["0105", "0120", "0204"], ["0204", "0308", "0407"]]


def test_reindexed_catalog_uses_original_positions(tiles):
    shuffled = tiles.iloc[::-1].set_index("id", drop=False)
    samples = shuffled.gs.sample(collections=["s2"], window=2)
    assert samples[0].equals(shuffled.iloc[samples.indices[0]])
    assert samples[0]["id"].str.endswith("20240105").iloc[0]


def test_empty_and_too_short_samples(tiles):
    assert len(tiles.iloc[:0].gs.sample()) == 0
    assert len(tiles.gs.sample(collections=["s2"], window=99)) == 0


def test_grouped_rows_load_raw_dates_and_prepare_explicitly(tiles):

    samples = tiles.gs.sample(collections=["s2", "s1", "dem"], freq="MS", window=2)
    rows = samples[0]
    assert rows.equals(tiles.iloc[samples.indices[0]])
    with rows.gs.to_datatree() as tree:
        optical = tree["s2"].gs.to_dataset()
        radar = tree["s1"].gs.to_dataset()
        assert optical.sizes["time"] == 3
        assert radar.sizes["time"] == 2
        assert radar.gs.resolution.x == 20
        assert "time" not in tree["dem"].gs.to_dataset().dims
        assert pd.DatetimeIndex(optical.time.values).day.tolist() == [5, 20, 4]
        monthly = optical.gs.mask_and_scale().gs.composite("MS", "mean")
        assert monthly.red.chunks is not None
        assert monthly.sizes["time"] == 2
        expected = optical.red.isel(time=slice(0, 2)).mean("time")
        np.testing.assert_allclose(
            monthly.red.isel(time=0).compute(), expected.compute()
        )


def test_missing_slots_do_not_modify_selected_rows_or_loaded_dates(tiles):

    rows = tiles.loc[~tiles["id"].str.endswith("20240204")]
    samples = rows.gs.sample(collections=["s2"], freq="MS", window=3, missing="keep")
    assert samples.slots[0].iloc[1]["indices"] == []
    with samples[0].gs.to_datatree() as tree:
        observations = tree["s2"].gs.to_dataset()
        assert 2 not in pd.DatetimeIndex(observations.time.values).month
        monthly = observations.gs.mask_and_scale().gs.composite("MS", "mean")
        assert monthly.red.isel(time=1).isnull().all().compute().item()
        filled = monthly.chunk(time=-1).interpolate_na(dim="time", method="linear")
        assert filled.red.chunks is not None
        assert filled.red.isel(time=1).notnull().all().compute().item()


def test_selected_collection_metadata_describes_all_rows(tiles):
    from geosave.geodata.attrs import StacMetadata
    from geosave.geodata.stac.catalog import create_catalog

    entries = tiles.gs.to_items()
    for position, entry in enumerate(entries):
        entry.properties["platform"] = f"platform-{position}"
    rows = create_catalog(entries)
    samples = rows.gs.sample(collections=["s2"], freq="MS", window=2)
    with samples[0].gs.to_datatree() as tree:
        metadata = StacMetadata.from_attrs(tree["s2"].attrs)
        assert metadata is not None
        assert "platform" not in tree["s2"].to_dataset().gs.stac_common_properties


def test_wholly_missing_collection_is_only_grouping_metadata(tiles):
    samples = tiles.gs.sample(collections=["s2", "s1"], freq="MS", missing="keep")
    position = len(samples) - 1
    missing = samples.slots[position].loc[
        samples.slots[position]["collection"] == "s1", "indices"
    ]
    assert missing.iloc[0] == []
    with samples[position].gs.to_datatree() as tree:
        assert "s1" not in tree.children
