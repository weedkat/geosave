"""Sampling an item table into windows of paired time steps, and opening a sample."""

import pytest

from geosave.geodata.core.vector.catalog import GeoCatalog
from geosave.geodata.warnings import DroppedRowsWarning


def read_samples(samples) -> list[list[str]]:
    """Return each sample's Items, in sample order."""
    sample_ids = []
    for rows in samples:
        sample_ids.append(read_ids(rows["id"]))
    return sample_ids


def read_ids(item_ids) -> list[str]:
    """Return Item ids without the tile prefix, as `collection date`."""
    ids = []
    for item_id in item_ids:
        collection, _, date = item_id.split("_")
        ids.append(f"{collection} {date}")
    return ids


def test_a_sample_is_the_catalog_rows_at_its_indices(tiles):
    with pytest.warns(DroppedRowsWarning):
        samples = tiles.gs.sample(pair=["s2", "s1"], tolerance="2D", window=2)

    rows = samples[0]
    assert type(rows.gs) is GeoCatalog
    assert rows["collection"].tolist() == ["s2", "s1", "s2", "s1", "dem"]
    assert rows.equals(tiles.iloc[samples.indices[0]])
    assert len(samples) == len(samples.indices)


def test_a_sample_footprint_is_where_its_items_overlap(tiles):
    footprints = tiles.gs.sample().footprints

    assert footprints.geometry.iloc[0].equals(tiles.geometry.iloc[0])
    assert not footprints.geometry.is_empty.any()
    assert footprints.crs == tiles.crs


def test_footprints_without_pair_are_one_sample_each(tiles):
    samples = tiles.gs.sample()

    assert samples.footprints["items"].tolist() == [10, 1]


def test_a_step_pairs_each_item_with_its_nearest_partner(tiles):
    with pytest.warns(DroppedRowsWarning, match="3 s2 Items have no partner"):
        samples = tiles.gs.sample(pair=["s2", "s1"], tolerance="2D")

    assert read_samples(samples) == [
        ["s2 20240105", "s1 20240106", "dem 20240101"],
        ["s2 20240204", "s1 20240203", "dem 20240101"],
        ["s2 20240308", "s1 20240309", "dem 20240101"],
    ]


def test_a_dated_collection_outside_pair_joins_no_sample(tiles):
    samples = tiles.gs.sample(pair=["s2"])

    for rows in samples:
        assert "s1" not in set(rows["collection"])
    assert read_samples(samples)[0] == ["s2 20240105", "dem 20240101"]


def test_windows_roll_over_steps_like_rolling(tiles):
    samples = tiles.gs.sample(pair=["s2"], window=3, step=2)

    assert read_samples(samples) == [
        ["s2 20240105", "s2 20240120", "s2 20240204", "dem 20240101"],
        ["s2 20240204", "s2 20240308", "s2 20240407", "dem 20240101"],
    ]


def test_freq_slots_are_resample_buckets_filled_by_keep(tiles):
    last = tiles.gs.sample(pair=["s2"], freq="MS")
    first = tiles.gs.sample(pair=["s2"], freq="MS", keep="first")

    assert read_samples(last)[0] == ["s2 20240120", "dem 20240101"]
    assert read_samples(first)[0] == ["s2 20240105", "dem 20240101"]


def test_an_empty_bucket_breaks_a_window(tiles):
    without_february = tiles[~tiles["id"].str.endswith("20240204")]

    with pytest.warns(DroppedRowsWarning, match="windows hold an empty slot"):
        samples = without_february.gs.sample(pair=["s2"], window=2, freq="MS")

    assert read_samples(samples) == [["s2 20240308", "s2 20240407", "dem 20240101"]]


def test_a_monthly_window_of_pairs(tiles):
    with pytest.warns(DroppedRowsWarning):
        samples = tiles.gs.sample(
            pair=["s2", "s1"], tolerance="2D", window=2, freq="MS"
        )

    assert read_samples(samples) == [
        ["s2 20240105", "s1 20240106", "s2 20240204", "s1 20240203", "dem 20240101"],
        ["s2 20240204", "s1 20240203", "s2 20240308", "s1 20240309", "dem 20240101"],
    ]


def test_a_footprint_without_a_window_gives_no_sample(tiles):
    with pytest.warns(DroppedRowsWarning, match="1 footprints hold fewer than 2"):
        samples = tiles.gs.sample(pair=["s2"], window=2)

    # Tile B holds one scene, so only tile A has windows of two.
    for rows in samples:
        for item_id in rows["id"]:
            assert item_id.split("_")[1] == "300000"


def test_window_options_need_pair(tiles):
    with pytest.raises(ValueError, match="name them with pair"):
        tiles.gs.sample(window=3)


def test_pair_names_existing_collections(tiles):
    with pytest.raises(KeyError, match="l8"):
        tiles.gs.sample(pair=["s2", "l8"])


@pytest.mark.parametrize("overlap", [0, 1.5])
def test_overlap_is_a_fraction(tiles, overlap):
    with pytest.raises(ValueError, match="fraction"):
        tiles.gs.sample(overlap=overlap)


def test_no_complete_window_raises(tiles):
    with pytest.raises(ValueError, match="no footprint holds 9"):
        tiles.gs.sample(pair=["s2"], window=9)


def test_sample_an_empty_catalog(catalog):
    samples = catalog.iloc[:0].gs.sample()

    assert len(samples) == 0


def test_a_sample_opens_lazily_one_group_per_collection(tiles):
    with pytest.warns(DroppedRowsWarning):
        samples = tiles.gs.sample(pair=["s2", "s1"], tolerance="2D", window=2)

    tree = samples[0].gs.to_datatree()

    assert tree.gs.groups == ("s2", "s1", "dem")
    assert tree["s2"].gs.to_dataset().sizes["time"] == 2
    assert tree["s1"].gs.to_dataset().sizes["time"] == 2
    assert tree["s1"].gs.to_dataset().gs.resolution.x == 20
    assert "time" not in tree["dem"].gs.to_dataset().dims
    for raster in tree.gs.rasters.values():
        for variable in raster.data_vars.values():
            assert variable.chunks is not None


def test_a_sample_holds_the_catalog_it_grouped(tiles):
    samples = tiles.gs.sample(pair=["s2"], window=2)
    ids = samples[0]["id"].tolist()

    filtered = tiles[tiles["collection"] == "dem"]

    assert len(filtered) == 1
    assert samples[0]["id"].tolist() == ids


def test_samples_of_a_reindexed_catalog_select_by_position(tiles):
    shuffled = tiles.iloc[::-1].set_index("id", drop=False)

    samples = shuffled.gs.sample(pair=["s2"], window=2)

    assert read_samples(samples)[0] == ["s2 20240105", "s2 20240120", "dem 20240101"]
