"""An item table lists one window per anchor item, joined by what covers it."""

from __future__ import annotations

import geopandas as gpd
import orjson
import pandas as pd
import pytest
from shapely.affinity import translate

from geosave.geodata import cuts
from geosave.geodata.warnings import (
    DuplicateInstantsWarning,
    PartialCoverWarning,
)


def test_each_anchor_item_is_one_window_on_its_grid(items, grid) -> None:
    (window,) = cuts.stacks(items, on="label").to_dict("records")

    assert window["anchor"] == "s0/label"
    assert window["parent"] is None
    assert list(window["proj:shape"]) == [grid.shape[0], grid.shape[1]]
    assert list(window["proj:transform"]) == pytest.approx(list(grid.transform)[:6])
    sources = orjson.loads(window["sources"])
    assert sorted(sources) == ["dem", "label", "s1", "s2"]
    assert len(sources["s2"]) == 5


def test_sources_keep_absolute_hrefs_and_only_what_loading_needs(items) -> None:
    sources = orjson.loads(cuts.stacks(items, on="label").iloc[0]["sources"])

    s2 = sources["s2"][0]
    assert set(s2) == {
        "type",
        "stac_version",
        "stac_extensions",
        "id",
        "collection",
        "geometry",
        "bbox",
        "properties",
        "assets",
    }
    assert set(s2["properties"]) <= {"datetime", "start_datetime", "end_datetime"}
    assert all(asset["href"].startswith("/") for asset in s2["assets"].values())


def test_an_unknown_anchor_collection_refuses(items) -> None:
    with pytest.raises(ValueError, match="label"):
        cuts.stacks(items, on="labels")


def test_a_repeated_instant_keeps_the_first_and_warns(items) -> None:
    copy = items[items["collection"] == "s1"].head(1).copy()
    copy["id"] = "s0/s1_copy"
    doubled = gpd.GeoDataFrame(
        pd.concat([items, copy]), geometry="geometry", crs=items.crs
    )

    with pytest.warns(DuplicateInstantsWarning):
        window = cuts.stacks(doubled, on="label").iloc[0]

    assert len(orjson.loads(window["sources"])["s1"]) == 3


def test_an_item_covering_only_part_of_the_anchor_does_not_join(items) -> None:
    shifted = items.copy()
    s1 = shifted["collection"] == "s1"
    shifted.loc[s1, "geometry"] = shifted.loc[s1, "geometry"].apply(
        lambda geom: translate(geom, xoff=0.001)
    )

    with pytest.warns(PartialCoverWarning):
        window = cuts.stacks(shifted, on="label").iloc[0]

    assert "s1" not in orjson.loads(window["sources"])


def test_anchor_properties_ride_on_the_window(items) -> None:
    labelled = items.copy()
    labelled["annotator"] = None
    labelled.loc[labelled["collection"] == "label", "annotator"] = "ana"

    window = cuts.stacks(labelled, on="label").iloc[0]

    assert window["annotator"] == "ana"


def _windows_of(rows) -> list[tuple[str, list[str]]]:
    """Each window's anchor and the collections its sources hold."""
    return [
        (window["anchor"], sorted(orjson.loads(window["sources"])))
        for _, window in rows.iterrows()
    ]


def test_a_table_of_anchors_alone_lists_windows_of_the_anchor_only(items) -> None:
    anchors = items[items["collection"] == "label"]

    assert _windows_of(cuts.stacks(anchors, on="label")) == [("s0/label", ["label"])]


def test_a_reordered_table_keeps_its_own_index_out_of_the_join(items) -> None:
    reordered = items[items["collection"] != "dem"].iloc[::-1]

    assert _windows_of(cuts.stacks(reordered, on="label")) == [
        ("s0/label", ["label", "s1", "s2"])
    ]


def test_items_join_only_the_anchors_they_cover(items) -> None:
    far = items[items["collection"] == "label"].copy()
    far["id"] = "s0/label_far"
    far["geometry"] = far.geometry.apply(lambda geom: translate(geom, xoff=1.0))
    both = gpd.GeoDataFrame(
        pd.concat([items, far], ignore_index=True), geometry="geometry", crs=items.crs
    )

    assert _windows_of(cuts.stacks(both, on="label")) == [
        ("s0/label", ["dem", "label", "s1", "s2"]),
        ("s0/label_far", ["label"]),
    ]


def test_an_item_without_a_collection_refuses(items) -> None:
    unnamed = items.copy()
    unnamed.loc[unnamed["collection"] == "dem", "collection"] = None

    with pytest.raises(ValueError, match="collection"):
        cuts.stacks(unnamed, on="label")
