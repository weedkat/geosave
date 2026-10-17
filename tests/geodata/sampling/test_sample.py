"""Spatial grouping retains native observations and never chooses a grid."""

import pandas as pd
import pytest
import shapely

from geosave.geodata.core import GeoSamples


def test_sample_groups_all_acquisitions_at_one_footprint(items):
    samples = items.gs.sample()
    assert isinstance(samples.gs, GeoSamples)
    assert list(samples.columns) == list(GeoSamples.COLUMNS)
    assert len(samples) == 1
    row = samples.iloc[0]
    assert set(row.gs.items) == {"label", "dem", "s1", "s2"}
    assert len(row.gs.items["s2"]) == 5
    assert len(row.gs.items["s1"]) == 5
    assert row.gs.geobox is None
    assert row.gs.anchor is None
    assert "reference" not in samples


def test_sample_keeps_every_date_and_native_grid(items, grid, wide_grid):
    row = items.gs.sample().iloc[0]
    tree = row.gs.to_xarray()
    assert tree["s2"].dataset.sizes["time"] == 5
    assert tree["s1"].dataset.sizes["time"] == 5
    assert tree["s2"].dataset.gs.geobox == grid
    assert tree["s1"].dataset.gs.geobox == wide_grid


def test_sample_keeps_different_and_containing_footprints_separate(items):
    changed = items.copy()
    radar = changed.collection == "s1"
    changed.loc[radar, "geometry"] = changed.loc[radar, "geometry"].buffer(0.001)
    samples = changed.gs.sample()
    assert len(samples) == 2
    assert set(samples.iloc[0].gs.items) == {"label", "dem", "s2"}
    assert set(samples.iloc[1].gs.items) == {"s1"}


def test_sample_keeps_partial_overlaps_separate(items):
    shifted = items.copy()
    dem = shifted.collection == "dem"
    shifted.loc[dem, "geometry"] = shifted.loc[dem, "geometry"].translate(0.001)
    samples = shifted.gs.sample()
    assert len(samples) == 2
    assert sorted(len(row.gs.items) for _, row in samples.iterrows()) == [1, 3]


def test_sample_keeps_distinct_items_at_the_same_time(items):
    label = items[items.collection == "label"]
    repeated = pd.concat([items, label.assign(id="label-copy")], ignore_index=True)
    samples = repeated.gs.sample()
    assert len(samples) == 1
    assert [entry.id for entry in samples.iloc[0].gs.items["label"]] == [
        "a/label",
        "label-copy",
    ]


def test_sample_normalizes_vertex_order(items):
    changed = items.copy()
    changed.loc[0, "geometry"] = shapely.reverse(changed.loc[0, "geometry"])
    assert len(changed.gs.sample()) == 1


def test_sample_carries_collection_definitions(items):
    assert sorted(items.gs.sample().gs.collections) == ["dem", "label", "s1", "s2"]


def test_sample_refuses_items_without_a_collection(items):
    unnamed = items.copy()
    unnamed.loc[0, "collection"] = None
    with pytest.raises(ValueError, match="collection"):
        unnamed.gs.sample()


def test_sample_opens_no_file(items, monkeypatch):
    import geosave.geodata.io.raster.stac as reader

    def refuse(*args, **kwargs):
        raise AssertionError("sample opened a file")

    monkeypatch.setattr(reader, "open_raster", refuse)
    assert len(items.gs.sample()) == 1


def test_sample_uses_the_active_geometry_column(items):
    renamed = items.rename_geometry("footprint")
    assert renamed.gs.sample().iloc[0].gs.region == items.gs.sample().iloc[0].gs.region


def test_sample_does_not_require_grid_metadata(items):
    from geosave.geodata import stac

    entries = items.gs.to_items()
    for entry in entries:
        entry.properties = {
            key: value
            for key, value in entry.properties.items()
            if not key.startswith("proj:")
        }
        for asset in entry.assets.values():
            asset.extra_fields = {
                key: value
                for key, value in asset.extra_fields.items()
                if not key.startswith("proj:")
            }
    samples = stac.create_catalog(entries).gs.sample()
    assert len(samples) == 1
    assert len(samples.iloc[0].gs.items["s2"]) == 5


def test_sample_ignores_redundant_boundary_vertices(items):
    changed = items.copy()
    footprint = changed.loc[0, "geometry"]
    coords = list(footprint.exterior.coords)
    changed.loc[0, "geometry"] = shapely.Polygon([coords[0], *coords])
    assert changed.loc[0, "geometry"].equals(footprint)
    assert len(changed.gs.sample()) == 1
