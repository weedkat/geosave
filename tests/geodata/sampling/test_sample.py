"""Spatial grouping retains native observations and never chooses a grid."""

from __future__ import annotations

import runpy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import pytest
import shapely

from geosave.geodata.core import GeoSamples
from geosave.geodata.stac.translate import table as stac_table
from tests.geodata.stac.conftest import create_native_item


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


def test_sample_groups_equal_footprints_and_keeps_containing_scenes_separate(items):
    items = items.copy()
    radar = items.collection == "s1"
    items.loc[radar, "geometry"] = items.loc[radar, "geometry"].translate(0.01)
    samples = items.gs.sample()
    assert isinstance(samples.gs, GeoSamples)
    assert len(samples) == 2
    assert samples.iloc[0].gs.geobox is None
    assert set(samples.iloc[0].gs.items) == {"label", "dem", "s2"}
    assert set(samples.iloc[1].gs.items) == {"s1"}
    assert len(samples.iloc[0].gs.items["s2"]) == 5


def test_public_sample_frame_scenarios():
    root = Path(__file__).parents[3]
    runpy.run_path(
        str(
            root
            / "docs/contributing/handoffs/2026-10-16-sample-frame/probe_sample_api.py"
        )
    )


def test_thumbnail_cannot_change_sample_grid_or_data_names(tmp_path):
    from geosave.geodata.stac.extensions import projection

    entry, source = create_native_item(tmp_path)
    entry.collection_id = "optical"
    data = entry.assets["reflectance"]
    data.extra_fields.update(projection.from_geobox(source.gs.geobox))
    data.extra_fields["bands"] = [{"name": name} for name in source.data_vars]
    thumbnail = entry.assets["thumbnail"]
    thumbnail.extra_fields.update(projection.from_geobox(source.gs.geobox[:1, :1]))
    entry.assets = {"red": thumbnail, "reflectance": data}

    samples = stac_table.from_items([entry]).gs.sample()
    row = samples.iloc[0]
    loaded = row.gs.to_xarray()["optical"].dataset

    assert row.gs.geobox == source.gs.geobox
    assert list(loaded.data_vars) == list(source.data_vars)
    assert loaded.red.chunks is not None
    assert loaded.gs.geobox == source.gs.geobox
    np.testing.assert_array_equal(loaded.red.values, source.red.values)


def test_unreferenced_image_remains_readable_without_a_sampling_grid(tmp_path):
    import dask.array as da
    import rasterio

    from geosave.geodata.warnings import MissingGridWarning

    path = tmp_path / "plain.tif"
    pixels = np.arange(12, dtype="uint8").reshape(3, 4)
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning):
        with rasterio.open(
            path, "w", driver="GTiff", height=3, width=4, count=1, dtype="uint8"
        ) as output:
            output.write(pixels, 1)
    item = pystac.Item(
        "plain",
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        [0, 0, 1, 1],
        datetime(2025, 1, 1, tzinfo=UTC),
        {},
        collection="images",
    )
    item.add_asset(
        "image",
        pystac.Asset(str(path), media_type=pystac.MediaType.GEOTIFF, roles=["data"]),
    )
    rows = stac_table.from_items([item])

    with pytest.warns(MissingGridWarning, match="plain"):
        assert rows.iloc[0].gs.anchor is None
    with pytest.warns(MissingGridWarning, match="plain"):
        assert rows.gs.sample().empty
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning):
        loaded = rows.iloc[0].gs.to_xarray()

    assert loaded.gs.crs is None
    assert isinstance(loaded.image.data, da.Array)
    np.testing.assert_array_equal(loaded.image.values, pixels)
