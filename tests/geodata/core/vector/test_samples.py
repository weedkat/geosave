import numpy as np
import pytest

from geosave.geodata import open_vector
from geosave.geodata.core import GeoCatalog, GeoSamples
from geosave.geodata.io.vector import geoparquet


def test_gs_reads_a_sample_table_through_geosamples(items):
    assert isinstance(items.gs, GeoCatalog)
    assert isinstance(items.gs.sample().gs, GeoSamples)


@pytest.mark.parametrize("reader", [open_vector, geoparquet.open_vector])
@pytest.mark.parametrize("remote", [False, True])
def test_a_sample_table_round_trips_with_its_collections(
    items, tmp_path, bucket, reader, remote, grid
):
    chips = items.gs.sample().gs.chip(32, overlap=8, target=grid)

    destination = f"{bucket}/chips.parquet" if remote else tmp_path / "chips.parquet"
    path = chips.gs.to_geoparquet(destination)
    restored = reader(path)

    assert isinstance(restored.gs, GeoSamples)
    assert sorted(restored.gs.collections) == ["dem", "label", "s1", "s2"]
    for column in ("id", "area", "items", "start_datetime", "proj:code"):
        assert restored[column].tolist() == chips[column].tolist()
    assert [list(names) for names in restored["collections"]] == chips[
        "collections"
    ].tolist()
    for column in ("proj:transform", "proj:shape", "pad_widths"):
        restored_values = [np.array(list(value)).tolist() for value in restored[column]]
        assert restored_values == chips[column].tolist()
    original = chips.iloc[4].gs.to_xarray()
    assert restored.iloc[4].gs.to_xarray().identical(original)


def test_a_sample_table_refuses_write_options(items, tmp_path):
    with pytest.raises(ValueError, match="fixes its own encoding"):
        items.gs.sample().gs.to_geoparquet(tmp_path / "s.parquet", compression="zstd")


def test_a_sample_table_is_queried_by_the_vector_rules(items, grid):
    chips = items.gs.sample().gs.chip(25, target=grid)
    first_chip = chips.iloc[0].gs.anchor.geobox.extent

    assert len(chips.gs.query(first_chip, predicate="covers")) == 1
    assert len(chips.gs.query(when="2023")) == 0


def test_a_callers_own_columns_ride_through_frame_and_chip(composites, grid):
    samples = composites.gs.sample().assign(split="train")

    with pytest.warns(Warning):
        chips = samples.gs.frame(
            windows={"s2": {"slots": ["-1M", "0M"]}, "s1": {"slots": ["-1M", "0M"]}}
        ).gs.chip(32, target=grid)

    assert list(chips.columns[: len(GeoSamples.COLUMNS)]) == list(GeoSamples.COLUMNS)
    assert set(chips["split"]) == {"train"}


def test_concat_preserves_nested_collections_and_round_trips(items, tmp_path):
    left = items[items["collection"].isin(["label", "s2"])].gs.sample()
    right = items[items["collection"].isin(["label", "dem"])].gs.sample()
    combined = GeoSamples.concat([left, right])

    assert set(combined.gs.collections) == {"label", "s1", "s2", "dem"}
    restored = open_vector(combined.gs.to_geoparquet(tmp_path / "combined.parquet"))
    assert set(restored.gs.collections) == {"label", "s1", "s2", "dem"}
    assert restored.iloc[0].gs.to_xarray()["s2"].dataset.attrs["id"] == "s2"
    combined.attrs["stac:collections"]["s2"]["description"] = "changed"
    assert left.attrs["stac:collections"]["s2"]["description"] != "changed"


def test_concat_keeps_the_first_conflicting_collection_definition(items):
    from copy import deepcopy

    left = items.gs.sample()
    right = left.copy()
    right.attrs = deepcopy(left.attrs)
    right.attrs["stac:collections"]["s2"]["description"] = "different"
    combined = GeoSamples.concat([left, right])
    assert (
        combined.attrs["stac:collections"]["s2"] == left.attrs["stac:collections"]["s2"]
    )


@pytest.mark.parametrize(
    "method,suffix", [("to_geojson", ".geojson"), ("to_geopackage", ".gpkg")]
)
def test_samples_reject_exports_that_lose_collections(items, tmp_path, method, suffix):
    samples = items.gs.sample()
    path = tmp_path / ("samples" + suffix)
    with pytest.raises(ValueError, match="to_geoparquet"):
        getattr(samples.gs, method)(path)
    assert not path.exists()


def test_renamed_geometry_keeps_sample_operations_and_round_trips(
    items, tmp_path, grid
):
    from geosave.geodata.core import GeoSample

    samples = items.gs.sample().rename_geometry("footprint")
    assert isinstance(samples.gs, GeoSamples)
    assert isinstance(samples.iloc[0].gs, GeoSample)
    assert "s2" in samples.iloc[0].gs.to_xarray().children
    chips = samples.gs.chip(32, target=grid)
    assert chips.geometry.name == "footprint"
    assert isinstance(chips.gs, GeoSamples)
    assert chips.geometry.iloc[0].equals(
        chips.iloc[0].gs.anchor.geobox.extent.to_crs("EPSG:4326").geom
    )
    restored = open_vector(samples.gs.to_geoparquet(tmp_path / "renamed.parquet"))
    assert restored.geometry.name == "footprint"
    assert isinstance(restored.gs, GeoSamples)


def test_frames_preserve_renamed_geometry_and_crs(composites):
    samples = composites.gs.sample().rename_geometry("footprint").to_crs("EPSG:3857")
    with pytest.warns(Warning):
        frames = samples.gs.frame(
            windows={"s2": {"slots": ["-1M", "0M"]}, "s1": {"slots": ["-1M", "0M"]}}
        )
    assert len(frames) > 0
    assert frames.geometry.name == "footprint"
    assert frames.crs == samples.crs
    assert frames.geometry.notna().all()
    assert frames.geometry.geom_equals(samples.geometry.iloc[0]).all()
