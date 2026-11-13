"""A STAC GeoParquet table answers the queries a STAC API does."""

from __future__ import annotations

from tests.conftest import write_cogs

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from geosave.geodata.errors import CollectionNotFoundError
from geosave.geodata.stac import StacClient, StacQuery
from geosave.geodata.stac.catalog import create_catalog
from geosave.geodata.stac.search.table import StacTable
from tests.geodata.conftest import create_file_items, create_test_collection


@pytest.fixture
def catalog(scene, tmp_path: Path):
    """Write two dated scenes of one Collection as one table."""
    forest = create_test_collection(
        "forest", description="Forest samples", license="CC-BY-4.0"
    )
    paths = write_cogs(
        scene, tmp_path, format_paths="forest_{time:%Y%m%dT%H%M%S}/{variable}.tif"
    )
    first, second = create_file_items(
        paths, id_format="forest_{start:%Y%m%dT%H%M%S}", collection=forest
    )
    first.properties["eo:cloud_cover"] = 5.0
    second.properties["eo:cloud_cover"] = 40.0
    path = create_catalog([first, second]).gs.to_geoparquet(
        tmp_path / "catalog" / "items.parquet", collections=[forest]
    )
    return path, (first, second)


def test_a_table_answers_a_stac_query(catalog) -> None:
    folder, (first, second) = catalog
    client = StacTable(folder)
    forest = StacQuery(collections=["forest"])

    assert [item.id for item in client.search_items(forest)] == [first.id, second.id]
    assert client.search_items(StacQuery(collections=["absent"])) == []
    assert [
        item.id
        for item in client.search_items(forest.set_filter("eo:cloud_cover <= 10"))
    ] == [first.id]
    assert [
        item.id for item in client.search_items(forest.sort_by("-eo:cloud_cover"))
    ] == [
        second.id,
        first.id,
    ]
    assert [
        item.id
        for item in client.search_items(
            StacQuery(collections=["forest"], ids=[second.id])
        )
    ] == [second.id]


def test_a_table_is_searched_by_place_and_time(catalog) -> None:
    folder, (first, second) = catalog
    client = StacTable(folder)

    inside = StacQuery(collections=["forest"], bbox=tuple(first.bbox))
    outside = StacQuery(collections=["forest"], bbox=(0.0, 0.0, 1.0, 1.0))
    early = StacQuery(
        collections=["forest"],
        datetime=(datetime(2025, 5, 30), datetime(2025, 6, 5)),
    )

    assert len(client.search_items(inside)) == 2
    assert client.search_items(outside) == []
    assert [item.id for item in client.search_items(early)] == [first.id]


def test_a_table_states_its_collections(catalog) -> None:
    folder, _ = catalog
    client = StacTable(folder)

    assert client.collections() == {"forest"}
    assert client.collection("forest").license == "CC-BY-4.0"
    with pytest.raises(CollectionNotFoundError, match="missing"):
        client.collection("missing")


def test_a_source_loads_from_a_table(catalog, scene) -> None:
    folder, _ = catalog
    from geosave.geodata.stac.load import load_dataset

    client = StacTable(str(folder))
    items = client.search_items(
        StacQuery(collections=["forest"]).set_region(scene.gs.region)
    )
    loaded = load_dataset(
        items,
        collection=client.collection("forest"),
        geobox=scene.gs.geobox,
        bands=["red", "nir"],
        groupby="time",
    )

    assert dict(loaded.sizes) == {"time": 2, "y": 64, "x": 64}
    # odc-stac masks the file's fill value; every stored reading comes back.
    stored = scene.red.values != 0
    np.testing.assert_array_equal(loaded.red.values[stored], scene.red.values[stored])
    assert loaded.gs.stac_collection.license == "CC-BY-4.0"


def test_search_returns_an_item_table(scene, tmp_path) -> None:
    forest = create_test_collection("forest", description="Forest samples")
    paths = write_cogs(scene, tmp_path, format_paths="forest_{time:%Y%m%dT%H%M%S}.tif")
    rows = create_catalog(
        create_file_items(paths, collection=forest), collections=[forest]
    )
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet", collections=[forest])
    client = StacTable(path)

    found = client.search(StacQuery(collections=["forest"]))
    none = client.search(StacQuery(collections=["absent"]))

    assert found["id"].tolist() == rows["id"].tolist()
    assert none.empty


def test_stac_client_opens_a_parquet_table_like_an_api(catalog) -> None:
    folder, _ = catalog

    client = StacClient.open(folder)

    assert isinstance(client, StacTable)
    assert client.collections() == {"forest"}
    assert not client.search(StacQuery(collections=["forest"])).empty
    with pytest.raises(ValueError, match="duckdb"):
        StacClient.open("element84", duckdb=object())  # type: ignore[arg-type]


def test_search_results_retain_definitions_for_publication(catalog, tmp_path):
    from geosave.geodata import open_vector

    path, (first, _) = catalog
    client = StacTable(path)
    found = client.search(StacQuery(collections=["forest"], ids=[first.id]))

    assert found.gs.collections["forest"].license == "CC-BY-4.0"
    saved = found.gs.to_geoparquet(tmp_path / "subset.parquet")
    restored = open_vector(saved)
    assert restored.id.tolist() == [first.id]
    assert restored.gs.collections["forest"].description == "Forest samples"

    found.attrs["stac-geoparquet"]["collections"]["forest"]["license"] = "other"
    assert client.collection("forest").license == "CC-BY-4.0"
