"""A STAC GeoParquet table answers the queries a STAC API does."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from geosave.geodata.errors import CollectionNotFoundError
from geosave.geodata.stac import StacClient, StacQuery
from geosave.geodata.stac.catalog import from_items
from geosave.geodata.stac.search.table import StacTable
from tests.geodata.conftest import create_file_items, create_test_collection


@pytest.fixture
def catalog(scene, tmp_path: Path):
    """Write two dated scenes of one Collection as one table."""
    forest = create_test_collection(
        "forest", description="Forest samples", license="CC-BY-4.0"
    )
    paths = scene.gs.to_cog(
        tmp_path, path_format="forest_{time:%Y%m%dT%H%M%S}/{band}.tif"
    )
    first, second = create_file_items(
        paths, id_format="forest_{start:%Y%m%dT%H%M%S}", collection=forest
    )
    first.properties["eo:cloud_cover"] = 5.0
    second.properties["eo:cloud_cover"] = 40.0
    path = from_items([first, second]).gs.to_geoparquet(
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
    from geosave.geodata.io import load_dataset

    client = StacTable(str(folder))
    items = client.search_items(
        StacQuery(collections=["forest"]).set_anchor(scene.gs.anchor)
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


def test_search_uses_the_rustac_session_configuration(catalog) -> None:
    import rustac

    folder, _ = catalog
    duckdb = rustac.DuckdbClient()
    duckdb.execute("SET enable_external_access = false")
    client = StacTable(str(folder), duckdb=duckdb)

    with pytest.raises(rustac.RustacError, match="disabled by configuration"):
        client.search_items(StacQuery(collections=["forest"]))


def test_a_custom_s3_endpoint_reads_items_and_stored_collections(catalog, tmp_path):
    import subprocess
    import sys

    import rustac

    _, (first, _) = catalog
    # rustac holds the GIL during I/O, so the endpoint needs its own process.
    code = """
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        requested = self.headers.get('Range')
        if requested is None:
            return super().do_GET()
        path = Path(self.translate_path(self.path))
        start, end = requested.removeprefix('bytes=').split('-')
        start, end = int(start), int(end) if end else path.stat().st_size - 1
        with path.open('rb') as source:
            source.seek(start)
            data = source.read(end - start + 1)
        self.send_response(206)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{path.stat().st_size}')
        self.end_headers()
        self.wfile.write(data)

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""
    server = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert server.stdout is not None
    port = int(server.stdout.readline())
    try:
        duckdb = rustac.DuckdbClient()
        duckdb.execute(
            "CREATE SECRET (TYPE s3, KEY_ID 'test-key', SECRET 'test-secret', "
            f"ENDPOINT '127.0.0.1:{port}', URL_STYLE 'path', "
            "USE_SSL false)"
        )
        # The HTTP server represents an S3 endpoint with tmp_path as its buckets.
        location = "s3://catalog/items.parquet"
        client = StacTable(location, duckdb=duckdb)

        (found, _) = client.search_items(StacQuery(collections=["forest"]))

        assert found.id == first.id
        assert client.collection("forest").license == "CC-BY-4.0"
        assert found.assets["red"].href.startswith("s3://")
    finally:
        server.terminate()
        server.wait(timeout=5)
        server.stdout.close()


def test_search_returns_an_item_table(scene, tmp_path) -> None:
    forest = create_test_collection("forest", description="Forest samples")
    paths = scene.gs.to_cog(tmp_path, path_format="forest_{time:%Y%m%dT%H%M%S}.tif")
    rows = from_items(create_file_items(paths, collection=forest), collections=[forest])
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet", collections=[forest])
    client = StacTable(path)

    found = client.search(StacQuery(collections=["forest"]))
    none = client.search(StacQuery(collections=["absent"]))

    assert found["id"].tolist() == rows["id"].tolist()
    assert [item.id for item in found.gs.to_items()] == rows.id.tolist()
    assert none.empty
    assert none.gs.to_items() == []


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
