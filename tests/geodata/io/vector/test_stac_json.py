"""A static STAC catalog of JSON files written from an item table, and read back."""

from __future__ import annotations

import shutil

import orjson
import pystac
import pytest

from geosave.geodata import stac
from geosave.geodata.io.vector import stac_json
from tests.geodata.conftest import item_table
from tests.geodata.stac.conftest import entry


@pytest.fixture
def labels():
    return stac.create_collection(
        "label", description="Hand-drawn classes", license="CC-BY-4.0"
    )


def _read(path):
    root = pystac.Catalog.from_file(str(path))
    return root, stac.create_catalog(root.get_items(recursive=True))


def test_a_catalog_round_trips_through_json(rows, labels, tmp_path) -> None:
    path = rows.gs.to_stac_json(tmp_path / "stac", collections=[labels])

    root, restored = _read(path)

    assert path == tmp_path / "stac" / "catalog.json"

    def by_key(table):
        items = table.gs.to_items()
        return {(item.collection_id, item.id): item for item in items}

    written, read = by_key(rows), by_key(restored)
    assert read.keys() == written.keys()
    for key, item in written.items():
        assert read[key].datetime == item.datetime
        assert read[key].to_dict()["assets"] == item.to_dict()["assets"]
    assert root.get_child("label").description == "Hand-drawn classes"
    assert root.get_child("optical").description == "optical"


def test_a_written_catalog_validates(rows, tmp_path) -> None:
    path = rows.gs.to_stac_json(tmp_path / "stac")

    root = pystac.Catalog.from_file(str(path))

    assert root.validate_all() == len(rows)


def test_assets_are_stored_relative_so_the_catalog_moves_with_them(
    rows, tmp_path
) -> None:
    path = rows.gs.to_stac_json(tmp_path / "stac")
    stored = orjson.loads(next((path.parent / "label").rglob("*/*.json")).read_bytes())
    assert all(asset["href"].startswith("../") for asset in stored["assets"].values())

    moved = tmp_path.with_name(f"{tmp_path.name}_moved")
    shutil.move(tmp_path, moved)
    _, restored = _read(moved / "stac" / "catalog.json")

    hrefs = [
        asset.href for item in restored.gs.to_items() for asset in item.assets.values()
    ]
    assert hrefs
    assert all(href.startswith(str(moved)) for href in hrefs)


def test_items_without_a_collection_sit_under_the_root(scene, tmp_path) -> None:
    rows = item_table(scene.gs.to_cog(f"{tmp_path}/scene_{{time:%Y%m%d}}.tif"))

    root, restored = _read(rows.gs.to_stac_json(tmp_path / "stac"))

    assert [item.id for item in root.get_items()] == rows["id"].tolist()
    assert restored["id"].tolist() == rows["id"].tolist()


def test_an_existing_catalog_is_kept_unless_overwritten(rows, tmp_path) -> None:
    rows.gs.to_stac_json(tmp_path / "stac")

    with pytest.raises(FileExistsError):
        rows.gs.to_stac_json(tmp_path / "stac")
    rows.iloc[:1].gs.to_stac_json(tmp_path / "stac", overwrite=True)

    _, restored = _read(tmp_path / "stac" / "catalog.json")
    assert len(restored) == 1


def test_a_repeated_id_in_one_collection_is_refused(rows, tmp_path) -> None:
    import pandas as pd

    repeated = pd.concat([rows, rows.iloc[:1]], ignore_index=True)

    with pytest.raises(ValueError, match="share the id"):
        repeated.gs.to_stac_json(tmp_path / "stac")


def test_items_linking_to_an_api_root_are_written_without_reading_it(
    rows, tmp_path
) -> None:
    # A searched Item links to its API's root, parent and collection.
    unreachable = "http://127.0.0.1:9/stac"
    items = rows.gs.to_items()
    for item in items:
        item.add_link(pystac.Link(pystac.RelType.ROOT, f"{unreachable}/catalog.json"))
        item.add_link(pystac.Link(pystac.RelType.PARENT, f"{unreachable}/catalog.json"))
    searched = stac.create_catalog(items, collections=rows.gs.collections.values())

    root, restored = _read(searched.gs.to_stac_json(tmp_path / "stac"))

    assert len(restored) == len(rows)
    assert all(
        link.get_absolute_href().startswith(str(tmp_path))
        for item in root.get_items(recursive=True)
        for link in item.links
        if link.rel in ("root", "parent")
    )


def test_a_catalog_round_trips_through_a_remote_filesystem(rows) -> None:
    from upath import UPath

    root_file = rows.gs.to_stac_json("memory://catalogs/stac")
    root = pystac.Catalog.from_file(str(root_file), stac_io=stac_json.UPathStacIO())

    assert isinstance(root_file, UPath) and root_file.protocol == "memory"
    assert len(list(root.get_items(recursive=True))) == len(rows)


def test_json_overwrite_keeps_assets_and_unrelated_files(tmp_path):
    from geosave.geodata.io.vector import stac_json

    item = entry(None)
    stac_json.write([item], tmp_path)
    asset = tmp_path / "pixels.tif"
    asset.write_bytes(b"raster")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep")
    stac_json.write([item], tmp_path, overwrite=True)
    assert asset.read_bytes() == b"raster"
    assert notes.read_text() == "keep"


@pytest.mark.parametrize("existing_catalog", [False, True])
def test_json_refuses_unrelated_destination_collision(tmp_path, existing_catalog):
    from geosave.geodata.io.vector import stac_json

    if existing_catalog:
        stac_json.write([entry(None)], tmp_path)
    second = entry(None)
    second.id = "second"
    destination = tmp_path / "second" / "second.json"
    destination.parent.mkdir()
    destination.write_text('{"user": "keep"}')
    with pytest.raises(FileExistsError, match="not owned"):
        stac_json.write([second], tmp_path, overwrite=True)
    assert destination.read_text() == '{"user": "keep"}'


def test_native_json_preserves_relative_descriptive_links(tmp_path):
    from geosave.geodata.io.vector import stac_json

    item = entry(None)
    item.set_self_href(str(tmp_path / "source" / "item.json"))
    item.add_link(pystac.Link("license", "./license.txt"))
    root = stac_json.write([item], tmp_path / "published")
    restored = next(pystac.Catalog.from_file(str(root)).get_items(recursive=True))
    assert restored.get_single_link("license").get_absolute_href() == str(
        tmp_path / "source" / "license.txt"
    )
    assert item.get_single_link("license").target == "./license.txt"
