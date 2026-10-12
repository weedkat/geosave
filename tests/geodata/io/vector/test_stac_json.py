"""A static STAC catalog of JSON files written from an item table, and read back."""

from __future__ import annotations

import shutil

import orjson
import pystac
import pytest

from geosave.geodata import stac, stack
from geosave.geodata.io.vector import stac_json
from tests.geodata.conftest import stack_templates


@pytest.fixture
def rows(scene, label, tmp_path):
    sample = stack({"optical": scene, "label": label})
    saved = sample.gs.to_cog(stack_templates(sample, tmp_path / "s0"))
    return stac.create_item(saved, id="s0")


@pytest.fixture
def labels():
    return stac.create_collection("label", description="Hand-drawn classes", license="CC-BY-4.0")


def _read(path):
    root = pystac.Catalog.from_file(str(path))
    return root, stac.from_item(root.get_items(recursive=True))


def test_a_catalog_round_trips_through_json(rows, labels, tmp_path) -> None:
    path = stac_json.write(rows, tmp_path / "stac", collections=[labels])

    root, restored = _read(path)

    assert path == tmp_path / "stac" / "catalog.json"

    def by_key(table):
        items = table.gs.to_item()
        return {(item.collection_id, item.id): item for item in items}

    written, read = by_key(rows), by_key(restored)
    assert read.keys() == written.keys()
    for key, item in written.items():
        assert read[key].datetime == item.datetime
        assert read[key].to_dict()["assets"] == item.to_dict()["assets"]
    assert root.get_child("label").description == "Hand-drawn classes"
    assert root.get_child("optical").description == "optical"


def test_a_written_catalog_validates(rows, tmp_path) -> None:
    path = stac_json.write(rows, tmp_path / "stac")

    root = pystac.Catalog.from_file(str(path))

    assert root.validate_all() == len(rows)


def test_assets_are_stored_relative_so_the_catalog_moves_with_them(
    rows, tmp_path
) -> None:
    path = stac_json.write(rows, tmp_path / "stac")
    stored = orjson.loads(next((path.parent / "label").rglob("*/*.json")).read_bytes())
    assert all(asset["href"].startswith("../") for asset in stored["assets"].values())

    moved = tmp_path.with_name(f"{tmp_path.name}_moved")
    shutil.move(tmp_path, moved)
    _, restored = _read(moved / "stac" / "catalog.json")

    hrefs = [
        asset.href for item in restored.gs.to_item() for asset in item.assets.values()
    ]
    assert hrefs
    assert all(href.startswith(str(moved)) for href in hrefs)


def test_items_without_a_collection_sit_under_the_root(scene, tmp_path) -> None:
    rows = stac.create_item(scene.gs.to_cog(f"{tmp_path}/scene_{{time:%Y%m%d}}.tif"))

    root, restored = _read(stac_json.write(rows, tmp_path / "stac"))

    assert [item.id for item in root.get_items()] == rows["id"].tolist()
    assert restored["id"].tolist() == rows["id"].tolist()


def test_an_existing_catalog_is_kept_unless_overwritten(rows, tmp_path) -> None:
    stac_json.write(rows, tmp_path / "stac")

    with pytest.raises(FileExistsError):
        stac_json.write(rows, tmp_path / "stac")
    stac_json.write(rows.iloc[:1], tmp_path / "stac", overwrite=True)

    _, restored = _read(tmp_path / "stac" / "catalog.json")
    assert len(restored) == 1


def test_a_repeated_id_in_one_collection_is_refused(rows, tmp_path) -> None:
    import pandas as pd

    repeated = pd.concat([rows, rows.iloc[:1]], ignore_index=True)

    with pytest.raises(ValueError, match="repeat"):
        stac_json.write(repeated, tmp_path / "stac")


def test_items_linking_to_an_api_root_are_written_without_reading_it(
    rows, tmp_path
) -> None:
    # A searched Item links to its API's root, parent and collection.
    unreachable = "http://127.0.0.1:9/stac"
    items = rows.gs.to_item()
    for item in items:
        item.add_link(pystac.Link(pystac.RelType.ROOT, f"{unreachable}/catalog.json"))
        item.add_link(pystac.Link(pystac.RelType.PARENT, f"{unreachable}/catalog.json"))
    searched = stac.from_item(items)

    root, restored = _read(stac_json.write(searched, tmp_path / "stac"))

    assert len(restored) == len(rows)
    assert all(
        link.get_absolute_href().startswith(str(tmp_path))
        for item in root.get_items(recursive=True)
        for link in item.links
        if link.rel in ("root", "parent")
    )


def test_a_catalog_round_trips_through_a_remote_filesystem(rows) -> None:
    from upath import UPath

    root_file = stac_json.write(rows, "memory://catalogs/stac")
    root = pystac.Catalog.from_file(str(root_file), stac_io=stac_json.UPathStacIO())

    assert isinstance(root_file, UPath) and root_file.protocol == "memory"
    assert len(list(root.get_items(recursive=True))) == len(rows)
