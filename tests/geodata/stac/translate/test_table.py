"""Pure Item-table conversion keeps order, fields and identity."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pystac.extensions.raster import RasterExtension

from geosave.geodata.core import GeoCatalog
from geosave.geodata.io.raster.file import RasterFile, Scene
from geosave.geodata.io.raster.layout import layout
from geosave.geodata.stac import create_items
from geosave.geodata.stac.translate import table as stac_table
from tests.geodata.stac.conftest import create_native_item, entry


def _items(raster, collection=None):
    scenes = []
    for planned in layout(
        raster,
        "/tmp/{time:%Y%m%d}_{band}.tif" if "time" in raster.dims else "/tmp/label.tif",
    ):
        files = {
            key: RasterFile.from_xarray(file.path, file.raster)
            for key, file in planned.files.items()
        }
        scenes.append(Scene(collection, files, coords=planned.coords))
    return list(
        create_items(
            scenes,
            id_format="forest_{start:%Y%m%dT%H%M%S}",
            datetime=datetime(2025, 6, 1, tzinfo=UTC),
        )
    )


def test_items_become_rows_in_wgs84(scene, tmp_path) -> None:
    rows = stac_table.from_items(_items(scene))

    assert rows["id"].tolist() == ["forest_20250601T000000", "forest_20250611T000000"]
    assert rows.crs.to_epsg() == 4326
    assert "bbox" not in rows


def test_one_item_becomes_one_row(scene, tmp_path) -> None:
    first, _ = _items(scene)

    rows = stac_table.from_items(first)

    assert rows["id"].tolist() == [first.id]


def test_no_items_give_an_empty_catalog() -> None:
    rows = stac_table.from_items([])

    assert rows.empty
    assert rows.crs.to_epsg() == 4326
    assert type(rows.gs) is GeoCatalog


def test_rows_become_the_items_they_came_from(scene, tmp_path) -> None:
    items = _items(scene)

    restored = stac_table.to_items(stac_table.from_items(items))

    assert [item.id for item in restored] == [item.id for item in items]
    assert restored[0].bbox == pytest.approx(items[0].bbox)
    assert restored[0].assets["red"].href == items[0].assets["red"].href
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].nodata == 0


def test_filtered_rows_keep_only_the_assets_they_have(scene, label, tmp_path) -> None:
    rows = stac_table.from_items(_items(scene) + _items(label, "label"))

    labels = stac_table.to_items(rows[rows["collection"] == "label"])

    assert [list(item.assets) for item in labels] == [["label"]]


def test_concatenated_tables_become_the_items_they_came_from(scene, label) -> None:
    items = _items(scene) + _items(label, "label")
    # pandas 3 exports a concatenated table as sliced Arrow chunks.
    rows = GeoCatalog.concat(
        [
            stac_table.from_items(_items(scene)),
            stac_table.from_items(_items(label, "label")),
        ]
    )

    restored = stac_table.to_items(rows)

    assert [item.id for item in restored] == [item.id for item in items]
    assert [sorted(item.assets) for item in restored] == [
        sorted(item.assets) for item in items
    ]


def test_empty_table_converts_to_no_items():
    assert stac_table.to_items(stac_table.from_items([])) == []


def test_a_native_item_needs_no_geosave_fields(tmp_path):
    entry, _ = create_native_item(tmp_path)

    rows = stac_table.from_items([entry])

    assert rows.id.tolist() == [entry.id]
    assert rows.iloc[0]["provider:quality"] == "good"
    assert not any(name.startswith("geosave:") for name in rows.columns)
    assert (
        rows.iloc[0].assets["reflectance"]["href"] == entry.assets["reflectance"].href
    )


def test_relative_assets_resolve_without_mutating_thecreate_native_item(
    tmp_path,
):
    entry, _ = create_native_item(tmp_path)
    entry.assets["reflectance"].href = "scene.tif"
    entry.set_self_href(str(tmp_path / "scene.json"))

    rows = stac_table.from_items([entry])

    assert entry.assets["reflectance"].href == "scene.tif"
    assert rows.iloc[0].assets["reflectance"]["href"] == str(tmp_path / "scene.tif")


def test_duplicate_items_rejected_by_both_writers(tmp_path):
    from geosave.geodata.io.vector import stac_geoparquet, stac_json
    from geosave.geodata.stac.translate.table import from_items

    item = entry(None)
    rows = from_items([item, item])
    with pytest.raises(ValueError, match="share the id"):
        stac_geoparquet.write(rows, tmp_path / "duplicate.parquet")
    with pytest.raises(ValueError, match="share the id"):
        stac_json.write([item, item], tmp_path / "json")
