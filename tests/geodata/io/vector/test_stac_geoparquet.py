"""stac-geoparquet: Items to rows, to one file, and back to pixels."""

from __future__ import annotations


import geopandas as gpd
import pyarrow.parquet as pq
import pystac
import pytest

from stac_geoparquet.arrow import stac_table_to_items

from geosave.geodata import GeoCatalog, open_vector
from geosave.geodata.io.vector import stac_geoparquet as table
from geosave.geodata.stac.catalog import create_catalog
from tests.geodata.conftest import (
    create_test_collection,
)
from tests.geodata.stac.conftest import create_native_item, entry


def _refuse(*args, **kwargs):
    raise AssertionError("the table computed pixels")


def _save(items, path, **options):
    options.setdefault(
        "collections",
        [
            create_test_collection(key, description=key)
            for key in dict.fromkeys(item.collection_id for item in items)
            if key is not None
        ],
    )
    return create_catalog(items).gs.to_geoparquet(path, **options)


def _collections():
    return [
        create_test_collection(name, description=f"{name} rows", license="CC-BY-4.0")
        for name in ("optical", "label")
    ]


def test_a_table_that_is_not_stac_is_returned_as_read(tmp_path) -> None:
    from shapely.geometry import Point

    plain = gpd.GeoDataFrame(
        [{"id": "a", "assets": {"label": {"href": "a.tif"}}, "quality": 0.97}],
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    path = plain.gs.to_geoparquet(tmp_path / "labels.parquet")

    rows = open_vector(path)

    assert rows.loc[0, "quality"] == 0.97
    assert rows.loc[0, "assets"]["label"]["href"] == "a.tif"
    assert type(rows.gs) is GeoCatalog


def test_a_vector_that_is_not_an_item_table_does_not_write_as_one(tmp_path) -> None:
    from shapely.geometry import Point

    plain = gpd.GeoDataFrame([{"id": "a"}], geometry=[Point(0, 0)], crs="EPSG:4326")

    with pytest.raises(ValueError, match="item table"):
        table.write_stac_geoparquet(plain, tmp_path / "plain.parquet")
    written = plain.gs.to_geoparquet(tmp_path / "plain.parquet")
    assert b"stac-geoparquet" not in pq.read_metadata(written).metadata


def test_importing_core_and_io_loads_no_stac() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import geosave.geodata.core, geosave.geodata.io; "
        "assert not {'pystac', 'rustac', 'stac_geoparquet'} & set(sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_collection_survives_read_filter_rewrite(tmp_path):
    collection = create_test_collection(
        "optical", description="Surface reflectance", license="CC-BY-4.0"
    )
    collection.extra_fields["custom:version"] = "v1"
    rows = create_catalog([entry()], collections=[collection])
    saved = rows.gs.to_geoparquet(tmp_path / "first.parquet")
    restored = open_vector(saved)
    second = restored.iloc[:1].gs.to_geoparquet(tmp_path / "second.parquet")
    assert (
        open_vector(second).gs.collections["optical"].description
        == collection.description
    )
    assert (
        open_vector(second).gs.collections["optical"].extra_fields["custom:version"]
        == "v1"
    )
    assert collection.extent.spatial.bboxes == [[-180.0, -90.0, 180.0, 90.0]]


def test_collection_links_and_nested_fields_survive_publication(tmp_path):
    collection = create_test_collection(
        "optical", description="Reflectance", keywords=["optical"]
    )
    collection.add_link(pystac.Link("license", "https://example.com/license"))
    rows = create_catalog([entry()], collections=[collection])
    collection.keywords.append("changed")
    assert rows.gs.collections["optical"].keywords == ["optical"]
    saved = rows.gs.to_geoparquet(tmp_path / "links.parquet")
    restored = open_vector(saved).gs.collections["optical"]
    assert restored.get_single_link("license").target == "https://example.com/license"


def test_a_written_table_retains_item_and_asset_properties(tmp_path):
    first, source = create_native_item(tmp_path, "first")
    second, _ = create_native_item(tmp_path, "second")
    bands = [
        {"name": name, "data_type": source[name].dtype.name}
        for name in source.gs.variables
    ]
    first.assets["reflectance"].extra_fields["bands"] = bands
    second.properties["new:property"] = 7
    second.assets["other"] = second.assets.pop("reflectance")

    saved = create_catalog([second, first]).gs.to_geoparquet(tmp_path / "items.parquet")

    restored = list(stac_table_to_items(pq.read_table(saved)))
    assert [entry["id"] for entry in restored] == ["second", "first"]
    assert restored[0]["properties"]["new:property"] == 7
    assert restored[1]["assets"]["reflectance"]["bands"] == bands


def test_a_table_keeps_its_bbox_through_read_and_rewrite(tmp_path):
    entry, _ = create_native_item(tmp_path)
    first = create_catalog([entry]).gs.to_geoparquet(tmp_path / "catalog.parquet")
    second = open_vector(first).gs.to_geoparquet(tmp_path / "again.parquet")

    for path in (first, second):
        (restored,) = stac_table_to_items(pq.read_table(path))
        assert restored["bbox"] == pytest.approx(entry.bbox)


def test_an_edited_geometry_changes_the_bbox_written(tmp_path):
    entry, _ = create_native_item(tmp_path)
    rows = create_catalog([entry])
    rows = rows.set_geometry(rows.geometry.translate(xoff=1.0))

    (restored,) = stac_table_to_items(
        pq.read_table(rows.gs.to_geoparquet(tmp_path / "catalog.parquet"))
    )

    assert restored["bbox"][0] == pytest.approx(entry.bbox[0] + 1.0)
