"""Construct native Item tables without reading assets."""

from copy import deepcopy

import geopandas as gpd

from geosave import GeoCatalog
from geosave.geodata.stac.catalog import from_items
from tests.geodata.conftest import create_test_collection
from tests.geodata.stac.conftest import create_native_item, entry


def test_catalog_preserves_item_order_and_copies_collection_definitions():
    collection = create_test_collection("optical", description="Optical imagery")
    before = deepcopy(collection.to_dict())
    first = entry("optical")
    second = entry("optical")
    first.id = "first"
    second.id = "second"

    catalog = from_items(iter([second, first]), collections=[collection])
    collection.description = "Changed"

    assert isinstance(catalog, gpd.GeoDataFrame)
    assert type(catalog.gs) is GeoCatalog
    assert catalog.id.tolist() == ["second", "first"]
    assert catalog.crs.to_epsg() == 4326
    assert "bbox" not in catalog
    assert catalog.attrs["stac-geoparquet"]["collections"]["optical"] == before


def test_empty_catalog_retains_its_schema_and_collections():
    collection = create_test_collection("optical", description="Optical imagery")

    catalog = from_items([], collections=[collection])

    assert catalog.empty
    assert type(catalog.gs) is GeoCatalog
    assert set(catalog.gs.collections) == {"optical"}
    assert catalog.crs is not None
    assert catalog.crs.to_epsg() == 4326


def test_catalog_preserves_native_item_fields(tmp_path):
    entry, _ = create_native_item(tmp_path)
    rows = from_items([entry])
    assert rows.id.tolist() == [entry.id]
    assert rows.iloc[0]["provider:quality"] == "good"
    assert (
        rows.iloc[0].assets["reflectance"]["href"] == entry.assets["reflectance"].href
    )


def test_relative_assets_resolve_without_mutating_the_item(tmp_path):
    entry, _ = create_native_item(tmp_path)
    entry.assets["reflectance"].href = "scene.tif"
    entry.set_self_href(str(tmp_path / "scene.json"))
    rows = from_items([entry])
    assert entry.assets["reflectance"].href == "scene.tif"
    assert rows.iloc[0].assets["reflectance"]["href"] == str(tmp_path / "scene.tif")
