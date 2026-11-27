"""Items lay out as a stac-geoparquet table and convert back without reading assets."""

import json
from copy import deepcopy

import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point

from geosave import GeoCatalog
from geosave.geodata import io, stac
from geosave.geodata.stac.catalog import concat_catalogs, create_catalog
from geosave.geodata.stac.stamps import stamp_properties
from tests.geodata.conftest import create_test_collection
from tests.geodata.stac.conftest import entry
from tests.geodata.stac.test_publish import publish_two_grids


def test_catalog_preserves_item_order_and_copies_collection_definitions():
    collection = create_test_collection("optical", description="Optical imagery")
    before = deepcopy(collection.to_dict())
    first = entry("optical")
    second = entry("optical")
    first.id = "first"
    second.id = "second"

    catalog = create_catalog([second, first], collections=[collection])
    collection.description = "Changed"

    assert isinstance(catalog, gpd.GeoDataFrame)
    assert type(catalog.gs) is GeoCatalog
    assert catalog.id.tolist() == ["second", "first"]
    assert catalog.crs.to_epsg() == 4326
    assert "bbox" not in catalog
    assert catalog.attrs["stac-geoparquet"]["collections"]["optical"] == before


def test_empty_catalog_has_the_item_columns():
    catalog = create_catalog([])

    assert catalog.empty
    assert {"id", "links", "assets", "stac_extensions", "geometry"} <= set(catalog)
    assert catalog.gs.to_items() == []


def test_catalog_takes_the_collection_published_items_link_to(tmp_path):
    items, _, _ = publish_two_grids(tmp_path)

    catalog = create_catalog(items)

    assert catalog.id.tolist() == ["20250601", "20250602"]
    assert catalog["collection"].tolist() == ["optical", "optical"]
    assert catalog["eo:cloud_cover"].tolist() == [24.001125, 40.0]
    assert list(catalog.gs.collections) == ["optical"]


def test_catalog_rows_convert_back_to_the_items(tmp_path):
    items, _, _ = publish_two_grids(tmp_path)

    restored = create_catalog(items).gs.to_items()

    assert [item.id for item in restored] == [item.id for item in items]
    for original, item in zip(items, restored, strict=True):
        assert item.datetime == original.datetime
        # stac-geoparquet restates the datetime with microseconds.
        original.properties.pop("datetime", None)
        item.properties.pop("datetime", None)
        assert item.properties == original.properties
        assert item.stac_extensions == original.stac_extensions
        assert item.get_collection().id == "optical"
        for key, asset in original.assets.items():
            assert item.assets[key].to_dict() == asset.to_dict()


def test_published_catalog_round_trips_through_geoparquet(tmp_path):
    items, fine, coarse = publish_two_grids(tmp_path / "published")
    path = create_catalog(items).gs.to_geoparquet(
        tmp_path / "published" / "catalog.parquet"
    )

    opened = io.open_vector(path)
    restored = opened.gs.to_items()

    # Hrefs are stored absolute, so any stac-geoparquet reader locates the assets.
    stored = pq.read_table(path).column("assets")[0].as_py()["red"]["href"]
    assert stored == str(tmp_path / "published" / "20250601" / "red.tif")
    assert restored[0].assets["red"].href == stored
    assert list(opened.gs.collections) == ["optical"]
    loaded = stac.load_native_datasets(
        restored,
        chunks={},
        with_properties=[{"key": "eo:cloud_cover", "dtype": "float64"}],
        stamps=["collection", stamp_properties],
    )
    sources = [fine, coarse]
    for (_, _, native), source in zip(loaded, sources, strict=True):
        # Arrow keeps a row's assets as a struct, ordered by key.
        assert sorted(native.data_vars) == sorted(source.data_vars)
        assert native.gs.geobox == source.gs.geobox
        assert native.gs.stac_collection.id == "optical"
        assert native["eo_cloud_cover"].values.tolist() == [24.001125, 40.0]
        for name in native.data_vars:
            assert (native[name].values == source[name].values).all()


def test_a_row_converts_to_the_item_its_table_gives(tmp_path):
    items, _, _ = publish_two_grids(tmp_path)
    catalog = create_catalog(items)

    item = catalog.iloc[1].gs.to_item()

    # Compared as JSON: the row's geometry holds tuples, the table's lists.
    table_item = catalog.gs.to_items()[1]
    assert json.loads(json.dumps(item.to_dict())) == json.loads(
        json.dumps(table_item.to_dict())
    )
    assert item.get_collection().id == "optical"
    assert catalog.iloc[1].gs.region.timespan == catalog.iloc[[1]].gs.region.timespan


def test_a_plain_feature_row_is_not_an_item():
    from geosave.geodata.errors import VectorValidationError

    rows = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    with pytest.raises(VectorValidationError, match="STAC Items need columns"):
        rows.iloc[0].gs.to_item()


def test_catalog_rows_load_as_one_dataset(tmp_path):
    items, fine, _ = publish_two_grids(tmp_path)
    catalog = create_catalog(items)

    scene = catalog.gs.to_dataset(
        chunks={}, like=fine, bands=["red", "nir"], stamps=["collection"]
    )

    assert scene.gs.stac_collection.id == "optical"
    for name in ("red", "nir"):
        assert scene[name].chunks is not None
        assert (scene[name].values == fine[name].values).all()

    mixed = concat_catalogs([catalog, create_catalog([entry("radar")])])
    from geosave.geodata.errors import VectorValidationError

    with pytest.raises(VectorValidationError, match="collections"):
        mixed.gs.to_dataset(chunks={})


def test_relative_item_hrefs_become_absolute_in_the_table():
    item = entry()
    item.assets["red"].href = "./red.tif"
    with pytest.raises(ValueError, match="relative href"):
        create_catalog([item])

    item.set_self_href("s3://bucket/scenes/one.json")
    catalog = create_catalog([item])

    assert catalog["assets"][0]["red"]["href"] == "s3://bucket/scenes/red.tif"
