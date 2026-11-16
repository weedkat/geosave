"""The rules a STAC Item table can be asked to meet name the row that breaks them."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geosave.geodata.errors import VectorValidationError
from geosave.geodata.stac.catalog import (
    create_catalog,
    one_collection,
    stac_collections,
    stac_items,
)
from tests.geodata.conftest import create_test_collection
from tests.geodata.stac.conftest import entry


def build_rows(**columns) -> gpd.GeoDataFrame:
    """One Item row per entry of `id`, at the origin, with the Item columns."""
    count = len(columns["id"])
    fields = {"links": [[]] * count, "stac_extensions": [[]] * count}
    fields["assets"] = [{"image": {"href": "/data/image.tif"}}] * count
    fields.update(columns)
    return gpd.GeoDataFrame(fields, geometry=[Point(0, 0)] * count, crs=4326)


def test_stac_items_requires_the_item_columns():
    features = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)

    build_rows(id=["scene"]).gs.validate(stac_items)
    with pytest.raises(VectorValidationError, match="need columns"):
        features.gs.validate(stac_items)


def test_stac_items_requires_an_href_on_every_asset():
    rows = build_rows(id=["scene"], assets=[{"image": {"title": "no file"}}])

    with pytest.raises(VectorValidationError, match="'image' has no absolute href"):
        rows.gs.validate(stac_items)


def test_stac_items_requires_absolute_hrefs():
    assets = [{"image": {"href": "/data/a.tif"}}, {"image": {"href": "./b.tif"}}]
    rows = build_rows(id=["a", "b"], assets=assets)

    with pytest.raises(VectorValidationError, match="position 1.*'image'"):
        rows.gs.validate(stac_items)


def test_stac_items_requires_ids_unique_within_a_collection():
    across = build_rows(id=["scene", "scene"], collection=["s2", "s1"])
    within = build_rows(id=["scene", "scene"], collection=["s2", "s2"])

    across.gs.validate(stac_items)
    with pytest.raises(VectorValidationError, match="share an id"):
        within.gs.validate(stac_items)


def test_stac_collections_requires_a_definition_of_each_named_collection():
    optical = create_test_collection("optical", description="Optical imagery")
    defined = create_catalog([entry("optical")], collections=[optical])
    undefined = create_catalog([entry("optical")])

    defined.gs.validate(stac_collections)
    with pytest.raises(VectorValidationError, match="without their definition"):
        undefined.gs.validate(stac_collections)


def test_one_collection_refuses_rows_of_several():
    rows = build_rows(id=["a", "b"], collection=["s2", "s1"])

    rows[rows["collection"] == "s2"].gs.validate(one_collection)
    with pytest.raises(VectorValidationError, match="select one"):
        rows.gs.validate(one_collection)
