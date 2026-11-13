from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geosave.geodata.core.vector.catalog import (
    dated,
    one_collection,
    stac_collections,
    stac_items,
    valid_geometry,
)
from geosave.geodata.errors import VectorValidationError
from geosave.geodata.stac.catalog import concat_catalogs


def build_rows(**columns) -> gpd.GeoDataFrame:
    """One Item row per entry of `id`, at the origin, with the Item columns."""
    count = len(columns["id"])
    fields = {"links": [[]] * count, "stac_extensions": [[]] * count}
    fields["assets"] = [{"image": {"href": "/data/image.tif"}}] * count
    fields.update(columns)
    return gpd.GeoDataFrame(fields, geometry=[Point(0, 0)] * count, crs=4326)


def test_a_catalog_keeps_its_collections_through_geoparquet(catalog, tmp_path):
    from geosave import open_vector

    path = catalog.gs.to_geoparquet(tmp_path / "items.parquet")

    assert set(open_vector(path).gs.collections) == {"s2", "s1", "label"}


def test_concat_refuses_tables_on_different_crs() -> None:
    geographic = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    projected = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3857")

    with pytest.raises(ValueError, match="CRS"):
        concat_catalogs([geographic, projected])


def test_concat_keeps_every_collection_and_refuses_conflicting_ones(catalog):
    combined = concat_catalogs([catalog, catalog])

    assert len(combined) == 2 * len(catalog)
    assert set(combined.gs.collections) == set(catalog.gs.collections)

    # The footer is a copy: editing it leaves the source table unchanged.
    name = next(iter(combined.attrs["stac-geoparquet"]["collections"]))
    combined.attrs["stac-geoparquet"]["collections"][name]["description"] = "changed"
    assert catalog.gs.collections[name].description != "changed"
    with pytest.raises(ValueError, match="description"):
        concat_catalogs([catalog, combined])


def test_concat_needs_a_table() -> None:
    with pytest.raises(ValueError, match="at least one"):
        concat_catalogs([])


def test_valid_geometry_names_the_broken_row() -> None:
    rows = build_rows(id=["fine", "broken"])
    rows.loc[1, "geometry"] = None

    with pytest.raises(VectorValidationError, match="position 1.*broken"):
        rows.gs.validate(valid_geometry)


def test_dated_refuses_a_period_that_ends_before_it_starts():
    rows = build_rows(
        id=["invalid"],
        datetime=["2025-01-15"],
        start_datetime=["2025-02-01"],
        end_datetime=["2025-01-01"],
    )

    with pytest.raises(VectorValidationError, match="end_datetime precedes"):
        rows.gs.validate(dated)


def test_stac_items_requires_the_item_columns():
    features = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)

    build_rows(id=["scene"]).gs.validate(stac_items)
    with pytest.raises(VectorValidationError, match="need columns"):
        features.gs.validate(stac_items)


def test_stac_items_requires_an_href_on_every_asset():
    rows = build_rows(id=["scene"], assets=[{"image": {"title": "no file"}}])

    with pytest.raises(VectorValidationError, match="'image' has no href"):
        rows.gs.validate(stac_items)


def test_stac_items_requires_ids_unique_within_a_collection():
    across = build_rows(id=["scene", "scene"], collection=["s2", "s1"])
    within = build_rows(id=["scene", "scene"], collection=["s2", "s2"])

    across.gs.validate(stac_items)
    with pytest.raises(VectorValidationError, match="share an id"):
        within.gs.validate(stac_items)


def test_stac_collections_requires_a_definition_of_each_named_collection(catalog):
    undefined = catalog.copy()
    undefined.attrs = {"stac-geoparquet": {"collections": {}}}

    catalog.gs.validate(stac_collections)
    with pytest.raises(VectorValidationError, match="without their definition"):
        undefined.gs.validate(stac_collections)


def test_one_collection_refuses_rows_of_several(catalog):
    optical = catalog[catalog["collection"] == "s2"]

    optical.gs.validate(one_collection)
    with pytest.raises(VectorValidationError, match="select one"):
        catalog.gs.validate(one_collection)
