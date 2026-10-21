"""Shared collection-table contracts through the public accessors."""

from __future__ import annotations

from copy import deepcopy

import pytest

from geosave.geodata import stac
from geosave.geodata.core import GeoCatalog, GeoSamples
from tests.geodata.conftest import build_raster
from tests.geodata.stac.conftest import entry


@pytest.fixture(params=[GeoCatalog, GeoSamples])
def table(request, items):
    if request.param is GeoSamples:
        return items.gs.sample()
    return items


def test_collection_objects_are_independent_of_stored_definitions(table):
    original = deepcopy(table.attrs["stac:collections"])
    collection = next(iter(table.gs.collections.values()))
    collection.description = "edited"
    collection.extra_fields["review"] = {"status": "changed"}
    assert table.attrs["stac:collections"] == original


def test_concat_keeps_independent_definitions_and_the_first_duplicate(table):
    combined = table.gs.concat([table, table])
    assert set(combined.gs.collections) == set(table.gs.collections)
    name = next(iter(combined.attrs["stac:collections"]))
    combined.attrs["stac:collections"][name]["description"] = "changed"
    assert table.attrs["stac:collections"][name]["description"] != "changed"
    assert (
        table.gs.concat([table, combined]).attrs["stac:collections"][name]
        == (table.attrs["stac:collections"][name])
    )


def test_empty_tables_keep_their_kind(table):
    empty = table.gs.concat([table.iloc[:0]])
    assert type(empty.gs) is type(table.gs)
    assert empty.empty


@pytest.mark.parametrize(
    "method,suffix", [("to_geojson", ".geojson"), ("to_geopackage", ".gpkg")]
)
def test_collection_tables_refuse_lossy_exports(table, tmp_path, method, suffix):
    path = tmp_path / ("table" + suffix)
    with pytest.raises(ValueError, match="to_geoparquet"):
        getattr(table.gs, method)(path)
    assert not path.exists()


def test_concat_merges_collections_and_keeps_the_first_definition():
    optical = stac.create_collection("optical", description="Reflectance")
    label = stac.create_collection("label", description="Land cover")
    a = stac.create_catalog([entry()], collections=[optical])
    b = stac.create_catalog([entry("label")], collections=[label])
    assert set(GeoCatalog.concat([a, b]).gs.collections) == {"optical", "label"}
    optical.description = "Conflicting description"
    c = stac.create_catalog([entry()], collections=[optical])
    assert (
        GeoCatalog.concat([a, c]).gs.collections["optical"].description == "Reflectance"
    )
    assert (
        GeoCatalog.concat([c, a]).gs.collections["optical"].description
        == "Conflicting description"
    )


def test_concat_feeds_multicollection_stacks(tmp_path):
    from geosave.geodata import tree

    ds = build_raster(times=1)
    layers = tree({"optical": ds, "label": ds[["red"]].rename(red="label")})
    scenes = layers.gs.to_cog(str(tmp_path / "{collection}_{time:%Y%m%d}.tif"))
    catalogs = []
    for scene in scenes:
        collection = stac.create_collection(
            scene.collection, description=scene.collection
        )
        catalogs.append(
            stac.create_catalog(stac.create_items(scene), collections=[collection])
        )
    combined = GeoCatalog.concat(catalogs)
    samples = combined.gs.sample()
    assert len(samples) == 1
    assert set(samples.iloc[0].gs.items) == {"optical", "label"}
    loaded = samples.iloc[0].gs.to_xarray()
    assert set(loaded.children) == {"optical", "label"}
    assert loaded["optical"].dataset.red.chunks is not None


def test_concat_keeps_the_first_definition_even_after_filtering_rows():
    optical = stac.create_collection("optical", description="Old optical")
    label = stac.create_collection("label", description="Labels")
    rows = stac.create_catalog([entry(), entry("label")], collections=[optical, label])
    optical.description = "New optical"
    fresh = stac.create_catalog([entry()], collections=[optical])
    combined = GeoCatalog.concat([rows[rows.collection == "label"], fresh])
    assert combined.gs.collections["optical"].description == "Old optical"


def test_concat_refuses_mixing_item_and_sampling_rows():
    import geopandas as gpd

    rows = stac.create_catalog([entry(None)])
    sampling = gpd.GeoDataFrame(
        {"sources": ["{}"]}, geometry=rows.geometry, crs=rows.crs
    )
    with pytest.raises(ValueError, match="different kinds"):
        GeoCatalog.concat([rows, sampling])
