"""Shared collection-table contracts through the public accessors."""

from copy import deepcopy

import pytest

from geosave.geodata.core import GeoCatalog, GeoSamples


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
