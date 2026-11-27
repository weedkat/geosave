"""A declared extent keeps its shape and CRS, in x/y order."""

import pytest
from odc.geo.geom import Geometry
from shapely import from_wkt

from geosave.geodata import attrs


def test_extent_round_trips_through_attrs_in_xy_order():
    city = Geometry(from_wkt("POLYGON ((13 52, 14 52, 13 53, 13 52))"), "EPSG:4326")

    stored = attrs.Extent.from_geometry(city).to_attrs()

    assert stored == {
        "extent": "POLYGON ((13 52, 14 52, 13 53, 13 52))",
        "extent_crs": "EPSG:4326",
    }
    restored = attrs.Extent.from_attrs(stored)
    assert restored is not None
    assert restored.to_geometry() == city


def test_extent_clips_to_a_geometry_or_clears_when_they_miss():
    square = Geometry(from_wkt("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"), "EPSG:32749")
    extent = attrs.Extent.from_geometry(square)

    half = Geometry(from_wkt("POLYGON ((2 0, 6 0, 6 4, 2 4, 2 0))"), "EPSG:32749")
    away = Geometry(from_wkt("POLYGON ((9 9, 10 9, 10 10, 9 10, 9 9))"), "EPSG:32749")

    clipped = extent.clip(half)
    assert clipped is not None
    assert clipped.to_geometry().area == 8
    assert extent.clip(away) is None


@pytest.mark.parametrize(
    "fields",
    [
        {"extent": "POINT (13 52)"},
        {"extent": "POLYGON EMPTY", "extent_crs": "EPSG:4326"},
        {"extent": "POINT Z (1 2 3)", "extent_crs": "EPSG:4326"},
        {"extent": "not wkt", "extent_crs": "EPSG:4326"},
    ],
)
def test_extent_refuses_a_missing_crs_or_an_empty_3d_or_unreadable_geometry(fields):
    with pytest.raises(ValueError):
        attrs.Extent.from_attrs(fields)
