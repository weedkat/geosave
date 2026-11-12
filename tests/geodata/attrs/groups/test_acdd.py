"""ACDD spatial extent retains shape and coordinate order."""

import pytest
from odc.geo.geom import Geometry
from shapely import from_wkt

from geosave.geodata import attrs


def test_acdd_extent_axis_order_and_invalid_metadata():
    city = Geometry(from_wkt("POLYGON ((13 52, 14 52, 13 53, 13 52))"), "EPSG:4326")
    group = attrs.ACDDExtent.from_geometry(city)
    stored = group.to_attrs()
    assert stored["geospatial_bounds_crs"] == "EPSG:4326"
    assert "52 13" in stored["geospatial_bounds"]
    assert attrs.ACDDExtent.from_attrs(stored).to_geometry() == city
    default = attrs.ACDDExtent.from_attrs({"geospatial_bounds": "POINT (52 13)"})
    assert default.to_geometry().coords[0] == (13, 52)
    for fields in (
        {"geospatial_bounds_crs": "EPSG:4326"},
        {"geospatial_bounds": "POLYGON EMPTY"},
        {"geospatial_bounds": "POINT Z (1 2 3)"},
    ):
        with pytest.raises(ValueError):
            attrs.ACDDExtent.from_attrs(fields)
