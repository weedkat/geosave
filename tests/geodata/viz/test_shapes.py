import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, box

from geosave.geodata.viz.shapes import split_by_geometry_type


def frame(*shapes) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(shapes), crs=4326)


def test_splits_by_type_in_drawing_order_with_multi_types_joined() -> None:
    mixed = frame(
        Point(1, 1),
        MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
        LineString([(0, 0), (1, 1)]),
        box(4, 0, 5, 1),
    )

    rows_by_type = split_by_geometry_type(mixed)

    assert list(rows_by_type) == ["Polygon", "LineString", "Point"]
    assert rows_by_type["Polygon"].index.tolist() == [1, 3]


def test_leaves_out_rows_without_geometry() -> None:
    rows_by_type = split_by_geometry_type(frame(box(0, 0, 1, 1), None))

    assert rows_by_type["Polygon"].index.tolist() == [0]


def test_refuses_no_geometry_and_collections() -> None:
    with pytest.raises(ValueError, match="nothing to draw"):
        split_by_geometry_type(frame(None))
    with pytest.raises(ValueError, match="GeometryCollection"):
        split_by_geometry_type(frame(GeometryCollection([Point(0, 0)])))
