import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, MultiPoint, Point, box

from geosave.geodata.viz import shapes


def frame(*geometries) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(geometries), crs=3857)


class TestByType:
    def test_folds_multi_into_each_type_in_drawing_order(self) -> None:
        geometry_frames = shapes.group_by_type(
            frame(Point(0, 0), box(0, 0, 1, 1), MultiPoint([(1, 1), (2, 2)]))
        )

        assert list(geometry_frames) == ["Polygon", "Point"]
        assert list(geometry_frames["Point"].index) == [0, 2]

    def test_refuses_nothing_to_draw_and_collections(self) -> None:
        with pytest.raises(ValueError, match="no geometry"):
            shapes.group_by_type(frame())
        with pytest.raises(ValueError, match="explode"):
            shapes.group_by_type(frame(GeometryCollection([Point(0, 0)])))


def test_counts_rows_by_geometry_type() -> None:
    summarize_geometry = shapes.summarize_geometry(
        frame(box(0, 0, 1, 1), Point(0, 0), Point(1, 1))
    )

    assert summarize_geometry == "1 polygon, 2 points"
