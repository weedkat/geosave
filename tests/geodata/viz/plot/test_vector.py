import geopandas as gpd
import holoviews as hv
import matplotlib
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon


matplotlib.use("agg")
hv.extension("matplotlib")


RING = [(0, 0), (10, 0), (10, 10), (0, 10)]
HOLE = [(2, 2), (4, 2), (4, 4), (2, 4)]


def frame(*shapes, crs="EPSG:32749") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(shapes), crs=crs)


class TestOutline:
    def test_draws_polygons_with_holes(self) -> None:
        drawn = frame(Polygon(RING, [HOLE])).gs.plot()

        (polygons,) = drawn
        assert isinstance(polygons, hv.Polygons)
        assert [len(holes) for holes in polygons.holes()] == [1]
        assert polygons.range("x") == (0, 10)
        hv.render(drawn)

    def test_draws_lines_and_points(self) -> None:
        (lines,) = frame(LineString(RING)).gs.plot()
        (points,) = frame(MultiPoint([(1, 1), (2, 2)])).gs.plot()

        assert isinstance(lines, hv.Path) and isinstance(points, hv.Points)

    def test_overlays_mixed_geometry(self) -> None:
        drawn = frame(Polygon(RING), Point(20, 20)).gs.plot()

        assert [type(layer).__name__ for layer in drawn] == ["Polygons", "Points"]
        hv.render(drawn)

    def test_refuses_nothing_to_draw_and_collections(self) -> None:
        with pytest.raises(ValueError, match="no geometry"):
            frame().gs.plot()
        with pytest.raises(ValueError, match="explode"):
            frame(GeometryCollection([Point(0, 0)])).gs.plot()
