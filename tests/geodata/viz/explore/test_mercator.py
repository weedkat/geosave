from odc.geo.geom import BoundingBox
from geosave.geodata.viz.explore.mercator import frame_bbox


def test_frames_projected_extents_and_points():
    point = frame_bbox(BoundingBox(500000, 9000000, 500000, 9000000, crs=32749))
    assert point.span_x > 0 and point.span_y > 0
    box = BoundingBox(0, 0, 1000, 1000, crs=3857)
    assert frame_bbox(box).left < box.left and frame_bbox(box).right > box.right


def test_web_mercator_extent_keeps_points_and_refuses_polar_data():
    import pytest
    from odc.geo.geom import box, point

    from geosave.geodata.viz.explore.mercator import to_web_mercator

    assert to_web_mercator(point(10, 5, "EPSG:4326")).span_x == 0
    clipped = to_web_mercator(box(0, 80, 10, 89, "EPSG:4326"))
    assert clipped.top == pytest.approx(20037508.34, rel=1e-6)
    with pytest.raises(ValueError, match="beyond 85.05° latitude"):
        to_web_mercator(box(0, 86, 10, 89, "EPSG:4326"))


def test_web_mercator_extent_continues_across_the_antimeridian():
    from odc.geo.geom import box

    from geosave.geodata.viz.explore.mercator import WEB_MERCATOR_EDGE, to_web_mercator

    # UTM zone 1 south, reaching about half a degree west of 180°.
    extent = to_web_mercator(box(120000, 8000000, 220000, 8100000, "EPSG:32701"))
    assert extent.left < WEB_MERCATOR_EDGE < extent.right
    assert extent.span_x < 200_000


def create_view_with_pixel_size(pixel_size):
    from odc.geo.geobox import GeoBox

    return GeoBox.from_bbox(
        (0, 0, 900 * pixel_size, 600 * pixel_size), crs="EPSG:3857", shape=(600, 900)
    )


def test_find_zoom_level_takes_the_nearest_level():
    from geosave.geodata.viz.explore.mercator import find_zoom_level

    # z11 pixels are 76.4 m and z10 pixels 152.9 m; they meet near 108 m
    assert find_zoom_level(create_view_with_pixel_size(100.0)) == 11
    assert find_zoom_level(create_view_with_pixel_size(115.0)) == 10
    assert find_zoom_level(create_view_with_pixel_size(152.0)) == 10


def test_find_zoom_level_stays_within_the_tile_levels():
    from geosave.geodata.viz.explore.mercator import find_zoom_level

    assert find_zoom_level(create_view_with_pixel_size(1e-6)) == 24
    assert find_zoom_level(create_view_with_pixel_size(1e7)) == 0


def test_create_zoom_grid_covers_the_bounds():
    import pytest

    from geosave.geodata.viz.explore.mercator import ZOOM_0_PIXEL_SIZE, create_zoom_grid

    bounds = BoundingBox(1000, 2000, 51000, 32000, crs=3857)

    grid = create_zoom_grid(bounds, 11)

    assert grid.resolution.x == pytest.approx(ZOOM_0_PIXEL_SIZE / 2**11)
    covered = grid.boundingbox
    assert covered.left <= bounds.left and covered.right >= bounds.right
    assert covered.bottom <= bounds.bottom and covered.top >= bounds.top
