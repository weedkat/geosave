from odc.geo.geom import BoundingBox
from geosave.geodata.viz.map import BASEMAPS, frame_bbox


def test_native_basemaps_and_attribution():
    assert set(BASEMAPS) == {"Light", "Dark", "Imagery", "OpenStreetMap", "None"}
    for name, tile in BASEMAPS.items():
        if tile is not None:
            assert tile.data.startswith("https://")


def test_frames_projected_extents_and_points():
    point = frame_bbox(BoundingBox(500000, 9000000, 500000, 9000000, crs=32749))
    assert point.span_x > 0 and point.span_y > 0
    box = BoundingBox(0, 0, 1000, 1000, crs=3857)
    assert frame_bbox(box).left < box.left and frame_bbox(box).right > box.right
