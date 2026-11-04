"""Extents on a Web Mercator map: what it can show, and how to frame it."""

import math

from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, Geometry, bbox_union, box

# Latitude where Web Mercator tiles stop: its y span there equals its x span.
MERCATOR_LATITUDE = 85.0511287798

# Web Mercator x at 180° longitude, in metres.
WEB_MERCATOR_EDGE = 20037508.342789244

# Pixel size at zoom 0, in metres: the world across one 256-pixel tile.
ZOOM_0_PIXEL_SIZE = 2 * WEB_MERCATOR_EDGE / 256

# Plot size in pixels, (height, width), of a view drawn before the page has one.
DEFAULT_VIEW_SHAPE = (600, 900)

# Finest zoom level web maps tile, about 1 cm per pixel.
MAX_ZOOM = 24


def is_known_range(values: tuple[float, float] | None) -> bool:
    """Say whether a map range holds two finite bounds.

    A page reports NaN bounds before its plot has a size.
    """
    return values is not None and all(math.isfinite(value) for value in values)


def frame_bbox(bbox: BoundingBox) -> BoundingBox:
    """Frame a projected extent with a margin, including isolated points."""
    box = bbox.to_crs("EPSG:3857")
    margin = max(box.span_x, box.span_y) * 0.05 or 1000
    return box.buffered(margin, margin)


def to_web_mercator(extent: Geometry) -> BoundingBox:
    """Return the part of an extent a web map can show, in Web Mercator metres.

    Raises:
        ValueError: The extent lies wholly poleward of Web Mercator's latitudes.
    """
    shown = extent.to_crs("EPSG:4326", wrapdateline=True)
    bounds = shown.boundingbox
    if bounds.bottom >= MERCATOR_LATITUDE or bounds.top <= -MERCATOR_LATITUDE:
        raise ValueError(
            f"the data lies beyond {MERCATOR_LATITUDE:.2f}° latitude, "
            "which a web map cannot show"
        )
    # A point's or a line's extent has no area, and intersecting it empties it.
    if shown.area > 0:
        shown = shown & box(
            -180, -MERCATOR_LATITUDE, 180, MERCATOR_LATITUDE, "EPSG:4326"
        )
    if shown.geom_type != "MultiPolygon":
        return shown.to_crs("EPSG:3857").boundingbox
    parts = [part.to_crs("EPSG:3857").boundingbox for part in shown.geoms]
    if all(part.left >= 0 for part in parts) or all(part.left < 0 for part in parts):
        return bbox_union(parts)
    # An extent split at the antimeridian continues east of it: tiles wrap,
    # and reprojecting back to the data's CRS wraps x to the same ground.
    world_width = 2 * WEB_MERCATOR_EDGE
    continued = []
    for part in parts:
        if part.left < 0:
            part = BoundingBox(
                part.left + world_width,
                part.bottom,
                part.right + world_width,
                part.top,
                crs=part.crs,
            )
        continued.append(part)
    return bbox_union(continued)


def find_zoom_level(view: GeoBox) -> int:
    """Return the zoom level whose pixel size is nearest the view's.

    Within a factor of 1.4 of the screen's pixels, so a read stays about the
    screen's sharpness without sending the browser twice its pixels per axis.
    """
    # The finer axis, so the read stays sharp across a stretched view.
    pixel_size = min(abs(view.resolution.x), abs(view.resolution.y))
    zoom = round(math.log2(ZOOM_0_PIXEL_SIZE / pixel_size))
    return min(max(zoom, 0), MAX_ZOOM)


def create_zoom_grid(bounds: BoundingBox, zoom: int) -> GeoBox:
    """Return a Web Mercator grid covering `bounds` at the zoom level's pixel size."""
    return GeoBox.from_bbox(
        bounds.to_crs("EPSG:3857"),
        crs="EPSG:3857",
        resolution=ZOOM_0_PIXEL_SIZE / 2**zoom,
    )


def create_view(
    x_range: tuple[float, float], y_range: tuple[float, float], shape: tuple[int, int]
) -> GeoBox:
    """Return the grid a map shows: its ranges in Web Mercator, `shape` its (height, width) pixels."""
    return GeoBox.from_bbox(
        (x_range[0], y_range[0], x_range[1], y_range[1]),
        crs="EPSG:3857",
        shape=shape,
        tight=True,
    )
