"""Split a frame by geometry type, since plotting libraries draw one type per element.

GeoViews and hvPlot refuse a frame mixing polygons, lines and points, so
figures and the explorer draw one element per type.
"""

from __future__ import annotations

from typing import cast

import geopandas as gpd

# Geometry types drawn, in drawing order: areas under lines under points.
TYPES = ("Polygon", "LineString", "Point")


def split_by_geometry_type(frame: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """Split a frame's rows by geometry type, a Multi* type with its single one.

    Rows without geometry are left out.

    Returns:
        {"Polygon" | "LineString" | "Point": rows of that type, index kept},
        in drawing order, only the types the frame holds.

    Raises:
        ValueError: The frame holds no geometry, or a GeometryCollection.
    """
    has_geometry = ~(frame.geometry.is_empty | frame.geometry.isna())
    drawn = cast("gpd.GeoDataFrame", frame[has_geometry])
    if drawn.empty:
        raise ValueError("nothing to draw: the frame holds no geometry")
    geometry_types = drawn.geom_type.str.removeprefix("Multi")
    if unsupported_types := set(geometry_types) - set(TYPES):
        raise ValueError(f"cannot draw {sorted(unsupported_types)}; explode them first")
    rows_by_type = {}
    for geometry_type in TYPES:
        is_type = geometry_types == geometry_type
        if is_type.any():
            rows_by_type[geometry_type] = cast("gpd.GeoDataFrame", drawn[is_type])
    return rows_by_type
