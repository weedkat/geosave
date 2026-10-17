"""The geometry a figure or map can draw, split by geometry type.

Figures and maps draw one geometry type per element or layer, a Multi* type
with its own: polygons, lines, points.
"""

from __future__ import annotations

from typing import cast

import geopandas as gpd
import pandas as pd

# Geometry types drawn, in drawing order: areas under lines under points.
TYPES = ("Polygon", "LineString", "Point")

# Word counting each geometry type, as in "3 polygons".
NOUNS = {"Polygon": "polygon", "LineString": "line", "Point": "point"}


def select_drawable(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the rows of a frame that carry geometry a figure or map can draw.

    Raises:
        ValueError: The frame holds no geometry, or a GeometryCollection.
    """
    drawable_df = cast(
        "gpd.GeoDataFrame", frame[~(frame.geometry.is_empty | frame.geometry.isna())]
    )
    if drawable_df.empty:
        raise ValueError("nothing to draw: the frame holds no geometry")
    if unsupported_types := set(_get_geometry_types(drawable_df)) - set(TYPES):
        raise ValueError(f"cannot draw {sorted(unsupported_types)}; explode them first")
    return drawable_df


def group_by_type(frame: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """Split a frame's drawable rows by geometry type, in drawing order.

    Returns:
        {"Polygon" | "LineString" | "Point": rows of that type, index kept},
        only the types the frame holds.

    Raises:
        ValueError: The frame holds no geometry, or a GeometryCollection.
    """
    drawable_df = select_drawable(frame)
    geometry_types = _get_geometry_types(drawable_df)
    groups = {}
    for geometry_type in TYPES:
        rows = geometry_types == geometry_type
        if rows.any():
            groups[geometry_type] = cast("gpd.GeoDataFrame", drawable_df[rows])
    return groups


def summarize_geometry(frame: gpd.GeoDataFrame) -> str:
    """Count a frame's rows by geometry type, such as "1 polygon, 4 points"."""
    counts = _get_geometry_types(select_drawable(frame)).value_counts()
    return ", ".join(
        f"{counts[geometry_type]} {noun}{'' if counts[geometry_type] == 1 else 's'}"
        for geometry_type, noun in NOUNS.items()
        if geometry_type in counts
    )


def _get_geometry_types(frame: gpd.GeoDataFrame) -> pd.Series:
    """Return each row's geometry type, a Multi* type as its single one."""
    return frame.geom_type.str.removeprefix("Multi")
