"""Projection extension: the grid a file's pixels sit on."""

from __future__ import annotations

from typing import Any

import pystac
from affine import Affine
from odc.geo.geobox import GeoBox
from pystac.extensions.projection import AssetProjectionExtension, ProjectionExtension


def from_geobox(grid: GeoBox | None) -> dict[str, Any]:
    """Spell a grid as the Projection fields an asset carries.

    Args:
        grid: The grid of a file's pixels.

    Returns:
        `proj:code`, or `proj:wkt2` for a CRS without an EPSG code, with
        `proj:shape` and `proj:transform`.

    Raises:
        ValueError: The grid is missing or has no CRS.

    Examples:
        >>> from_geobox(scene.gs.geobox)["proj:code"]
        'EPSG:32633'
    """
    if not isinstance(grid, GeoBox) or grid.crs is None:
        raise ValueError(
            "the raster carries no locatable grid, which a STAC asset states; "
            "keep it in an ordinary reference table instead"
        )
    epsg = grid.crs.epsg
    stated: dict[str, Any] = {
        "proj:shape": list(grid.shape),
        "proj:transform": list(grid.transform)[:6],
    }
    if epsg is None:
        stated["proj:wkt2"] = grid.crs.to_wkt()
    else:
        stated["proj:code"] = f"EPSG:{epsg}"
    return stated


def to_geobox(asset: pystac.Asset) -> GeoBox | None:
    """Read the grid an asset's Projection fields state.

    Args:
        asset: Asset of an Item.

    Returns:
        The grid, or None where the asset states no shape or transform.

    Examples:
        >>> to_geobox(item.assets["image"]).shape
        Shape2d(x=64, y=64)
    """
    # Read without the schema check, as catalogs often omit declaring it.
    stated = AssetProjectionExtension(asset)
    if stated.shape is None or stated.transform is None:
        return None
    crs = stated.code if stated.code is not None else stated.wkt2
    height, width = stated.shape
    return GeoBox((height, width), Affine(*stated.transform[:6]), crs)


def write_asset(asset: pystac.Asset, grid: GeoBox) -> None:
    """Write the file grid and declare Projection on the Asset's owner."""
    fields = from_geobox(grid)
    extension = ProjectionExtension.ext(asset, add_if_missing=True)
    extension.shape = fields["proj:shape"]
    extension.transform = fields["proj:transform"]
    extension.code = fields.get("proj:code")
    extension.wkt2 = fields.get("proj:wkt2")
