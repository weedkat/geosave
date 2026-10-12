"""STAC extensions GeoSave writes, one module per schema.

Each module writes its schema's fields onto an asset from a saved raster,
through that schema's PySTAC class.
"""

from . import classification, eo, projection, raster

# Written onto every asset, in this order: classes sit inside the bands the
# Raster extension stores, so it comes before Classification.
ASSET = (projection, raster, eo, classification)

__all__ = [
    "ASSET",
    "classification",
    "eo",
    "projection",
    "raster",
]
