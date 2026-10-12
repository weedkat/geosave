"""Per-pixel features derived from bands: spectral indices and cloud masks.

Examples:
    >>> scene = scene.assign(ndvi=features.ndvi(scene, nir="B08", red="B04"))
    >>> clear = features.scl_valid_mask(scene, scl="SCL")
"""

from geosave_engine.geodata.features.cloud_mask import (
    cdi_cloud_mask,
    cirrus_cloud_mask,
    s2cloudless_mask,
    scl_valid_mask,
)
from geosave_engine.geodata.features.spectral_indices import (
    bsi,
    evi,
    evi2,
    mndwi,
    msavi2,
    nbr,
    ndbi,
    ndci,
    ndmi,
    ndre,
    ndsi,
    ndvi,
    ndwi,
    savi,
)
from geosave_engine.geodata.features.shadow_mask import shadow_mask

__all__ = [
    "s2cloudless_mask",
    "cdi_cloud_mask",
    "cirrus_cloud_mask",
    "scl_valid_mask",
    "ndvi",
    "evi",
    "evi2",
    "savi",
    "msavi2",
    "ndre",
    "bsi",
    "ndbi",
    "ndwi",
    "mndwi",
    "ndci",
    "ndmi",
    "nbr",
    "ndsi",
    "shadow_mask",
]
