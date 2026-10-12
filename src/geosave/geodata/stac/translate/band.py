"""A STAC band as the CF attrs of the variable it loads into."""

from __future__ import annotations

from typing import Any

from pystac.extensions.eo import Band
from pystac.extensions.raster import RasterBand

from geosave.geodata.stac.extensions import classification, eo, raster


def read(stored: RasterBand, spectral: Band) -> dict[str, Any]:
    """Translate one band into the CF attrs it states.

    `nodata` and `data_type` describe the file, and the loader that opens it
    states them, so neither is translated.

    Args:
        stored: The band as the Raster extension describes it.
        spectral: The band as the EO extension describes it.

    Returns:
        The attrs the Raster, EO and Classification extensions read from the
        band; empty where it states none.

    Examples:
        >>> read(RasterBand.create(unit="1", scale=0.0001), Band({}))
        {'scale_factor': 0.0001, 'units': '1'}
    """
    return {**raster.read(stored), **eo.read(spectral), **classification.read(stored)}
