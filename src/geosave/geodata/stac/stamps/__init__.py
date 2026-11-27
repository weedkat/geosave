"""Optional metadata callables for STAC raster loads.

A stamp receives the loaded Dataset, its source Items and the load's ODC
`stac_cfg`, which band stamps need to find a configured alias's asset. Pass a
name from `STAMPS`, a callable, or a sequence of both in execution order.

Examples:
    >>> from geosave.geodata.stac.stamps import stamp_raster
    >>> scene = stac.load_dataset(items, geobox=grid, stamps=(stamp_raster, "eo"))
    >>> scene = stac.load_dataset(items, geobox=grid, stac_cfg=cfg, stamps=["raster"])
"""

from collections.abc import Callable, Sequence

import pystac
import xarray as xr
from odc.stac import ConversionConfig

from .classification import stamp_classification
from .collection import stamp_collection
from .display import stamp_rgb, stamp_title
from .eo import stamp_eo
from .period import stamp_observation_period
from .properties import stamp_properties
from .raster import stamp_nodata, stamp_raster

type Stamp = Callable[
    [xr.Dataset, Sequence[pystac.Item], ConversionConfig | None], xr.Dataset
]
"""A function that adds metadata from the source Items to a loaded Dataset.

It receives the Dataset, the Items and the load's ODC `stac_cfg`, or None."""

STAMPS: dict[str, Stamp] = {
    "classification": stamp_classification,
    "collection": stamp_collection,
    "eo": stamp_eo,
    "observation_period": stamp_observation_period,
    "properties": stamp_properties,
    "raster": stamp_raster,
    "rgb": stamp_rgb,
    "title": stamp_title,
}

__all__ = [
    "STAMPS",
    "Stamp",
    "stamp_classification",
    "stamp_collection",
    "stamp_eo",
    "stamp_nodata",
    "stamp_observation_period",
    "stamp_properties",
    "stamp_raster",
    "stamp_rgb",
    "stamp_title",
]
