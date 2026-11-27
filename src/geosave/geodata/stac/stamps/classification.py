"""Stamp class values, labels and colours from the STAC Classification extension."""

from collections.abc import Sequence

import pystac
import xarray as xr
from odc.stac import ConversionConfig
from pystac.extensions.raster import RasterBand

from geosave.geodata.stac.extensions import classification

from .bands import find_band_sources, read_band_fields


def stamp_classification(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> xr.Dataset:
    """Write the values, names and colours of each classified band."""
    stamped = {}
    for variable, sources in find_band_sources(ds, items, stac_cfg):
        statements = []
        for _, asset, index in sources:
            fields = read_band_fields(asset, index)
            legend = classification.read_attrs(RasterBand(fields))
            statements.append(xr.Dataset(attrs=legend))

        agreed = xr.merge(statements, combine_attrs="drop_conflicts").attrs
        stamped[variable.name] = variable.assign_attrs({**agreed, **variable.attrs})

    return ds.assign(stamped)
