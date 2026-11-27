"""Stamp the spectral metadata declared by the STAC EO extension."""

from collections.abc import Sequence

import pystac
import xarray as xr
from odc.stac import ConversionConfig
from pystac.extensions.eo import Band

from geosave.geodata.stac.extensions import eo

from .bands import find_band_sources, read_band_fields


def stamp_eo(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> xr.Dataset:
    """Write the common name, wavelength, width and description of each band."""
    stamped = {}
    for variable, sources in find_band_sources(ds, items, stac_cfg):
        statements = []
        for _, asset, index in sources:
            fields = {}
            for key, value in read_band_fields(asset, index).items():
                fields[key.removeprefix("eo:")] = value

            spectral = eo.read_attrs(Band(fields))
            statements.append(xr.Dataset(attrs=spectral))

        agreed = xr.merge(statements, combine_attrs="drop_conflicts").attrs
        stamped[variable.name] = variable.assign_attrs({**agreed, **variable.attrs})

    return ds.assign(stamped)
