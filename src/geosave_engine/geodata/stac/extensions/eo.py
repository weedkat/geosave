"""EO extension: what each band is called and where in the spectrum it sits."""

from __future__ import annotations

import pystac
import xarray as xr
from pystac.extensions.eo import Band, EOExtension

from typing import Any

import geosave_engine.geodata.attrs as attrs


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Name one EO band per variable, adding spectral facts where it has them.

    Tools that read a catalog without opening its files, such as odc-stac,
    name a multi-band asset's variables from these names.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `open_raster` opens it.

    Examples:
        >>> write(asset, scene)
        >>> [band.name for band in EOExtension.ext(asset).bands]
        ['red', 'nir']
    """
    bands = []
    for name in raster.gs.variables:
        facts = raster[name].gs.spectral
        bands.append(
            Band.create(
                name=name,
                common_name=None if facts is None else facts.common_name,
                center_wavelength=None if facts is None else facts.center_wavelength,
                full_width_half_max=None
                if facts is None
                else facts.full_width_half_max,
                description=raster[name].attrs.get("long_name"),
            )
        )
    EOExtension.ext(asset, add_if_missing=True).apply(bands)


def read(band: Band) -> dict[str, Any]:
    """Read the CF attrs one EO band states.

    Args:
        band: One band of an asset's EO listing.

    Returns:
        Its spectral facts, and its description as `long_name`. Empty where
        it states only its name.

    Examples:
        >>> read(Band.create(name="B04", common_name="red"))
        {'common_name': 'red'}
    """
    spectral = attrs.Spectral(
        common_name=band.common_name,
        center_wavelength=band.center_wavelength,
        full_width_half_max=band.full_width_half_max,
    )
    found = spectral.to_attrs()
    if band.description is not None:
        found["long_name"] = band.description
    return found
