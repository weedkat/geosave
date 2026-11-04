"""EO extension: what a band is called and where in the spectrum it sits."""

from __future__ import annotations

import pystac
import xarray as xr
from pystac.extensions.eo import Band, EOExtension

import geosave.geodata.attrs as attrs


def build_band(variable: xr.DataArray) -> Band:
    """Build the EO band of one variable.

    Tools that read a catalog without opening its files, such as odc-stac,
    name a several-band asset's variables from these names.

    Args:
        variable: Named variable as `open_raster` opens it.

    Returns:
        Its band: its name, spectral facts, and `long_name` as `description`.

    Examples:
        >>> build_band(scene.red).common_name
        'red'
    """
    facts = variable.gs.spectral
    return Band.create(
        name=str(variable.name),
        common_name=None if facts is None else facts.common_name,
        center_wavelength=None if facts is None else facts.center_wavelength,
        full_width_half_max=None if facts is None else facts.full_width_half_max,
        description=variable.attrs.get("long_name"),
    )


def stamp(variable: xr.DataArray, band: Band) -> xr.DataArray:
    """Stamp what an EO band states onto the variable it belongs to.

    Args:
        variable: Pixel-free variable.
        band: Its band.

    Returns:
        The variable carrying the band's spectral facts, and its description
        as `long_name`.

    Examples:
        >>> stamp(variable, Band.create(name="B04", common_name="red")).attrs
        {'common_name': 'red'}
    """
    # PySTAC's getters raise where a band leaves a field out.
    stated = band.properties
    found = attrs.Spectral(
        common_name=stated.get("common_name"),
        center_wavelength=stated.get("center_wavelength"),
        full_width_half_max=stated.get("full_width_half_max"),
    ).to_attrs()
    if stated.get("description") is not None:
        found["long_name"] = stated["description"]
    return variable.assign_attrs(found)


def write_asset(asset: pystac.Asset, bands: list[Band]) -> None:
    """Write spectral bands and declare the EO schema on the owner."""
    EOExtension.ext(asset, add_if_missing=True).bands = bands


def stamp_fields(variable: xr.DataArray, fields: dict) -> xr.DataArray:
    """Read either core or extension band spelling."""
    return stamp(
        variable,
        Band({key.removeprefix("eo:"): value for key, value in fields.items()}),
    )
