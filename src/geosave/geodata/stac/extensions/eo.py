"""EO extension: what a band is called and where in the spectrum it sits."""

from __future__ import annotations

from typing import TypedDict


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


class SpectralAttrs(TypedDict, total=False):
    """Spectral and descriptive attrs translated from one EO band."""

    common_name: str
    center_wavelength: float
    full_width_half_max: float
    long_name: str


def read_attrs(band: Band) -> SpectralAttrs:
    """Translate an EO band's spectral facts and description into attrs."""
    stated = band.properties
    facts = attrs.Spectral(
        common_name=stated.get("common_name"),
        center_wavelength=stated.get("center_wavelength"),
        full_width_half_max=stated.get("full_width_half_max"),
    )
    fields: SpectralAttrs = {}
    if facts.common_name is not None:
        fields["common_name"] = facts.common_name
    if facts.center_wavelength is not None:
        fields["center_wavelength"] = facts.center_wavelength
    if facts.full_width_half_max is not None:
        fields["full_width_half_max"] = facts.full_width_half_max
    if stated.get("description") is not None:
        fields["long_name"] = stated["description"]
    return fields


def write_asset(asset: pystac.Asset, bands: list[Band]) -> None:
    """Write spectral bands and declare the EO schema on the owner."""
    EOExtension.ext(asset, add_if_missing=True).bands = bands
