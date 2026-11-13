"""EO extension: what a band is called and where in the spectrum it sits."""

from __future__ import annotations

from typing import TypedDict

import pystac
import xarray as xr
from pystac.extensions.eo import Band, EOExtension

import geosave.geodata.attrs as attrs
from geosave.geodata.stac.stamps import LoadSource, register_stamp


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
    facts = attrs.EOBand(
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


def read_item_attrs(item: pystac.Item, name: str) -> SpectralAttrs:
    """Translate the EO fields of the band ODC loads as `name` into attrs."""
    from geosave.geodata.stac.band import read_band  # band imports this module

    fields = {}
    for key, value in read_band(item, name).items():
        fields[key.removeprefix("eo:")] = value
    return read_attrs(Band(fields))


@register_stamp("eo")
def stamp_eo(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Write each loaded band's spectral facts and description.

    The variable's own attrs win over STAC's. Attrs are kept only when every
    Item that states them agrees.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.

    Returns:
        `loaded` with `common_name`, `center_wavelength`, `full_width_half_max`
        and `long_name` stated.
    """
    stamped = {}
    for name, variable in loaded.data_vars.items():
        observations = []
        for item in source.items:
            observations.append(xr.Dataset(attrs=read_item_attrs(item, str(name))))
        stated = xr.merge(observations, combine_attrs="drop_conflicts").attrs

        stamped[name] = variable.assign_attrs({**stated, **variable.attrs})
    return loaded.assign(stamped)


def read_colour(item: pystac.Item, name: str) -> str | None:
    """Read the colour name the Item states for the band ODC loads as `name`."""
    from geosave.geodata.stac.band import read_band  # band imports this module

    fields = read_band(item, name)
    return fields.get("eo:common_name", fields.get("common_name"))


@register_stamp("rgb")
def stamp_rgb(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """State the RGB composite the bands' common names give.

    The first band naming each colour is used. Nothing is written unless red,
    green and blue are all named.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.

    Returns:
        `loaded` with `Display.rgb` stated, or unchanged without all three colours.
    """
    names_by_colour: dict[str, str] = {}
    for name in loaded.data_vars:
        colours = set()
        for item in source.items:
            colour = read_colour(item, str(name))
            if colour is not None:
                colours.add(colour)

        # Items that disagree on a band's colour state none.
        if len(colours) != 1:
            continue
        colour = colours.pop()
        if colour in ("red", "green", "blue"):
            names_by_colour.setdefault(colour, str(name))
    if len(names_by_colour) < 3:
        return loaded

    return loaded.gs.write_rgb(
        names_by_colour["red"], names_by_colour["green"], names_by_colour["blue"]
    )
