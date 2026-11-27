"""Derive display titles and RGB band choices from loaded STAC assets."""

from collections.abc import Sequence

import pystac
import xarray as xr
from odc.stac import ConversionConfig

from .bands import BAND_LISTS, find_band_sources, read_band_fields


def stamp_title(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> xr.Dataset:
    """Name a one-band asset after its title when its band has no description."""
    stamped = {}
    for variable, sources in find_band_sources(ds, items, stac_cfg):
        titles = set()
        for _, asset, index in sources:
            if not asset.title:
                continue

            count = max(len(asset.extra_fields.get(key, [])) for key in BAND_LISTS)
            fields = read_band_fields(asset, index)
            if count <= 1 and "description" not in fields:
                titles.add(asset.title)

        if len(titles) == 1 and "long_name" not in variable.attrs:
            stamped[variable.name] = variable.assign_attrs(long_name=titles.pop())

    return ds.assign(stamped)


def stamp_rgb(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> xr.Dataset:
    """State the true-colour composite the bands' STAC common names identify.

    The first band for each colour is used. A band with conflicting colour
    declarations is skipped, and a composite requires red, green and blue.
    """
    names_by_colour = {}
    for variable, sources in find_band_sources(ds, items, stac_cfg):
        colours = set()
        for _, asset, index in sources:
            fields = read_band_fields(asset, index)
            colour = fields.get("eo:common_name", fields.get("common_name"))
            if colour is not None:
                colours.add(colour)

        if len(colours) != 1:
            continue
        colour = colours.pop()
        if colour in ("red", "green", "blue"):
            names_by_colour.setdefault(colour, str(variable.name))

    if len(names_by_colour) < 3:
        return ds

    return ds.gs.write_rgb(
        names_by_colour["red"], names_by_colour["green"], names_by_colour["blue"]
    )
