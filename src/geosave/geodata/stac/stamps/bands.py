"""Stamps that write each loaded band's metadata, one per STAC schema.

Every stamp reads the band ODC loaded under a variable's name from each Item
(`read_band`) and keeps what all Items state alike. The variable's own attrs,
as ODC set them, win over STAC's.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import odc.stac
import pystac
import xarray as xr
from pystac.extensions.eo import Band
from pystac.extensions.raster import RasterBand

import geosave.geodata.attrs as attrs
from geosave.geodata.stac.extensions import classification, eo, raster

from .registry import register_stamp
from .source import LoadSource

type BandReader = Callable[[pystac.Item, str], Mapping[str, Any]]

# Fields STAC 1.1 lets an asset state once for all its bands.
_ASSET_FIELDS = ("nodata", "data_type", "unit", "statistics")
_ASSET_PREFIXES = ("raster:", "eo:", "classification:")
_BAND_LISTS = ("raster:bands", "eo:bands", "bands")


def find_asset(item: pystac.Item, name: str) -> tuple[str, int] | None:
    """Find the asset and zero-based band index ODC loads as `name`.

    Names resolve through odc-stac's parser, so `red` finds asset `B04` as the
    loader did.
    """
    parsed = odc.stac.parse_item(item)
    try:
        key, position = parsed.collection.band_key(name)
    except ValueError:
        return None
    if key not in item.assets or (key, position) not in parsed.bands:
        return None
    return key, position - 1


def read_band(item: pystac.Item, name: str) -> dict[str, Any]:
    """Read the STAC fields of the band ODC loads as `name`, in STAC spelling.

    Fields stated on the asset apply to every band of it; the band's own entry
    in `raster:bands`, `eo:bands` or `bands` overrides them. A name that does
    not resolve reads as empty.
    """
    found = find_asset(item, name)
    if found is None:
        return {}
    key, index = found

    stated = item.assets[key].extra_fields
    fields: dict[str, Any] = {}
    for field, value in stated.items():
        if field in _BAND_LISTS:
            continue
        if field in _ASSET_FIELDS or field.startswith(_ASSET_PREFIXES):
            fields[field] = value

    for listing in _BAND_LISTS:
        entries = stated.get(listing, [])
        if index < len(entries):
            fields.update(entries[index])
    return fields


def write_band_attrs(
    loaded: xr.Dataset, source: LoadSource, read: BandReader
) -> xr.Dataset:
    """Write, per loaded variable, the attrs every Item states alike for its band.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.
        read: Reads the attrs one Item states for the band loaded as a name.

    Returns:
        `loaded` with the agreed attrs added; the variable's own attrs win.
    """
    stamped = {}
    for name, variable in loaded.data_vars.items():
        stated = []
        for item in source.items:
            stated.append(xr.Dataset(attrs=dict(read(item, str(name)))))
        agreed = xr.merge(stated, combine_attrs="drop_conflicts").attrs

        stamped[name] = variable.assign_attrs({**agreed, **variable.attrs})
    return loaded.assign(stamped)


def read_packing(item: pystac.Item, name: str) -> raster.RasterAttrs:
    """Read the CF packing and units an Item states for the band loaded as `name`."""
    fields = {}
    for key, value in read_band(item, name).items():
        fields[key.removeprefix("raster:")] = value
    return raster.read_attrs(RasterBand(fields))


def check_packing(name: str, items: Sequence[pystac.Item]) -> None:
    """Refuse Items that pack one band with different scale or offset.

    Fused pixels can share one scale and offset only if every source does.

    Raises:
        ValueError: Two Items state different `(scale, offset)` for `name`.
    """
    expected = None
    for item in items:
        packing = attrs.CFPacking.model_validate(read_packing(item, name))

        # Omitted scale/offset mean identity. Nodata is excluded: ODC remaps
        # source sentinels on loading.
        scale = (packing.scale_factor, packing.add_offset)
        if expected is not None and scale != expected:
            raise ValueError(
                f"band {name!r} has incompatible source scale/offset: "
                f"{expected} and {scale} in Item {item.id!r}; "
                "load and decode the sources separately before combining them"
            )
        expected = scale


@register_stamp("raster")
def stamp_raster(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Write each band's packing and units, and declare the fill ODC cast.

    Raises:
        ValueError: Items pack a loaded band with different scale or offset.
    """
    for name in loaded.data_vars:
        check_packing(str(name), source.items)

    stamped = write_band_attrs(loaded, source, read_packing)

    # The declaration must match the fill ODC cast into the output dtype.
    declared = {}
    for name, variable in stamped.data_vars.items():
        fill = variable.attrs.pop("nodata", variable.attrs.get("_FillValue"))
        if fill is not None:
            variable = variable.rio.write_nodata(fill)
        declared[name] = variable
    return stamped.assign(declared)


def read_spectrum(item: pystac.Item, name: str) -> eo.SpectralAttrs:
    """Read the spectral facts and description an Item states for a band."""
    fields = {}
    for key, value in read_band(item, name).items():
        fields[key.removeprefix("eo:")] = value
    return eo.read_attrs(Band(fields))


@register_stamp("eo")
def stamp_eo(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Write each band's common name, wavelength, width and description."""
    return write_band_attrs(loaded, source, read_spectrum)


def read_legend(item: pystac.Item, name: str) -> classification.ClassificationAttrs:
    """Read the classes an Item states for a band, as CF flags and colours."""
    return classification.read_attrs(RasterBand(dict(read_band(item, name))))


@register_stamp("classification")
def stamp_classification(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Write each band's class values, names and colour hints."""
    return write_band_attrs(loaded, source, read_legend)


def read_title(item: pystac.Item, name: str) -> dict[str, str]:
    """Read the asset title naming a band, as `long_name`.

    Only a one-band asset's title names its band, and only when the band has
    no description of its own, which `eo` writes instead.
    """
    found = find_asset(item, name)
    if found is None:
        return {}
    asset = item.assets[found[0]]

    count = 0
    for listing in ("bands", "eo:bands", "raster:bands"):
        count = max(count, len(asset.extra_fields.get(listing, [])))
    if count > 1 or "description" in read_band(item, name) or not asset.title:
        return {}
    return {"long_name": asset.title}


@register_stamp("title")
def stamp_title(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Name each band after its asset's title, where the asset holds only that band."""
    return write_band_attrs(loaded, source, read_title)


def read_colour(item: pystac.Item, name: str) -> str | None:
    """Read the common name an Item states for a band, such as `red`."""
    fields = read_band(item, name)
    return fields.get("eo:common_name", fields.get("common_name"))


@register_stamp("rgb")
def stamp_rgb(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """State the true-colour composite the bands' common names give.

    The first band named for each colour is used; a band the Items name
    differently names none. Nothing is written without red, green and blue.
    """
    names_by_colour: dict[str, str] = {}
    for name in loaded.data_vars:
        colours = set()
        for item in source.items:
            colour = read_colour(item, str(name))
            if colour is not None:
                colours.add(colour)

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
