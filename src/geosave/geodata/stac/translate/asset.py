"""A STAC Asset as one raster file and the variables its bands load into."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

import pystac
import xarray as xr
from pystac.extensions.eo import AssetEOExtension, Band
from pystac.extensions.raster import AssetRasterExtension, RasterBand

from geosave.geodata.stac import extensions

from . import band

# Asset fields that list bands, in the spelling each STAC generation used.
_BAND_LISTINGS = ("bands", "raster:bands", "eo:bands")

# What STAC calls a saved raster, by its suffix.
_MEDIA_TYPES = {
    ".tif": pystac.MediaType.GEOTIFF,
    ".tiff": pystac.MediaType.GEOTIFF,
    ".jp2": pystac.MediaType.JPEG2000,
    ".png": pystac.MediaType.PNG,
}


def bands(asset: pystac.Asset) -> list[tuple[RasterBand, Band]]:
    """Return an asset's bands as PySTAC's Raster and EO bands, in file order.

    Args:
        asset: Asset listing its bands as `raster:bands` and `eo:bands`, or
            as the STAC 1.1 `bands` PySTAC has no class for.

    Returns:
        One (raster band, EO band) pair per band either listing reaches,
        without the nulls Parquet fills absent keys with. Empty where the
        asset lists no bands.

    Examples:
        >>> stored, spectral = bands(asset)[0]
        >>> stored.scale, spectral.common_name
        (0.0001, 'red')
    """
    core = asset.extra_fields.get("bands")
    if core is not None:
        pairs = [_split(entry) for entry in core]
    else:
        stored = AssetRasterExtension(asset).bands or []
        spectral = AssetEOExtension(asset).bands or []
        pairs = [
            (
                stored[index].properties if index < len(stored) else {},
                spectral[index].properties if index < len(spectral) else {},
            )
            for index in range(max(len(stored), len(spectral)))
        ]
    return [
        (RasterBand(_stated(stored)), Band(_stated(named))) for stored, named in pairs
    ]


def names(key: str, asset: pystac.Asset) -> tuple[str, ...]:
    """Name the variables an asset loads into, as odc-stac names them.

    Args:
        key: The asset's key in its Item.
        asset: The asset.

    Returns:
        `(key,)` for an asset of at most one band, else `key.1` to `key.n`.

    Examples:
        >>> names("visual", item.assets["visual"])
        ('visual.1', 'visual.2', 'visual.3')
    """
    count = len(bands(asset))
    if count <= 1:
        return (key,)
    return tuple(f"{key}.{index}" for index in range(1, count + 1))


def attrs(key: str, asset: pystac.Asset) -> dict[str, dict[str, Any]]:
    """Return the CF attrs each variable of an asset takes from its bands.

    Args:
        key: The asset's key in its Item.
        asset: The asset.

    Returns:
        Variable name, as `names` names it, mapped to the attrs its band
        states. Variables stating none are absent.

    Examples:
        >>> attrs("B04", item.assets["B04"])
        {'B04': {'scale_factor': 0.0001, 'common_name': 'red'}}
    """
    found = {}
    for name, pair in zip(names(key, asset), bands(asset)):
        if stated := band.read(*pair):
            found[name] = stated
    return found


def fields(asset: pystac.Asset, *, index: int = 1) -> dict[str, object]:
    """Read one asset's fields, merging the band description it nests.

    Args:
        asset: Asset to read.
        index: One-based source band index.

    Returns:
        {
            "<field>": its value,
        }
        The asset's own fields without its band listings, overlaid with the
        selected band as the Raster and EO extensions spell it, as `"scale"`.

    Examples:
        >>> fields(asset)
        {'gsd': 10, 'unit': '1', 'scale': 0.0001, 'name': 'B04', 'common_name': 'red'}
    """
    found = {
        key: value
        for key, value in asset.extra_fields.items()
        if key not in _BAND_LISTINGS
    }
    listed = bands(asset)
    if len(listed) >= index:
        stored, spectral = listed[index - 1]
        found.update(stored.to_dict())
        found.update(spectral.to_dict())
    return found


def write(item: pystac.Item, key: str, href: str, raster: xr.Dataset) -> None:
    """Add one raster file to an Item as a data asset describing its bands.

    Args:
        item: Item to add the asset to.
        key: Asset key.
        href: Absolute path or URL of the file.
        raster: The file as `open_raster` opens it. No pixel is read.

    Raises:
        ValueError: `href` names no raster file, a class name is refused by
            the Classification schema, or the raster has no locatable grid.

    Examples:
        >>> write(item, "label", "/data/s0/label.tif", label)
        >>> item.assets["label"].roles
        ['data']
    """
    # An extension declares its schema on the asset's owner, so add first.
    item.add_asset(key, pystac.Asset(href, media_type=_media_type(href), roles=["data"]))
    for extension in extensions.ASSET:
        extension.write(item.assets[key], raster)


def _media_type(href: str) -> str:
    """Name the media type of the raster file an href points at."""
    found = _MEDIA_TYPES.get(PurePosixPath(href).suffix.lower())
    if found is None:
        raise ValueError(
            f"{href} names no raster file an Item lists; save the rasters with "
            f"to_cog and pass the files it returned"
        )
    return found


def _split(entry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split one STAC 1.1 band into the Raster and EO bands PySTAC models.

    `raster:` and `eo:` keys lose their prefix; `name` and `description` name
    the EO band; every other key describes the stored values and stays with
    the raster band.
    """
    stored: dict[str, Any] = {}
    spectral: dict[str, Any] = {}
    for key, value in entry.items():
        if key.startswith("eo:"):
            spectral[key.removeprefix("eo:")] = value
        elif key in ("name", "description"):
            spectral[key] = value
        else:
            stored[key.removeprefix("raster:")] = value
    return stored, spectral


def _stated(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the nulls Parquet fills a key other rows carry with."""
    return {key: value for key, value in fields.items() if value is not None}
