"""The CF attrs one STAC load describes: Collection metadata and each band's fields."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import odc.stac
from pystac.extensions.eo import AssetEOExtension, Band
from pystac.extensions.raster import AssetRasterExtension, RasterBand

from geosave_engine import __path__ as _package_paths
from geosave_engine.geodata.warnings import DroppedAttrsWarning

import geosave_engine.geodata.attrs as attrs

if TYPE_CHECKING:
    import pystac
    import xarray as xr


# Asset fields that list bands, in the spelling each STAC generation used.
_BAND_LISTINGS = ("bands", "raster:bands", "eo:bands")


def load_attrs(
    items: Sequence[pystac.Item],
    collection: pystac.Collection,
    loaded: xr.Dataset,
    *,
    stac_cfg: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Return the loaded Dataset carrying the attrs its Collection and assets state.

    The Collection describes the whole raster, so it lands on the root as
    ACDD keys; each asset describes the variable it loads into, where only
    what every item publishes alike is kept. The loader's own attrs describe
    the pixels it produced, so they win.

    Args:
        items: Items making up one load, in search order.
        collection: Collection the items belong to.
        loaded: Dataset odc-stac loaded from them.
        stac_cfg: The same ODC conversion settings used for loading.

    Returns:
        The Dataset, its root and variables carrying their attrs.

    Raises:
        ValueError: `items` is empty.

    Warns:
        DroppedAttrsWarning: The items publish a field differently, or only
            some of them publish it.
    """
    if not items:
        raise ValueError("reading a STAC load's attrs needs at least one item")

    parsed = list(odc.stac.parse_items(items, cfg=stac_cfg))
    providers = collection.providers or []
    root: dict[str, Any] = {
        "id": collection.id,
        "title": collection.title,
        "summary": collection.description,
        "keywords": ", ".join(collection.keywords) if collection.keywords else None,
        "institution": providers[0].name if providers else None,
        "license": collection.license,
    }

    described = {}
    for name, variable in loaded.data_vars.items():
        published = []
        for item, entry in zip(items, parsed, strict=True):
            asset_name, band_index = entry.collection.band_key(str(name))
            asset = item.assets.get(asset_name)
            bands = read_bands(asset) if asset else []
            # An item may lack the asset, or list fewer bands than the file holds.
            if len(bands) >= band_index:
                published.append(band_attrs(*bands[band_index - 1]))
            else:
                published.append({})
        kept = attrs.shared(published)
        disputed = sorted({key for fields in published for key in fields} - set(kept))
        if disputed:
            values = {
                key: {
                    item.id: item_attrs.get(key)
                    for item, item_attrs in zip(items, published, strict=True)
                }
                for key in disputed
            }
            warnings.warn(
                f"STAC items publish {disputed} differently for {str(name)!r}, so "
                f"those source attrs are dropped: {values}",
                DroppedAttrsWarning,
                skip_file_prefixes=tuple(_package_paths),
            )
        stated = variable.copy(deep=False)
        stated.attrs = {**kept, **variable.attrs}
        # odc states its fill as `nodata`; written once, CF's spelling carries it.
        described[name] = attrs.stamp(
            stated, attrs.Nodata, attrs.parse(stated, attrs.Nodata)
        )
    result = loaded.assign(described)
    result.attrs = {**loaded.attrs, **{k: v for k, v in root.items() if v is not None}}
    return result


def read_asset_fields(asset: pystac.Asset, *, band_index: int = 1) -> dict[str, object]:
    """Read one asset's fields, merging the band description it nests.

    Args:
        asset: Asset to read.
        band_index: One-based source band index.

    Returns:
        {
            "<field>": its value,
        }
        The asset's own fields without its band listings, overlaid with the
        selected band as the Raster and EO extensions spell it, as `"scale"`.

    Examples:
        >>> read_asset_fields(asset)
        {'gsd': 10, 'unit': '1', 'scale': 0.0001, 'name': 'B04', 'common_name': 'red'}
    """
    found = {
        key: value
        for key, value in asset.extra_fields.items()
        if key not in _BAND_LISTINGS
    }
    bands = read_bands(asset)
    if len(bands) >= band_index:
        stored, spectral = bands[band_index - 1]
        found.update(stored.to_dict())
        found.update(spectral.to_dict())
    return found


def read_bands(asset: pystac.Asset) -> list[tuple[RasterBand, Band]]:
    """Return an asset's bands as PySTAC's Raster and EO bands, in file order.

    Args:
        asset: Asset listing its bands as `raster:bands` and `eo:bands`, or
            as the STAC 1.1 `bands` PySTAC has no class for.

    Returns:
        One (raster band, EO band) pair per band either listing reaches,
        without the nulls Parquet fills absent keys with. Empty where the
        asset lists no bands.

    Examples:
        >>> stored, spectral = read_bands(asset)[0]
        >>> stored.scale, spectral.common_name
        (0.0001, 'red')
    """
    core = asset.extra_fields.get("bands")
    if core is not None:
        pairs = [_legacy(band) for band in core]
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


def _legacy(band: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split one STAC 1.1 band into the Raster and EO bands PySTAC models.

    Args:
        band: One entry of an asset's `bands`.

    Returns:
        (raster fields, EO fields). `raster:` and `eo:` keys lose their
        prefix; `name` and `description` name the EO band; every other key
        describes the stored values and stays with the raster band.
    """
    stored: dict[str, Any] = {}
    spectral: dict[str, Any] = {}
    for key, value in band.items():
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


def band_attrs(stored: RasterBand, spectral: Band) -> dict[str, Any]:
    """Translate one band into the CF attrs it states.

    `nodata` and `data_type` describe the file, and the loader that opens it
    states them, so neither is translated.

    Args:
        stored: The band as the Raster extension describes it.
        spectral: The band as the EO extension describes it.

    Returns:
        The attrs the Raster, EO and Classification modules read from the
        band; empty where it states none.

    Examples:
        >>> band_attrs(RasterBand.create(unit="1", scale=0.0001), Band({}))
        {'scale_factor': 0.0001, 'units': '1'}
    """
    # Imported here: the STAC package imports this module while it loads.
    from geosave_engine.geodata.stac.extensions import classification, eo, raster

    return {**raster.read(stored), **eo.read(spectral), **classification.read(stored)}
