"""Publish rasters as STAC Items and assets."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime as DateTime
from typing import TYPE_CHECKING, Any

import pystac

from geosave.geodata import attrs
from geosave.geodata.io.storage import Location, StorageOptions
from geosave.geodata.stac import item


if TYPE_CHECKING:
    from rasterio.io import DatasetReader
    from shapely.geometry.base import BaseGeometry
    from geosave.geodata.stac.extensions.eo import Band
    from geosave.geodata.stac.extensions.raster import RasterBand


def _read_footprint(
    source: DatasetReader, densify: int, precision: int
) -> BaseGeometry:
    """Transform the saved pixel-grid footprint to geographic coordinates."""
    from odc.geo.geobox import GeoBox
    from rasterio.warp import transform_geom
    from shapely.geometry import shape

    grid = GeoBox((source.height, source.width), source.transform, str(source.crs))
    footprint = grid.extent
    if densify:
        length = min(
            abs(source.bounds.right - source.bounds.left),
            abs(source.bounds.top - source.bounds.bottom),
        )
        footprint = footprint.segmented(length / (densify + 1))

    return shape(
        transform_geom(source.crs, "EPSG:4326", footprint.json, precision=precision)
    )


def _read_media_type(source: DatasetReader) -> str | None:
    """Identify GeoTIFF and COG from the saved driver's metadata."""
    if source.driver != "GTiff":
        return None
    if source.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG":
        return pystac.MediaType.COG
    return pystac.MediaType.GEOTIFF


def create_stac_item(
    sources: dict[str, Location],
    *,
    id: str,
    input_datetime: DateTime | None = None,
    properties: dict[str, Any] | None = None,
    geometry: dict[str, Any] | None = None,
    collection: str | None = None,
    collection_url: str | None = None,
    asset_roles: Sequence[str] | None = None,
    asset_media_type: str = "auto",
    asset_hrefs: dict[str, str] | None = None,
    with_proj: bool = False,
    with_raster: bool = False,
    with_eo: bool = False,
    geom_densify_pts: int = 0,
    geom_precision: int = -1,
    storage_options: StorageOptions | None = None,
) -> pystac.Item:
    """Describe named existing raster files as one native STAC Item.

    Args:
        sources: Asset keys mapped to readable local paths or filesystem URLs.
        id: Nonempty Item identifier, unique within its Collection.
        input_datetime: Observation instant. Otherwise properties must contain
            both start_datetime and end_datetime as ISO timestamps.
        properties: Additional STAC Item properties, copied without mutation.
        geometry: Optional longitude/latitude GeoJSON footprint. Otherwise use
            the union of file footprints; no pixels are inspected.
        collection: Optional Collection ID.
        collection_url: Optional URL of the Collection document.
        asset_roles: Roles applied to every asset, e.g. ["data"]. Customize
            individual assets on the returned Item.
        asset_media_type: Media type, or "auto" to recognize GeoTIFF/COG.
        asset_hrefs: Asset keys mapped to absolute advertised href overrides.
        with_proj: Extract each file's grid onto its Asset.
        with_raster: Extract saved dtype, nodata, packing, units and classes per
            band.
        with_eo: Extract known band identities and spectral metadata.
        geom_densify_pts: Extra points per footprint edge before reprojection.
        geom_precision: Decimal places in output geometry; -1 preserves precision.
        storage_options: Filesystem options for reading sources.

    Returns:
        Item with absolute asset hrefs. Additional assets and extensions can be
        added using native PySTAC methods.

    Raises:
        ValueError: Sources, ID, temporal coverage, CRS or hrefs are invalid.
    """
    from contextlib import ExitStack

    import rasterio
    from odc.geo.geobox import GeoBox
    from pystac.utils import is_absolute_href
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    from geosave.geodata.io import storage
    from geosave.geodata.io.rasterio import read_class_attrs
    from geosave.geodata.stac.extensions import (
        classification,
        eo,
        projection,
        raster,
    )

    if not id or not sources:
        raise ValueError("create_stac_item needs a nonempty id and at least one source")
    if geom_densify_pts < 0 or geom_precision < -1:
        raise ValueError(
            "geometry densification must be nonnegative and precision at least -1"
        )
    fields = deepcopy(properties or {})
    stated_datetime = fields.pop("datetime", None)
    if input_datetime is not None and stated_datetime is not None:
        raise ValueError("supply input_datetime or properties datetime, not both")
    time = item.normalize_time(
        input_datetime if input_datetime is not None else stated_datetime,
        fields.pop("start_datetime", None),
        fields.pop("end_datetime", None),
    )

    overrides = dict(asset_hrefs or {})
    if overrides.keys() - sources.keys():
        raise ValueError("asset_hrefs contains keys absent from sources")
    for href in overrides.values():
        if not is_absolute_href(href):
            raise ValueError(f"advertised asset href must be absolute: {href!r}")

    entry = item.create_item(id=id, time=time, properties=fields, collection=collection)
    if collection_url is not None:
        entry.add_link(pystac.Link(pystac.RelType.COLLECTION, collection_url))

    footprints = []
    for key, path in sources.items():
        if not key:
            raise ValueError("asset keys must be nonempty")
        path = storage.location(path, storage_options)
        with ExitStack() as opened:
            address = str(path)
            if not storage.is_local(path):
                address = opened.enter_context(path.open("rb"))
            source = opened.enter_context(rasterio.open(address))
            if source.crs is None:
                raise ValueError(f"source {key!r} has no CRS")

            grid = GeoBox(
                (source.height, source.width), source.transform, str(source.crs)
            )
            footprints.append(_read_footprint(source, geom_densify_pts, geom_precision))
            media_type = asset_media_type
            if media_type == "auto":
                media_type = _read_media_type(source)

            asset = pystac.Asset(
                overrides.get(key, storage.absolute_location(path)),
                media_type=media_type,
                roles=None if asset_roles is None else list(asset_roles),
            )
            entry.add_asset(key, asset)
            if with_proj:
                projection.write_asset(asset, grid)
            if with_raster:
                bands = _read_raster_bands(source)
                for index, band in zip(source.indexes, bands, strict=True):
                    found = read_class_attrs(source, index)
                    classification.write_band(
                        asset,
                        band,
                        attrs.CFFlags.from_attrs(found),
                        attrs.Display.from_attrs(found),
                    )
                raster.write_asset(asset, bands)
            if with_eo:
                identities = _read_eo_bands(source)
                if identities:
                    eo.write_asset(asset, identities)

    footprint = (
        shape(dict(geometry)) if geometry is not None else unary_union(footprints)
    )
    if footprint.is_empty or not footprint.is_valid:
        raise ValueError("Item footprint must be nonempty and valid")
    entry.geometry = mapping(footprint)
    entry.bbox = list(footprint.bounds)
    return entry


def _read_raster_bands(source: DatasetReader) -> list[RasterBand]:
    """Read stored band metadata from an open rasterio dataset, without pixels."""
    from math import isfinite
    from geosave.geodata.stac.extensions.raster import (
        DataType,
        NoDataStrings,
        RasterBand,
    )

    bands = []
    for index in source.indexes:
        position = index - 1
        nodata = source.nodatavals[position]
        if nodata is not None and not isfinite(nodata):
            nodata = NoDataStrings(str(nodata))

        bands.append(
            RasterBand.create(
                data_type=DataType(source.dtypes[position]),
                nodata=nodata,
                scale=source.scales[position],
                offset=source.offsets[position],
                unit=source.units[position],
            )
        )
    return bands


def _read_eo_bands(source: DatasetReader) -> list[Band]:
    """Read known band identities and spectral tags without estimating them.

    Returns:
        One band per file band, or none when a band states no name, since a
        loader then names it by its asset key and position.
    """
    from geosave.geodata.stac.extensions.eo import Band

    bands = []
    for index in source.indexes:
        tags = source.tags(index)
        facts = attrs.EOBand.from_attrs(tags)
        name = tags.get("variable_name") or source.descriptions[index - 1]
        if name is None:
            return []
        bands.append(
            Band.create(
                name=name,
                description=source.descriptions[index - 1],
                common_name=None if facts is None else facts.common_name,
                center_wavelength=None if facts is None else facts.center_wavelength,
                full_width_half_max=None
                if facts is None
                else facts.full_width_half_max,
            )
        )
    return bands
