"""Describe a STAC tree's Items and Collections as pystac objects."""

from __future__ import annotations

from typing import Any, Self

import pystac
import xarray as xr

import geosave.geodata.attrs as attrs

from geosave.geodata.stac.extensions import eo, projection
from geosave.geodata.stac.extensions import raster as raster_extension


class StacItem(pystac.Item):
    """A pystac Item built from a STAC tree's Item node, with assets added from rasters.

    It is a `pystac.Item` everywhere: `clone`, `from_dict`, extensions and
    validation keep working, and `add_asset` adds any other asset.

    Examples:
        >>> item = StacItem.from_datatree(tree["s2/20250601T000000_33UUU_..."], collection="s2")
        >>> item.add_raster_asset("red", red, "s3://bucket/s2/.../red.tif")
        <Asset href=s3://bucket/s2/.../red.tif>
    """

    @classmethod
    def from_datatree(
        cls,
        item: xr.DataTree,
        *,
        id: str | None = None,
        collection: str | None = None,
        properties: dict[str, Any] | None = None,
        geometry: dict[str, Any] | None = None,
        stac_extensions: list[str] | None = None,
    ) -> Self:
        """Create the Item an Item node states, without assets.

        Args:
            item: Item node of a STAC tree.
            id: Item id; None takes the node's name.
            collection: Id of the Collection the Item belongs to.
            properties: Item properties, over the node's marked property coordinates.
            geometry: GeoJSON footprint in WGS84; None takes the union of the
                node's assets' grids.
            stac_extensions: Schema URIs to declare beside those its assets add.

        Returns:
            Item with the node's time, properties and footprint; add assets
            with `add_raster_asset`.

        Raises:
            NotImplementedError: Always, while publication is rebuilt.
        """
        raise NotImplementedError(
            "STAC publication is being rebuilt on coordinates marked stac_property"
        )

    def add_raster_asset(
        self,
        key: str,
        raster: xr.Dataset,
        href: str,
        *,
        roles: list[str] | None = None,
        media_type: str | None = None,
        title: str | None = None,
        with_proj: bool = True,
        with_raster: bool = True,
        with_eo: bool = False,
    ) -> pystac.Asset:
        """Add the asset of one file holding `raster`'s variables, one band each.

        Args:
            key: Asset key.
            raster: The file's Dataset, on its own grid.
            href: Where the file is.
            roles: Asset roles; None is `["data"]`.
            media_type: Media type; None is a Cloud Optimized GeoTIFF.
            title: Human-readable title; None takes the `long_name` of a
                one-variable raster.
            with_proj: State the grid as `proj:*`.
            with_raster: State storage type, fill, packing and units as `raster:bands`.
            with_eo: State each band's spectrum as `eo:bands`, from `EOBand` attrs.

        Returns:
            The asset; the Item declares each extension it uses.
        """
        if title is None and len(raster.data_vars) == 1:
            title = next(iter(raster.data_vars.values())).attrs.get("long_name")
        asset = pystac.Asset(
            href,
            title=title,
            media_type=media_type or pystac.MediaType.COG,
            roles=roles or ["data"],
        )
        # An extension declares its schema on the Item an asset belongs to.
        self.add_asset(key, asset)

        if with_proj:
            projection.write_asset(asset, raster.odc.geobox)
        if with_raster:
            bands = []
            for variable in raster.data_vars.values():
                bands.append(raster_extension.build_band(variable))
            raster_extension.write_asset(asset, bands)
        if with_eo:
            spectra = []
            for variable in raster.data_vars.values():
                spectra.append(eo.build_band(variable))
            eo.write_asset(asset, spectra)
        return asset


def create_stac_collection(
    collection: xr.DataTree, items: list[pystac.Item]
) -> pystac.Collection:
    """Create the Collection a collection node states, or a minimal one over its Items.

    Args:
        collection: Collection node of a STAC tree; its name is the Collection id.
        items: The Collection's Items, whose extents a minimal Collection spans.

    Returns:
        The stated Collection, as its `StacCollection` attrs hold it, else one
        whose extent covers `items`.
    """
    stated = attrs.StacCollection.from_attrs(collection.attrs)
    if stated is not None and stated.collection is not None:
        return pystac.Collection.from_dict(stated.collection)

    return pystac.Collection(
        id=str(collection.name),
        description=f"Items of {collection.name}",
        extent=pystac.Extent.from_items(items),
    )
