"""Build native STAC Items from a region and the rasters of its assets."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any, Self

import pystac
import xarray as xr
from pystac.utils import datetime_to_str

from geosave.geodata.core.conventions import END_DATETIME, START_DATETIME
from geosave.geodata.stac.extensions import eo, projection
from geosave.geodata.stac.extensions import raster as raster_extension

if TYPE_CHECKING:
    from geosave.geodata.core.region import GeoRegion


class StacItem(pystac.Item):
    """A pystac Item built from a region, with assets added from rasters.

    It is a `pystac.Item` everywhere: `clone`, `from_dict`, extensions and
    validation keep working, and `add_asset` adds any other asset.
    `RasterStac.to_cog` builds one per staged Item.

    Examples:
        >>> item = StacItem.from_region(scene.gs.region, id="20250601")
        >>> item.add_raster_asset("red", red, "s3://bucket/s2/20250601/red.tif")
        <Asset href=s3://bucket/s2/20250601/red.tif>
    """

    @classmethod
    def from_region(
        cls,
        region: GeoRegion,
        *,
        id: str,
        properties: dict[str, Any] | None = None,
    ) -> Self:
        """Create the Item covering a region, without assets.

        The region's extent, in WGS84, is the footprint. An instant becomes
        the Item datetime; a longer timespan becomes `start_datetime` and
        `end_datetime`, with no datetime.

        Args:
            region: Extent and timespan, as `.gs.region` of a raster or of a
                staged Item node gives them.
            id: Item id.
            properties: STAC properties to state beside the time.

        Returns:
            Item without assets; add them with `add_raster_asset`.

        Raises:
            ValueError: The region is timeless.

        Examples:
            >>> StacItem.from_region(scene.gs.region, id="20250601").datetime
            datetime.datetime(2025, 6, 1, 10, 30, tzinfo=datetime.timezone.utc)
        """
        if region.timespan is None:
            raise ValueError(f"Item {id!r} needs a region with a timespan")
        start, end = region.timespan
        start = start.replace(tzinfo=UTC)
        end = end.replace(tzinfo=UTC)

        item_properties = dict(properties or {})
        item_datetime = start
        if start != end:
            item_datetime = None
            item_properties[START_DATETIME] = datetime_to_str(start)
            item_properties[END_DATETIME] = datetime_to_str(end)

        footprint = region.extent.to_crs("EPSG:4326")
        return cls(
            id=id,
            geometry=footprint.json,
            bbox=list(footprint.boundingbox),
            datetime=item_datetime,
            properties=item_properties,
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

        Examples:
            >>> item.add_raster_asset("red", red, "s3://bucket/red.tif").extra_fields["proj:shape"]
            [2, 2]
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
            raster_bands = []
            for variable in raster.data_vars.values():
                raster_bands.append(raster_extension.build_band(variable))
            raster_extension.write_asset(asset, raster_bands)
        if with_eo:
            eo_bands = []
            for variable in raster.data_vars.values():
                eo_bands.append(eo.build_band(variable))
            eo.write_asset(asset, eo_bands)
        return asset
