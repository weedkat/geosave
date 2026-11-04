"""Build a saved file's STAC asset from its path and its actual raster metadata."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import pystac

if TYPE_CHECKING:
    import xarray as xr

    from geosave.geodata.io.storage import Location
from geosave.geodata.io.storage import absolute_location

# What STAC calls a saved raster, by its suffix.
_MEDIA_TYPES = {
    ".tif": pystac.MediaType.GEOTIFF,
    ".tiff": pystac.MediaType.GEOTIFF,
    ".jp2": pystac.MediaType.JPEG2000,
    ".png": pystac.MediaType.PNG,
    ".zarr": pystac.MediaType.ZARR,
    ".nc": pystac.MediaType.NETCDF,
    ".nc4": pystac.MediaType.NETCDF,
    ".cdf": pystac.MediaType.NETCDF,
}


def build_asset(
    path: Location, raster: xr.Dataset, *, owner: pystac.Item
) -> pystac.Asset:
    """Build the STAC data Asset of a saved raster.

    Args:
        path: Location of the saved file.
        raster: Actual opened Dataset; pixels are not read.
        owner: Item on which the metadata extensions declare their schemas.
            The caller adds the returned Asset under its chosen key.

    Returns:
        Asset with an absolute href, media type, grid and ordered band metadata.
    """
    from geosave.geodata.stac import band
    from geosave.geodata.stac.extensions import projection

    href = absolute_location(path)
    media_type = _MEDIA_TYPES.get(PurePosixPath(href).suffix.lower())
    if media_type is None:
        raise ValueError(
            f"{href} names no raster file an Item lists; save the rasters with "
            f"to_cog, to_zarr or to_netcdf and pass the paths they returned"
        )

    asset = pystac.Asset(href, media_type=media_type, roles=["data"])
    asset.set_owner(owner)

    projection.write_asset(asset, raster.gs.geobox)
    band.write_band_fields(asset, raster)
    return asset
