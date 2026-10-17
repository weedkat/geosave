"""A STAC Asset as the `RasterFile` it lists."""

from __future__ import annotations

from pathlib import PurePosixPath

import pystac

from geosave.geodata.io.raster.file import RasterFile
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


def from_file(file: RasterFile, *, owner: pystac.Item) -> pystac.Asset:
    """Describe a saved raster as a STAC data Asset.

    Args:
        file: Written raster and its header; pixels are not read.
        owner: Item on which the metadata extensions declare their schemas.
            The caller adds the returned Asset under its chosen key.

    Returns:
        Asset with an absolute href, media type, grid and ordered band metadata.
    """
    from . import band
    from geosave.geodata.stac.extensions import projection

    href = absolute_location(file.path)
    media_type = _MEDIA_TYPES.get(PurePosixPath(href).suffix.lower())
    if media_type is None:
        raise ValueError(
            f"{href} names no raster file an Item lists; save the rasters with "
            f"to_cog, to_zarr or to_netcdf and pass the scenes they returned"
        )

    asset = pystac.Asset(href, media_type=media_type, roles=["data"])
    asset.set_owner(owner)

    projection.write_asset(asset, file.header.gs.geobox)
    band.write_metadata(asset, file.header)
    return asset
