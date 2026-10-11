"""Geographic metadata shared by written encoder context functions."""

import xarray as xr


def geographic_center(raster: xr.Dataset | xr.DataArray) -> tuple[float, float]:
    """Return latitude and longitude from the raster's grid.

    Args:
        raster: Raster the encoder receives, on its georeferenced grid.

    Returns:
        Geographic centre in latitude/longitude order.

    Raises:
        ValueError: The raster has no georeferenced grid.
    """
    grid = raster.odc.geobox
    if grid is None or grid.crs is None:
        raise ValueError("model location context requires a georeferenced grid")
    longitude, latitude = grid.extent.centroid.to_crs("EPSG:4326").coords[0]
    return float(latitude), float(longitude)
