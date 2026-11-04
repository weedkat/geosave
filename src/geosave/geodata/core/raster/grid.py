"""Place rasters on grids, name grids, and write CF axis metadata."""

from __future__ import annotations

import odc.geo.xr  # noqa: F401 registers .odc
import xarray as xr
from odc.geo.gcp import GCPGeoBox
from odc.geo import SomeCRS
from odc.geo.geobox import GeoBox

from geosave.geodata.conventions import SPATIAL_DIMENSIONS

def write_cf_axes[T: xr.DataArray | xr.Dataset](raster: T, geobox: GeoBox) -> T:
    """Write onto the spatial coordinates odc's attrs and what CF says they measure.

    Args:
        raster: DataArray or Dataset whose `y` and `x` coordinates take the attrs.
        geobox: Grid whose CRS determines coordinate units and CF axis names.

    Returns:
        New raster whose spatial coordinates carry the attrs. Coordinate values,
        pixels, and laziness are unchanged; this does not reproject the data.

    Raises:
        ValueError: The grid carries no CRS.
    """
    crs = geobox.crs
    if crs is None:
        raise ValueError(
            "geobox carries no CRS, so its axes measure pixels rather than "
            "ground position; assign one before writing their axes"
        )
    names = (
        ("latitude", "longitude")
        if crs.geographic
        else ("projection_y_coordinate", "projection_x_coordinate")
    )
    from odc.geo.xr import xr_coords

    native = xr_coords(geobox, always_yx=True)
    coords_with_axes = {}
    for dim, name, axis in zip(SPATIAL_DIMENSIONS, names, ("Y", "X"), strict=True):
        coord = raster[dim].copy()
        coord.attrs = {**native[dim].attrs, "standard_name": name, "axis": axis}
        coords_with_axes[dim] = coord
    return raster.assign_coords(coords_with_axes)


def place_on_grid[T: xr.DataArray | xr.Dataset](raster: T, geobox: GeoBox | GCPGeoBox) -> T:
    """Give a raster's `y` and `x` the grid's coordinates and CRS.

    Args:
        raster: DataArray or Dataset whose dimensions end in `y, x`, sized as
            the grid. Coordinates it already has on `y`, `x` and its CRS
            coordinate are replaced; every other coordinate is kept.
        geobox: Regular grid, or a grid placed by ground control points.

    Returns:
        New raster whose variables point at the CRS coordinate with
        `grid_mapping`, its axes carrying CF attrs on a regular grid. Pixels
        and laziness are unchanged.

    Raises:
        ValueError: A variable's dimensions do not end in `y, x`, their sizes
            differ from the grid's, or the grid carries no CRS.
    """
    from odc.geo.xr import xr_coords

    if geobox.crs is None:
        raise ValueError("a grid without a CRS places no pixel on the ground")
    variables = [raster] if isinstance(raster, xr.DataArray) else list(raster.data_vars.values())
    for variable in variables:
        if variable.dims[-2:] != SPATIAL_DIMENSIONS:
            raise ValueError(
                f"{variable.name!r}'s dimensions {variable.dims} must end in {SPATIAL_DIMENSIONS}"
            )
    sizes = (raster.sizes["y"], raster.sizes["x"])
    if sizes != tuple(geobox.shape):
        raise ValueError(f"the raster is {sizes} pixels (y, x) and the grid {tuple(geobox.shape)}")

    carriers = []
    for name, coord in raster.coords.items():
        if _is_grid_mapping(coord):
            carriers.append(name)
    grid = xr_coords(geobox, always_yx=True)
    placed = raster.drop_vars(carriers).assign_coords(grid)
    # odc points each variable at the CRS; the grid's own CRS coordinate goes
    # back on so attrs odc does not write, such as GCPs, survive.
    placed = placed.odc.assign_crs(geobox.crs).assign_coords(spatial_ref=grid["spatial_ref"])
    # A GCP grid's axes count pixels, so they measure no CF coordinate.
    if isinstance(geobox, GeoBox):
        return write_cf_axes(placed, geobox)
    return placed


def write_reference[T: xr.DataArray | xr.Dataset](raster: T, crs: SomeCRS | None = None) -> T:
    """Write onto existing `y` and `x` what they measure: the CRS and CF axis attrs.

    Raises:
        ValueError: The raster has no `y`/`x` coordinates to measure and no
            CRS, or is placed by ground control points.
    """
    if crs is not None and ("y" not in raster.coords or "x" not in raster.coords):
        raise ValueError(
            "the raster has no y and x coordinates for a CRS to measure; place it "
            "with gs.write_grid(geobox)"
        )
    result = raster if crs is None else raster.odc.assign_crs(crs)
    geobox = result.odc.geobox
    if geobox is None:
        raise ValueError(
            "the raster carries no CRS; pass one, or place it with gs.write_grid(geobox)"
        )
    if not isinstance(geobox, GeoBox):
        raise ValueError(
            f"the raster is placed by {type(geobox).__name__}, not a regular "
            f"grid, so its axes measure no CF coordinate"
        )
    return write_cf_axes(result, geobox)


def _is_grid_mapping(coord: xr.DataArray) -> bool:
    """Return whether a coordinate is a CF grid mapping, the CRS's carrier."""
    carriers = {"crs_wkt", "spatial_ref", "grid_mapping_name"}
    return not coord.dims and not carriers.isdisjoint(coord.attrs)


# Short spellings of projected CRS units in grid node names, as EOPF writes `r10m`.
UNIT_ABBREVIATIONS = {"metre": "m", "meter": "m"}


def name_grid(geobox: GeoBox) -> str:
    """Name a grid node after EOPF: `r` + pixel size + unit, as `r10m` or `r0.0001deg`.

    Raises:
        ValueError: The grid carries no CRS, so its pixels have no ground size.
    """
    if geobox.crs is None:
        raise ValueError("a grid without a CRS has no pixel size on the ground to name it by")
    # odc spells a geographic CRS's units per axis, as degrees_north and degrees_east.
    unit = "deg" if geobox.crs.geographic else geobox.crs.units[0]
    return f"r{abs(geobox.resolution.x):g}{UNIT_ABBREVIATIONS.get(unit, unit)}"
