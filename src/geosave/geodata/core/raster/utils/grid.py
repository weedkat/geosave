"""Write grid placement on rasters: coordinates, CRS and CF axis attrs."""

from __future__ import annotations

from typing import Literal, cast

import cf_xarray  # noqa: F401 registers .cf
import odc.geo.xr  # noqa: F401 registers .odc
import xarray as xr
import numpy as np
from odc.geo import SomeResolution
from odc.geo.crs import norm_crs
from odc.geo import SomeCRS
from odc.geo.crs import CRS
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox
from odc.geo.xr import spatial_dims, xr_coords

from geosave.geodata.errors import RasterValidationError

__all__ = ["write_grid", "write_crs"]


def write_grid[T: (xr.DataArray, xr.Dataset)](
    raster: T, geobox: GeoBox | GCPGeoBox
) -> T:
    """Assign grid coordinates and CRS, preserving spatial axis names and order.

    Args:
        raster: DataArray or Dataset with identifiable, shared spatial axes
            sized as the grid. Spatial coordinates and every grid mapping coordinate are
            replaced; dimension order and other coordinates are kept.
        geobox: Affine grid, or a grid placed by ground control points.

    Returns:
        New raster whose variables point at one CRS coordinate through
        `grid_mapping`, its axes carrying CF attrs on an affine grid. Pixels
        and laziness are unchanged.

    Raises:
        RasterValidationError: Spatial axes are missing or inconsistent, the
            pixel shape differs from the grid, or the grid carries no CRS.

    Examples:
        >>> image = xr.DataArray(pixels, dims=("latitude", "longitude"))
        >>> write_grid(image, geobox).dims
        ('latitude', 'longitude')
    """
    dims = raster.gs.spatial_dims
    if dims is None:
        raise RasterValidationError(
            "Writing a grid requires identifiable, shared spatial dimensions."
        )
    if geobox.crs is None:
        raise RasterValidationError(
            "The supplied grid has no CRS. Construct the GeoBox with its known CRS "
            "before assigning it to the raster.",
            operation="Cannot place the raster on the supplied grid",
        )
    y, x = dims
    sizes = (raster.sizes[y], raster.sizes[x])
    if sizes != tuple(geobox.shape):
        raise RasterValidationError(
            f"The raster contains {sizes} pixels on {dims}, but the supplied grid "
            f"has shape {tuple(geobox.shape)}. Supply a grid matching the pixel "
            "array's shape; use reprojection to change its resolution or extent.",
            operation="Cannot place the raster on the supplied grid",
        )

    # The CRS coordinate keeps its name, so files written before keep reading.
    reference = raster.odc.crs_coord
    crs_name = str(reference.name) if reference is not None else "spatial_ref"
    grid = xr_coords(geobox, dims=dims, crs_coord_name=crs_name)
    for dim in dims:
        if dim in raster.coords and dim in grid:
            grid[dim].attrs = {**raster[dim].attrs, **grid[dim].attrs}
    placed = raster.drop_vars(list(raster.gs.grid_mapping_coords)).assign_coords(grid)
    # odc points each variable at the CRS; the grid's own CRS coordinate goes
    # back on so attrs odc does not write, such as GCPs, survive.
    placed = placed.odc.assign_crs(geobox.crs, crs_coord_name=crs_name).assign_coords(
        {crs_name: grid[crs_name]}
    )
    # A GCP grid's axes count pixels, so they measure no CF coordinate.
    if isinstance(geobox, GeoBox):
        return _write_cf_axes(placed, geobox, dims)
    return cast("T", placed)


def write_crs[T: (xr.DataArray, xr.Dataset)](
    raster: T, crs: SomeCRS | None = None
) -> T:
    """Assign a CRS and CF axis attrs to existing spatial coordinates.

    Args:
        raster: DataArray or Dataset whose coordinates already place its pixels.
        crs: CRS those coordinates are in. None keeps the one it carries.

    Returns:
        New raster whose spatial coordinates carry the CRS and CF attrs.
        No pixel moves.

    Raises:
        RasterValidationError: Placement is missing or inconsistent, has no
            CRS, or uses ground control points instead of an affine grid.
    """
    located = raster if crs is None else cast("T", raster.odc.assign_crs(crs))
    located.gs.validate(georeferenced=True, affine=True)
    geobox = cast("GeoBox", located.gs.geobox)
    dims = cast("tuple[str, str]", located.gs.spatial_dims)
    return _write_cf_axes(located, geobox, dims)


def _write_cf_axes[T: (xr.DataArray, xr.Dataset)](
    raster: T, geobox: GeoBox, dims: tuple[str, str]
) -> T:
    """Write CF metadata on validated spatial axes without changing coordinates."""
    crs = cast("CRS", geobox.crs)
    if crs.geographic:
        names = ("latitude", "longitude")
    else:
        names = ("projection_y_coordinate", "projection_x_coordinate")
    native = xr_coords(geobox, dims=dims)
    axes = {}
    for dim, name, axis in zip(dims, names, ("Y", "X"), strict=True):
        coord = raster[dim].copy()
        coord.attrs = {
            **coord.attrs,
            **native[dim].attrs,
            "standard_name": name,
            "axis": axis,
        }
        axes[dim] = coord
    return raster.assign_coords(axes)


def read_window(
    source: GeoBox, target: GeoBox
) -> tuple[tuple[slice, slice], tuple[tuple[int, int], tuple[int, int]]]:
    """Return overlapping pixel slices and padding for an aligned target."""
    linear = (
        source.transform.a,
        source.transform.b,
        source.transform.d,
        source.transform.e,
    )
    incoming = (
        target.transform.a,
        target.transform.b,
        target.transform.d,
        target.transform.e,
    )
    col, row = source.wld2pix(target.transform.c, target.transform.f)
    left, top = round(col), round(row)
    if (
        source.crs != target.crs
        or not np.allclose(linear, incoming)
        or not np.allclose((col, row), (left, top), atol=1e-6)
    ):
        raise ValueError(
            "geobox is not pixel-aligned with the raster: it needs the same CRS "
            "and pixel size and a whole-pixel offset; reproject otherwise"
        )

    height, width = target.shape
    rows, cols = source.shape
    bottom, right = min(top + height, rows), min(left + width, cols)
    if bottom <= max(top, 0) or right <= max(left, 0):
        raise ValueError("geobox does not overlap the raster")
    slices = (slice(max(top, 0), bottom), slice(max(left, 0), right))
    widths = (
        (max(-top, 0), max(top + height - rows, 0)),
        (max(-left, 0), max(left + width - cols, 0)),
    )
    return slices, widths


def resolve_grid(
    source: GeoBox, how: GeoBox | SomeCRS, resolution: SomeResolution | None
) -> GeoBox:
    """Resolve an explicit destination without choosing a second grid policy."""
    # A grid is adopted whole; a bare CRS carries the source's grid into it.
    if isinstance(how, GeoBox):
        if how.crs is None:
            raise ValueError(
                "the target grid carries no CRS; supply a georeferenced target"
            )
        if resolution is not None:
            raise ValueError(
                f"the target grid already fixes its pixel size at {how.resolution}, "
                f"so resolution {resolution!r} contradicts it; drop resolution or "
                "name a CRS instead of a grid"
            )
        geobox = how
    else:
        crs = norm_crs(how)
        if crs is None:
            raise ValueError(
                f"{how!r} names no grid and no CRS to land in; pass a GeoBox, "
                f"a raster carrying one, or a CRS such as 'EPSG:3857'"
            )
        # A CRS fixes no pixel size, so keeping the source's is odc's "auto".
        pixel_size = "auto" if resolution is None else resolution
        geobox = source.to_crs(crs, resolution=cast("Literal['auto']", pixel_size))
    return geobox


def read_spatial_dims(raster: xr.DataArray) -> tuple[str, str] | None:
    """Find declared one-dimensional CF axes, falling back to conventional names."""
    axes = raster.cf.axes
    coordinates = raster.cf.coordinates
    if not any(axes.get(axis) for axis in ("Y", "X")) and not any(
        coordinates.get(name) for name in ("latitude", "longitude")
    ):
        return spatial_dims(raster)

    conventional = spatial_dims(raster)
    spatial = []
    for position, (axis, coordinate) in enumerate(
        (("Y", "latitude"), ("X", "longitude"))
    ):
        candidates = set()
        declared = (*axes.get(axis, []), *coordinates.get(coordinate, []))
        if not declared and conventional is not None:
            candidates.add(conventional[position])
        for name in declared:
            coord = raster.coords[name]
            if coord.ndim == 1 and coord.dims == (name,):
                candidates.add(str(name))
        if len(candidates) != 1:
            return None
        spatial.append(candidates.pop())
    if spatial[0] == spatial[1]:
        return None
    return spatial[0], spatial[1]


def clear_mapping_attr(variable: xr.DataArray) -> xr.DataArray:
    """Remove a stale attrs link after ODC writes the mapping into encoding."""
    if "grid_mapping" not in variable.encoding or "grid_mapping" not in variable.attrs:
        return variable
    result = variable.copy(deep=False)
    result.attrs.pop("grid_mapping")
    return result
