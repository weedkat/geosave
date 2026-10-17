"""Combining rasters on one grid into one.

Examples:
    Radar variables for the optical scene's ground::

        image = merge([optical, radar])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import xarray as xr
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import CRS_COORDINATE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geosave.geodata import Dataset


def merge(rasters: Sequence[xr.Dataset]) -> Dataset:
    """Combine uniquely named variables without aligning or broadcasting them.

    Args:
        rasters: Non-empty rasters on one exact grid, with equal coordinates
            and the same dimension order for every variable. Variable names
            must be unique.

    Returns:
        Dataset with variables in input order, attrs joined by
        `attrs.combine_attrs`.

    Raises:
        ValueError: Inputs are empty, variables collide, or grids, coordinates,
            or dimensions differ. Align grids and select observations before
            merging.

    Examples:
        >>> image = merge([optical[["red", "nir"]], radar[["VV", "VH"]]])
    """
    if not rasters or any(not raster.data_vars for raster in rasters):
        raise ValueError("merge requires non-empty rasters")
    first = rasters[0]
    grid = first.gs.geobox
    if not isinstance(grid, GeoBox):
        raise ValueError("merge requires a regular georeferenced grid")
    first_coords = first.coords.to_dataset()
    first_dims = next(iter(first.data_vars.values())).dims

    # Nothing is aligned, so every raster must already sit on the one grid.
    on_grid = []
    names: set[str] = set()
    for raster in rasters:
        if raster.gs.geobox != grid:
            raise ValueError("merge requires the same grid; align rasters first")
        # Readers write the grid mapping differently (0 or the EPSG code, bare
        # or CF x/y attrs), so each raster takes the first one's.
        if CRS_COORDINATE in raster.coords and CRS_COORDINATE in first.coords:
            raster = raster.assign_coords(
                {CRS_COORDINATE: first[CRS_COORDINATE].variable}
            )
        if raster.gs.crs is not None:
            raster = raster.gs.write_crs()
        if not raster.coords.to_dataset().equals(first_coords):
            raise ValueError(
                "merge requires equal coordinates; select observations explicitly"
            )
        if any(variable.dims != first_dims for variable in raster.data_vars.values()):
            raise ValueError(
                "merge requires equal variable dimensions; broadcasting is not "
                "supported"
            )
        duplicates = sorted(names & set(raster.gs.variables))
        if duplicates:
            raise ValueError(
                f"duplicate variables {duplicates}; rename them before merging"
            )
        names |= set(raster.gs.variables)
        on_grid.append(raster)

    merged = xr.merge(
        on_grid, join="exact", compat="equals", combine_attrs=attrs.combine_attrs
    )
    return cast("Dataset", merged)
