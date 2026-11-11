"""Read CF structural relationships without interpreting them as raster pixels."""

from __future__ import annotations

from collections.abc import Hashable

import cf_xarray  # noqa: F401 registers .cf
import xarray as xr
import numpy as np


def read_support_variables(dataset: xr.Dataset) -> tuple[Hashable, ...]:
    """Name bounds and CRS carriers in Dataset order, including coordinates.

    Ancillary observations and cell measures remain data variables. Being
    referenced by another variable does not make their values structural.
    """
    names = set()
    for references in dataset.cf.bounds.values():
        names.update(references)
    for references in dataset.cf.grid_mapping_names.values():
        names.update(references)
    # GDAL/ODC can carry WKT without a CF projection name.
    for name, variable in dataset.variables.items():
        if not variable.dims and {"crs_wkt", "spatial_ref"}.intersection(
            variable.attrs
        ):
            names.add(name)
    return tuple(name for name in dataset.variables if name in names)


def check_coordinate_change(dataset: xr.Dataset, dims: tuple[str, ...]) -> None:
    """Reject coordinate changes that require recomputing CF support values."""
    for name in read_support_variables(dataset):
        if set(dims).intersection(dataset[name].dims):
            raise ValueError(
                f"Changing {dims} requires recomputing CF support variable {name!r}; "
                "remove its relationship and variable explicitly or transform it first."
            )
    for references in dataset.cf.cell_measures.values():
        for name in references:
            if set(dims).intersection(dataset[name].dims):
                raise ValueError(
                    f"Changing {dims} requires recomputing cell measure {name!r}."
                )


def read_time_bounds(dataset: xr.Dataset, name: Hashable) -> xr.DataArray | None:
    """Read declared CF time bounds with their vertex dimension last."""
    time = dataset.coords.get(name)
    if time is None:
        return None
    reference = time.attrs.get("bounds", time.encoding.get("bounds"))
    if reference is None:
        return None
    if reference not in dataset.variables:
        raise ValueError(f"time names bounds {reference!r}, which the Dataset lacks")
    bounds = dataset.cf.get_bounds(name)
    if not isinstance(bounds, xr.DataArray):
        raise ValueError("time must identify one bounds variable")
    extra = set(bounds.dims) - set(time.dims)
    if len(extra) != 1:
        raise ValueError("time bounds require one extra dimension of size two")
    dim = extra.pop()
    if bounds.sizes[dim] != 2 or set(bounds.dims) != {*time.dims, dim}:
        raise ValueError("time bounds require one pair per time label")
    ordered = bounds.transpose(*time.dims, dim)
    if np.issubdtype(ordered.dtype, np.datetime64):
        values = ordered.values
        starts, ends = values[..., 0], values[..., 1]
        if np.any(np.isnat(starts) != np.isnat(ends)):
            raise ValueError(
                "time cell bounds must contain complete pairs or paired NaT"
            )
    return ordered
