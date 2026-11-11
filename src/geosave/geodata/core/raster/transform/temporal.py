"""Composite a Dataset or a DataArray into native resample bins."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import xarray as xr

from geosave.geodata.conventions import STAC_PROPERTIES
from geosave.geodata.utils.datetime import Freq, freq_offset
from geosave.geodata.utils.mapping import intersect_mappings, stack_mappings

type Reducer = Literal[
    "mean", "median", "sum", "min", "max", "std", "var", "first", "last"
]

CELL_METHODS = {
    "first": "point",
    "last": "point",
    "min": "minimum",
    "max": "maximum",
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "std": "standard_deviation",
    "var": "variance",
}


type Interpolation = Literal["linear", "nearest"]


def composite_ds(
    scene: xr.Dataset,
    freq: Freq,
    method: Reducer,
    *,
    dim: str = "time",
    interpolation: Interpolation | None = None,
    properties: Literal["intersect", "stack"] = "intersect",
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    origin: str | datetime = "start_day",
    offset: str | timedelta | None = None,
) -> xr.Dataset:
    """Reduce every band of a scene into the same resample bins, each bin its cell.

    Bands without `dim` follow native resample. Accessors check each band's
    packing and class semantics before calling this operation.
    """
    from geosave.geodata.stac.raster import (
        read_properties_coordinate,
        stamp_properties_coordinate,
    )

    if dim not in scene.dims:
        raise ValueError(f"composite requires a {dim!r} dimension")
    if properties not in ("intersect", "stack"):
        raise ValueError("properties must be 'intersect' or 'stack'")
    if interpolation not in (None, "linear", "nearest"):
        raise ValueError("interpolation must be None, 'linear' or 'nearest'")
    if STAC_PROPERTIES in scene.coords and scene[STAC_PROPERTIES].dims not in (
        (),
        (dim,),
    ):
        raise ValueError(f"{STAC_PROPERTIES} must be scalar or follow {dim!r}")

    alias = freq_offset(freq)
    grouped = scene.resample(
        {dim: alias}, closed=closed, label=label, origin=origin, offset=offset
    )
    reduced = getattr(grouped, method)(keep_attrs=True)
    if interpolation is not None and reduced.sizes[dim] > 1:
        if reduced.chunks:
            reduced = reduced.chunk({dim: -1})
        reduced = reduced.interpolate_na(dim=dim, method=interpolation)

    reduced = reduced.gs.write_time_bounds(
        alias, dim=dim, closed=closed, label=label, origin=origin, offset=offset
    )

    if STAC_PROPERTIES in scene.coords:
        records = [{} for _ in range(reduced.sizes[dim])]
        index = reduced.get_index(dim)
        for date, members in grouped:
            scenes = read_properties_coordinate(members)
            if members[STAC_PROPERTIES].ndim == 0:
                scenes = scenes * members.sizes[dim]
            # An empty record is a bin that observed nothing; it states nothing.
            stated = []
            for record in scenes:
                if record:
                    stated.append(record)
            if not stated:
                continue
            if properties == "intersect":
                records[index.get_loc(date)] = intersect_mappings(stated)
            else:
                records[index.get_loc(date)] = stack_mappings(stated)
        reduced = stamp_properties_coordinate(reduced, records, dim=dim)

    for name in reduced.data_vars:
        if dim not in scene[name].dims:
            continue
        history = str(scene[name].attrs.get("cell_methods", "")).strip()
        cell = f"{history} {dim}: {CELL_METHODS[method]}".strip()
        reduced[name].attrs["cell_methods"] = cell
        if method == "var":
            reduced[name].attrs.pop("units", None)
    return reduced


def composite_da(
    band: xr.DataArray,
    freq: Freq,
    method: Reducer,
    *,
    dim: str = "time",
    interpolation: Interpolation | None = None,
    properties: Literal["intersect", "stack"] = "intersect",
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    origin: str | datetime = "start_day",
    offset: str | timedelta | None = None,
) -> xr.DataArray:
    """Reduce one band into resample bins, each bin its cell.

    The accessor checks the band's packing and class semantics before calling
    this operation.
    """
    from geosave.geodata.stac.raster import (
        read_properties_coordinate,
        stamp_properties_coordinate,
    )

    if dim not in band.dims:
        raise ValueError(f"composite requires a {dim!r} dimension")
    if properties not in ("intersect", "stack"):
        raise ValueError("properties must be 'intersect' or 'stack'")
    if interpolation not in (None, "linear", "nearest"):
        raise ValueError("interpolation must be None, 'linear' or 'nearest'")
    if STAC_PROPERTIES in band.coords and band[STAC_PROPERTIES].dims not in (
        (),
        (dim,),
    ):
        raise ValueError(f"{STAC_PROPERTIES} must be scalar or follow {dim!r}")

    alias = freq_offset(freq)
    grouped = band.resample(
        {dim: alias}, closed=closed, label=label, origin=origin, offset=offset
    )
    reduced = getattr(grouped, method)(keep_attrs=True)
    if interpolation is not None and reduced.sizes[dim] > 1:
        if reduced.chunks:
            reduced = reduced.chunk({dim: -1})
        reduced = reduced.interpolate_na(dim=dim, method=interpolation)

    reduced = reduced.gs.write_time_bounds(
        alias, dim=dim, closed=closed, label=label, origin=origin, offset=offset
    )

    if STAC_PROPERTIES in band.coords:
        records = [{} for _ in range(reduced.sizes[dim])]
        index = reduced.get_index(dim)
        for date, members in grouped:
            scenes = read_properties_coordinate(members)
            if members[STAC_PROPERTIES].ndim == 0:
                scenes = scenes * members.sizes[dim]
            # An empty record is a bin that observed nothing; it states nothing.
            stated = []
            for record in scenes:
                if record:
                    stated.append(record)
            if not stated:
                continue
            if properties == "intersect":
                records[index.get_loc(date)] = intersect_mappings(stated)
            else:
                records[index.get_loc(date)] = stack_mappings(stated)
        reduced = stamp_properties_coordinate(reduced, records, dim=dim)

    history = str(band.attrs.get("cell_methods", "")).strip()
    reduced.attrs["cell_methods"] = f"{history} {dim}: {CELL_METHODS[method]}".strip()
    if method == "var":
        reduced.attrs.pop("units", None)
    return reduced
