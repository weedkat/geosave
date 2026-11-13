"""Composite a Dataset or a DataArray into native resample bins."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import xarray as xr

from geosave.geodata.utils.datetime import Freq, freq_offset

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


def composite(
    band: xr.DataArray,
    freq: Freq,
    method: Reducer,
    *,
    dim: str = "time",
    interpolation: Interpolation | None = None,
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    origin: str | datetime = "start_day",
    offset: str | timedelta | None = None,
) -> xr.DataArray:
    """Reduce one band into resample bins, each bin its cell.

    The accessor checks the band's packing and class semantics before calling
    this operation.
    """
    if dim not in band.dims:
        raise ValueError(f"composite requires a {dim!r} dimension")
    if interpolation not in (None, "linear", "nearest"):
        raise ValueError("interpolation must be None, 'linear' or 'nearest'")

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

    history = str(band.attrs.get("cell_methods", "")).strip()
    reduced.attrs["cell_methods"] = f"{history} {dim}: {CELL_METHODS[method]}".strip()
    if method == "var":
        reduced.attrs.pop("units", None)
    return reduced
