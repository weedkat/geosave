"""Readers of non-pixel model inputs, such as the grid, location and dates.

A reader takes the raster and returns one numpy array. `.gs.to_numpy(context=...)`
and `.gs.to_tensor(context=...)` return each reader's array beside the pixels, keyed
by the model parameter it feeds. A reader is given as a function, or by its name in
`CONTEXT_READERS` so that a configuration can select it:

    pixels, context = chip.gs.to_tensor(
        context={"centroid": "centroid", "elevation": read_elevation}
    )
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from geosave.geodata.utils.functions import resolve_fn
from geosave.geodata.utils.rasters import Raster

if TYPE_CHECKING:
    import torch
    from odc.geo.geobox import GeoBox

# Tensors read beside the pixels, keyed by the model input they feed.
type ContextTensors = dict[str, torch.Tensor]

# Reads one context array from a raster.
type ContextReader[T: Raster] = Callable[[T], np.ndarray]


def _read_timeline(raster: Raster) -> pd.DatetimeIndex:
    """Return the date of each frame: the `time` axis, else one scalar date.

    A single observation dates itself with a scalar `time` coordinate; it is
    one frame of context, though it forms no axis to sample.

    Raises:
        ValueError: Dated tree leaves hold different labels, or the raster has
            neither a `time` dimension nor one scalar date.
    """
    times = raster.gs.times
    if times is not None:
        return times

    time = raster.gs.time_coordinate
    if time is None or time.ndim != 0:
        raise ValueError(
            "time context needs a datetime `time` dimension or one scalar date"
        )
    return pd.DatetimeIndex([time.values])


def _read_geobox(raster: Raster) -> GeoBox:
    """Return the raster's georeferenced grid.

    Raises:
        ValueError: The raster has no grid or its grid has no CRS.
    """
    geobox = raster.gs.geobox
    if geobox is None or geobox.crs is None:
        raise ValueError("grid context needs a georeferenced grid")
    return geobox


def read_transform(raster: Raster) -> np.ndarray:
    """Return the six float64 affine coefficients, `(a, b, c, d, e, f)`.

    float64 keeps sub-metre placement at UTM northings, so the transform is
    never cast with the pixels.
    """
    return np.array(_read_geobox(raster).transform[:6], dtype="float64")


def read_crs(raster: Raster) -> np.ndarray:
    """Return the grid's int64 EPSG code, a scalar.

    Raises:
        ValueError: The CRS has no EPSG code.
    """
    crs = _read_geobox(raster).crs
    if crs is None or crs.epsg is None:
        raise ValueError(f"context `crs` is an EPSG code; reproject {crs} to one first")
    return np.array(crs.epsg, dtype="int64")


def read_centroid(raster: Raster) -> np.ndarray:
    """Return float32 latitude and longitude of the grid's centre, in degrees."""
    centre = _read_geobox(raster).extent.centroid.to_crs("EPSG:4326")
    longitude, latitude = centre.coords[0]
    return np.array([latitude, longitude], dtype="float32")


def read_time(raster: Raster) -> np.ndarray:
    """Return int64 nanoseconds since 1970 of each frame, shaped `(time,)`."""
    # `asi8` counts in the index's own unit; the contract is nanoseconds.
    return _read_timeline(raster).as_unit("ns").asi8.copy()


def read_year(raster: Raster) -> np.ndarray:
    """Return int64 year of each frame, shaped `(time,)`."""
    return _read_timeline(raster).year.to_numpy(dtype="int64")


def read_day_of_year(raster: Raster) -> np.ndarray:
    """Return int64 day of year of each frame, from 1, shaped `(time,)`."""
    return _read_timeline(raster).dayofyear.to_numpy(dtype="int64")


def read_week(raster: Raster) -> np.ndarray:
    """Return int64 ISO week of each frame, shaped `(time,)`."""
    return _read_timeline(raster).isocalendar().week.to_numpy(dtype="int64")


def read_hour(raster: Raster) -> np.ndarray:
    """Return int64 hour of each frame, shaped `(time,)`."""
    return _read_timeline(raster).hour.to_numpy(dtype="int64")


CONTEXT_READERS: dict[str, ContextReader[Raster]] = {
    "transform": read_transform,
    "crs": read_crs,
    "centroid": read_centroid,
    "time": read_time,
    "year": read_year,
    "day_of_year": read_day_of_year,
    "week": read_week,
    "hour": read_hour,
}


def read_context[T: Raster](
    raster: T, readers: Mapping[str, ContextReader[T] | str]
) -> dict[str, np.ndarray]:
    """Read each requested context array from `raster`.

    Args:
        raster: Raster the readers read.
        readers: Model input names mapped to readers, each a function of the
            raster or its name in `CONTEXT_READERS`.

    Returns:
        Context arrays keyed as `readers`.

    Raises:
        KeyError: A reader name is not in `CONTEXT_READERS`.
        ValueError: A reader refuses the raster.
    """
    values = {}
    for name, reader in readers.items():
        read = resolve_fn(reader, CONTEXT_READERS)
        values[name] = read(raster)
    return values
