"""Read and assign the temporal coverage coordinates of native rasters."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime as dt

import numpy as np
import xarray as xr

from geosave.geodata.conventions import END_DATETIME, START_DATETIME, TIME_COORDINATE


def assign_bounds[T: xr.DataArray | xr.Dataset](
    raster: T, bounds: Sequence[tuple[dt | np.datetime64, dt | np.datetime64]]
) -> T:
    """Record the period each `time` label covers.

    Args:
        raster: Raster carrying a `time` coordinate.
        bounds: One inclusive `(start, end)` per label. A single pair applies
            to every label, as an Item's one period covers all its time steps.

    Returns:
        Raster with `start_datetime` and `end_datetime` coordinates, along
        `time` when there is one pair per label. A scalar time label or a
        single pair shared by multiple labels gets scalar bounds. Pixel data
        and the original object are unchanged.

    Raises:
        KeyError: The raster has no `time` coordinate.
        ValueError: Bounds contain neither one shared pair nor one per label.

    Examples:
        >>> february = (dt(2024, 2, 1), dt(2024, 2, 29, 23, 59, 59, 999999))
        >>> assign_bounds(scene, [february]).end_datetime.item()
        numpy.datetime64('2024-02-29T23:59:59.999999')
    """
    starts = np.array([start for start, _ in bounds], dtype="datetime64[us]")
    ends = np.array([end for _, end in bounds], dtype="datetime64[us]")
    labels = raster[TIME_COORDINATE]
    if labels.ndim == 0 or len(bounds) != labels.size:
        if len(bounds) != 1:
            raise ValueError(
                f"{len(bounds)} bounds do not fit {labels.size} time labels; "
                f"pass one pair per label, or one pair for all of them"
            )
        return raster.assign_coords({START_DATETIME: starts[0], END_DATETIME: ends[0]})
    return raster.assign_coords(
        {
            START_DATETIME: (TIME_COORDINATE, starts),
            END_DATETIME: (TIME_COORDINATE, ends),
        }
    )


def read_bounds(coords: xr.Coordinates) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the period each `time` label covers.

    Args:
        coords: Coordinates of a raster or an Item's scene.

    Returns:
        `(starts, ends)` as one-dimensional arrays, retaining any NaT values.
        Scalar coverage produces arrays of length one; per-time coverage
        retains one entry per label. None means neither bound is stated.

    Raises:
        ValueError: Only one of `start_datetime` and `end_datetime` is present.
    """
    if START_DATETIME not in coords and END_DATETIME not in coords:
        return None
    if START_DATETIME not in coords or END_DATETIME not in coords:
        raise ValueError(
            "a composite needs both start_datetime and end_datetime "
            "coordinates; add the missing one or drop both"
        )
    return (
        np.atleast_1d(coords[START_DATETIME].values),
        np.atleast_1d(coords[END_DATETIME].values),
    )
