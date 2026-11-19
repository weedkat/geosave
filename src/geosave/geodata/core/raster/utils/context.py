"""Grid and non-pixel model inputs read beside a raster's tensors.

`.gs.to_tensor(context=[...])` looks each name up in `TENSOR_CONTEXT` and
stores the reader's tensor under that name. Add a reader to add a name:

    TENSOR_CONTEXT["elevation"] = read_mean_elevation
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr


if TYPE_CHECKING:
    import torch

# Tensors read beside the pixels, keyed by their `TENSOR_CONTEXT` name.
type ContextTensors = dict[str, torch.Tensor]


def read_timeline(tree: xr.DataTree) -> pd.DatetimeIndex:
    """Return the dated nodes' one timeline, in axis order.

    Raises:
        ValueError: No node is dated, or dated nodes hold different labels.
    """
    times = None
    for node in tree.leaves:
        if not node.data_vars:
            continue
        node_times = node.gs.to_dataset().gs.times
        if node_times is None:
            continue
        if times is None:
            times = node_times
        elif not node_times.equals(times):
            raise ValueError(
                f"node {node.path!r} has its own time labels; this operation needs "
                "every dated node on one timeline"
            )
    if times is None:
        raise ValueError("time context needs a node with a datetime `time` dimension")
    return times


def read_time(tree: xr.DataTree) -> torch.Tensor:
    """Return int64 nanoseconds since 1970 of each frame, shaped `(time,)`."""
    import torch

    # `asi8` counts in the index's own unit; the contract is nanoseconds.
    nanoseconds = read_timeline(tree).as_unit("ns").asi8.copy()
    return torch.as_tensor(nanoseconds)


def read_year(tree: xr.DataTree) -> torch.Tensor:
    """Return int64 year of each frame, shaped `(time,)`."""
    import torch

    years = [when.year for when in read_timeline(tree)]
    return torch.tensor(years, dtype=torch.int64)


def read_day_of_year(tree: xr.DataTree) -> torch.Tensor:
    """Return int64 day of year of each frame, from 1, shaped `(time,)`."""
    import torch

    days = [when.dayofyear for when in read_timeline(tree)]
    return torch.tensor(days, dtype=torch.int64)


def read_week(tree: xr.DataTree) -> torch.Tensor:
    """Return int64 ISO week of each frame, shaped `(time,)`."""
    import torch

    weeks = read_timeline(tree).isocalendar().week
    return torch.as_tensor(weeks.to_numpy(dtype="int64"))


def read_hour(tree: xr.DataTree) -> torch.Tensor:
    """Return int64 hour of each frame, shaped `(time,)`."""
    import torch

    hours = [when.hour for when in read_timeline(tree)]
    return torch.tensor(hours, dtype=torch.int64)


def read_centroid(tree: xr.DataTree) -> torch.Tensor:
    """Return float32 latitude and longitude of the grid's centre, in degrees.

    Raises:
        ValueError: The tree has no georeferenced grid.
    """
    import torch

    geobox = tree.gs.common_geobox()
    if geobox is None or geobox.crs is None:
        raise ValueError("centroid needs a georeferenced grid")
    centre = geobox.extent.centroid.to_crs("EPSG:4326")
    longitude, latitude = centre.coords[0]
    return torch.tensor([latitude, longitude], dtype=torch.float32)


TENSOR_CONTEXT: dict[str, Callable[[xr.DataTree], torch.Tensor]] = {
    "time": read_time,
    "year": read_year,
    "day_of_year": read_day_of_year,
    "week": read_week,
    "hour": read_hour,
    "centroid": read_centroid,
}


def read_context(
    tree: xr.DataTree,
    *,
    grid: bool,
    context: Sequence[str],
) -> ContextTensors:
    """Read requested grid and context values as tensors.

    Args:
        tree: The rasters the layers were read from.
        grid: Add `transform`, the six float64 affine coefficients, and `crs`,
            the int64 EPSG code.
        context: Names in `TENSOR_CONTEXT`.

    Raises:
        KeyError: A context name is not in `TENSOR_CONTEXT`.
        ValueError: The grid CRS has no EPSG code, or a reader refuses the tree.
    """
    import torch

    entries = {}
    if grid:
        geobox = tree.gs.common_geobox()
        if geobox is None or geobox.crs is None:
            raise ValueError("grid entries need a georeferenced grid")
        if geobox.crs.epsg is None:
            raise ValueError(
                f"tensor `crs` is an EPSG code; reproject {geobox.crs} to one first"
            )
        entries["transform"] = torch.tensor(geobox.transform[:6], dtype=torch.float64)
        entries["crs"] = torch.tensor(geobox.crs.epsg, dtype=torch.int64)
    for name in context:
        if name not in TENSOR_CONTEXT:
            raise KeyError(f"Unknown context {name!r}; known: {sorted(TENSOR_CONTEXT)}")
        entries[name] = TENSOR_CONTEXT[name](tree)

    return entries
