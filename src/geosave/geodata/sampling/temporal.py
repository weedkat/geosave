"""Sample each prepared raster on its own timeline."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd
import xarray as xr

from geosave.geodata.core.conventions import TIME_COORDINATE
from geosave.geodata.utils.datetime import Freq
from geosave.geodata.utils.rasters import Raster, as_rasters


def choose_bucket_positions(
    times: pd.DatetimeIndex, freq: Freq, keep: Literal["first", "last"]
) -> pd.Series:
    """Return the position of the observation each `freq` bucket keeps.

    Args:
        times: Time labels in axis order.
        freq: Bucket length as a pandas offset alias, such as "MS".
        keep: Which observation a bucket keeps when several fall in it.

    Returns:
        Positions along `times` indexed by bucket label, from the first bucket
        to the last; NaN for a bucket without an observation.
    """
    buckets = pd.Series(np.arange(len(times)), index=times).resample(freq)
    return buckets.last() if keep == "last" else buckets.first()


def find_filled_buckets(
    rasters: Sequence[xr.DataArray | xr.Dataset], freq: Freq
) -> pd.Series:
    """Mark the `freq` buckets where every dated raster holds an observation.

    Args:
        rasters: Rasters on their own timelines, such as the leaves of a tree.
            Those without a `time` dimension are not counted.
        freq: Bucket length as a pandas offset alias, such as "MS".

    Returns:
        True or False indexed by bucket label, from the earliest bucket of any
        raster to the latest.

    Raises:
        ValueError: No raster has a datetime `time` dimension.
        TypeError: A `time` dimension does not hold datetimes.
    """
    positions_per_raster = []
    for raster in rasters:
        times = raster.gs.times
        if times is not None:
            positions_per_raster.append(choose_bucket_positions(times, freq, "last"))
    if not positions_per_raster:
        raise ValueError("sampling needs a datetime `time` dimension")

    # Aligning the rasters leaves NaN where a bucket is outside one's dates.
    positions = pd.concat(positions_per_raster, axis=1)
    return cast("pd.Series", positions.notna().all(axis=1))


def select_window[T: (xr.DataArray, xr.Dataset)](
    raster: T, freq: Freq, buckets: pd.DatetimeIndex, keep: Literal["first", "last"]
) -> T:
    """Select the observation `raster` keeps in each bucket of one window.

    Only time labels are read; the selection is lazy.

    Args:
        raster: One raster, or one Dataset of a tree.
        freq: Bucket length as a pandas offset alias, such as "MS".
        buckets: Labels of the window's buckets; each holds an observation.
        keep: Which observation a bucket keeps when several fall in it.

    Returns:
        `raster` on one observation per bucket, each with its own date; a
        raster without a `time` dimension, such as a label, unchanged.
    """
    if TIME_COORDINATE not in raster.dims:
        return raster

    times = raster.get_index(TIME_COORDINATE)
    positions = choose_bucket_positions(cast("pd.DatetimeIndex", times), freq, keep)
    return raster.isel({TIME_COORDINATE: positions.loc[buckets].to_numpy(dtype=int)})


def iter_samples[T: Raster](
    rasters: T | Iterable[T],
    freq: Freq,
    *,
    size: int = 1,
    stride: int = 1,
    keep: Literal["first", "last"] = "last",
    missing: Literal["skip", "raise"] = "skip",
) -> Iterator[T]:
    """Yield temporal windows of each raster, in input order, without reading pixels.

    A window is `size` consecutive time buckets of `freq`, each holding one
    observation. A bucket only selects; to blend a bucket's observations or
    fill a bucket without any, composite first with
    `gs.composite(freq, method, interpolation=...)`. In a DataTree, nodes
    observed on their own dates, such as Sentinel-1 and Sentinel-2, meet in
    the same buckets and keep their own observations.

    Args:
        rasters: One DataArray, Dataset or DataTree, or an iterable of them.
            Static tree nodes accompany every window.
        freq: Bucket length as a pandas offset alias, such as "MS".
        size: Number of consecutive buckets in a window.
        stride: Number of buckets between window starts.
        keep: Which observation to select when several fall in a bucket.
        missing: What a window with an empty bucket does: `"skip"` leaves it
            out; `"raise"` refuses the raster before yielding any of its
            windows.

    Yields:
        One native raster per window, each dated node with `size` observations.

    Raises:
        ValueError: A raster lacks a datetime dimension, or a window has an
            empty bucket under `missing="raise"`.
        TypeError: A `time` dimension does not hold datetimes.

    Examples:
        >>> windows = gs.iter_samples(tree, "MS", size=2)
        >>> location, chip = next(iter(gs.RasterChipper(windows, size=256)))
    """
    for raster in as_rasters(rasters):
        # A tree buckets each of its Datasets; a DataArray or Dataset buckets itself.
        if isinstance(raster, xr.DataTree):
            layers = []
            for node in raster.leaves:
                if node.data_vars:
                    layers.append(node.gs.to_dataset())
        else:
            layers = [raster]
        filled_buckets = find_filled_buckets(layers, freq)

        # Keep the windows whose every bucket is filled; labels only, no pixels.
        complete_windows = []
        for first in range(0, len(filled_buckets) - size + 1, stride):
            window_filled = filled_buckets.iloc[first : first + size]
            empty_buckets = window_filled.index[~window_filled]
            if empty_buckets.empty:
                complete_windows.append(window_filled.index)
            elif missing == "raise":
                labels = [bucket.strftime("%Y-%m-%d") for bucket in empty_buckets]
                raise ValueError(
                    f"the window from {window_filled.index[0]:%Y-%m-%d} has empty "
                    f"{freq} buckets {labels}; composite with interpolation to fill "
                    "them, or pass missing='skip'"
                )

        for buckets in complete_windows:
            if isinstance(raster, xr.DataTree):
                window = raster.map_over_datasets(select_window, freq, buckets, keep)
            else:
                window = select_window(raster, freq, buckets, keep)
            yield cast("T", window)
