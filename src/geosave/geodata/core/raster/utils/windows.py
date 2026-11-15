"""Group time labels into windows of consecutive slots, without reading pixels."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from geosave.geodata.utils.datetime import Freq, freq_offset

type Keep = Literal["first", "last"]


def window_times(
    times: pd.DatetimeIndex,
    freq: Freq,
    *,
    size: int = 1,
    stride: int = 1,
    keep: Keep = "last",
) -> list[list[int]]:
    """Pick the time-axis positions of each window of consecutive slots.

    The labels are binned into slots of `freq`. A slot holding several
    observations keeps one; nothing is read or blended. A window is `size`
    slots in a row, and one holding an empty slot is left out.

    Args:
        times: Time labels of the raster, in order.
        freq: Length of one slot, as a pandas offset alias such as "MS".
        size: Slots per window.
        stride: Slots between the starts of consecutive windows.
        keep: The observation a slot keeps when several fall in it.

    Returns:
        One list of positions along `times` per window, in time order.

    Raises:
        ValueError: `size` or `stride` is below 1, or `keep` is unknown.

    Examples:
        >>> times = pd.DatetimeIndex(["2025-06-03", "2025-06-28", "2025-07-23", "2025-09-11"])
        >>> window_times(times, "MS", size=2, keep="last")
        [[1, 2]]
    """
    if size < 1 or stride < 1:
        raise ValueError("window size and stride must be at least 1")
    if keep not in ("first", "last"):
        raise ValueError(f"keep must be 'first' or 'last', got {keep!r}")

    positions = pd.Series(np.arange(len(times)), index=times)
    slots = positions.resample(freq_offset(freq))
    kept_positions = slots.first() if keep == "first" else slots.last()

    windows = []
    for first_slot in range(0, len(kept_positions) - size + 1, stride):
        window_positions = kept_positions.iloc[first_slot : first_slot + size]
        # A window holding an empty slot is not a sample.
        if window_positions.isna().any():
            continue
        windows.append(window_positions.astype(int).tolist())
    return windows
