"""The calendar buckets a time axis was resampled into."""

from __future__ import annotations

from datetime import datetime as dt, timedelta
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from geosave.geodata.utils.datetime import (
    DateRange,
    Freq,
    edge_rules,
    freq_offset,
)

from ..base import AttrsGroup, Join


class TimeSpec(AttrsGroup):
    """Where the edges fall of the buckets a time axis's labels name.

    It lives on the `time` coordinate: `resample` stamps it before xarray
    groups, so the reduced axis carries it. An axis of instants carries none.
    What values were collapsed with is CF `cell_methods`, on the variables.

    Args:
        time_freq: Cadence, any pandas offset alias such as `"5D"` or `"MS"`.
        time_closed: Which bucket edge pandas treated as inclusive.
        time_label: Which bucket edge names the bucket.
        time_origin: Timestamp the edge grid was phased from, or a strategy
            name like `"start_day"`.
        time_offset: Shift added on top of `time_origin`.

    Examples:
        >>> attrs.parse(monthly.time, attrs.TimeSpec).time_freq
        'MS'
        >>> monthly.gs.timespan
        (datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 3, 31, 23, 59, 59, 999999))
    """

    # Axes cut on different cadences join as instants.
    JOIN: ClassVar[Join] = "drop"

    time_freq: Freq
    time_closed: Literal["left", "right"]
    time_label: Literal["left", "right"]
    time_origin: str | dt = "start_day"
    time_offset: str | timedelta | None = None

    @classmethod
    def from_resample(
        cls,
        freq: Freq,
        *,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | dt = "start_day",
        offset: str | timedelta | None = None,
    ) -> TimeSpec:
        """Record the bucket grid a resample call cuts, on pandas' own defaults.

        Args:
            freq: Any pandas offset alias, e.g. `"5D"`, `"MS"`.
            closed: Which bucket edge is inclusive. None resolves as pandas
                does: a period-end alias such as `"ME"` closes on the right,
                every other alias on the left.
            label: Which bucket edge names the bucket, resolved the same way.
            origin: Timestamp the edge grid is phased from, or a strategy name.
            offset: Shift added on top of `origin`.

        Returns:
            TimeSpec holding the resolved call.

        Raises:
            ValueError: pandas doesn't know `freq`.
        """
        alias = freq_offset(freq)
        closed, label = edge_rules(alias, closed, label)
        return cls(
            time_freq=alias,
            time_closed=closed,
            time_label=label,
            time_origin=origin,
            time_offset=offset,
        )

    def bounds(self, labels: np.ndarray) -> np.ndarray:
        """Return the half-open `[start, end)` edges of the bucket each label names.

        Args:
            labels: The `time` coordinate, a scalar one holding its one label.

        Returns:
            `(len(labels), 2)` `datetime64[us]` array, row `i` holding label
            `i`'s bucket start and next edge.

        Raises:
            ValueError: pandas doesn't know this spec's `time_freq`.
        """
        # A raster at one instant carries time as a scalar, which names one label.
        index = pd.DatetimeIndex(np.atleast_1d(labels))
        # time_offset already sits in `labels`; consecutive edges stay one step apart.
        step = to_offset(self.time_freq)
        low, high = (
            (index - step, index)
            if self.time_label == "right"
            else (index, index + step)
        )
        return np.stack(
            [low.to_numpy("datetime64[us]"), high.to_numpy("datetime64[us]")], axis=1
        )

    def timespan(self, labels: np.ndarray) -> DateRange:
        """Return the inclusive period `labels` cover under this spec.

        Args:
            labels: The `time` coordinate.

        Returns:
            First and last covered instant.
        """
        edges = self.bounds(labels)
        # The closing edge opens the next bucket, so back off one instant.
        last = edges[:, 1].max() - np.timedelta64(1, "us")
        return edges[:, 0].min().astype(object), last.astype(object)
