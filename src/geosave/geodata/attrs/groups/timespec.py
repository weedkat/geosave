"""The calendar buckets a time axis was resampled into."""

from __future__ import annotations

from datetime import datetime as dt, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import model_validator

from geosave.geodata.utils.datetime import (
    DateRange,
    Freq,
    edge_rules,
    freq_offset,
)

from .base import AttrsGroup


class TimeSpec(AttrsGroup):
    """Where the edges fall of the buckets a time axis's labels name.

    It lives on the `time` coordinate: `resample` stamps it before xarray
    groups, so the reduced axis carries it. An axis of instants carries none.
    What values were collapsed with is CF `cell_methods`, on the variables.
    Axes cut on different cadences join as instants, without it.

    Attributes:
        time_freq: Cadence, any pandas offset alias such as `"5D"` or `"MS"`.
        time_closed: Which bucket edge pandas treated as inclusive. Omit it to
            take pandas' default for `time_freq`.
        time_label: Which bucket edge names the bucket, resolved the same way.
        time_origin: Timestamp the edge grid was phased from, or a strategy
            name like `"start_day"`.
        time_offset: Shift added on top of `time_origin`.

    Raises:
        ValueError: pandas doesn't know `time_freq`.

    Examples:
        >>> attrs.TimeSpec(time_freq="ME").time_closed
        'right'
        >>> attrs.TimeSpec.from_attrs(monthly.time.attrs).time_freq
        'MS'
        >>> attrs.TimeSpec(time_freq="MS").bounds(monthly.time.values)[0]
        (datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 1, 31, 23, 59, 59, 999999))
    """

    time_freq: Freq
    # None only until validation resolves them from `time_freq`; never after.
    time_closed: Literal["left", "right"] | None = None
    time_label: Literal["left", "right"] | None = None
    time_origin: str | dt = "start_day"
    time_offset: str | timedelta | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_edge_rules(cls, data: Any) -> Any:
        """Fill `time_closed` and `time_label` the way pandas resolves them for `time_freq`.

        pandas closes and labels a period-end alias such as `"ME"` on the right
        and every other alias on the left. Resolving them here records the grid
        that was cut, so it can be reproduced from the attrs alone.

        Raises:
            ValueError: pandas doesn't know `time_freq`.
        """
        if not isinstance(data, dict) or "time_freq" not in data:
            return data
        alias = freq_offset(data["time_freq"])
        closed, label = edge_rules(
            alias, data.get("time_closed"), data.get("time_label")
        )
        return {
            **data,
            "time_freq": alias,
            "time_closed": closed,
            "time_label": label,
        }

    def bounds(self, labels: np.ndarray) -> list[DateRange]:
        """Return the first and last instant of the bucket each label names.

        Args:
            labels: The `time` coordinate, a scalar one holding its one label.

        Returns:
            One `(start, end)` per label, both inclusive. A bucket's closing
            edge opens the next one, so `end` stops one microsecond short of it.

        Raises:
            ValueError: pandas doesn't know this spec's `time_freq`.
        """
        # A raster at one instant carries time as a scalar, which names one label.
        index = pd.DatetimeIndex(np.atleast_1d(labels))
        # time_offset already sits in `labels`; consecutive edges stay one step apart.
        step = to_offset(self.time_freq)
        if self.time_label == "right":
            starts, next_starts = index - step, index
        else:
            starts, next_starts = index, index + step
        ends = next_starts - pd.Timedelta(1, "us")
        bounds = []
        for start, end in zip(starts, ends):
            bounds.append((start.to_pydatetime(), end.to_pydatetime()))
        return bounds
