from __future__ import annotations

from datetime import datetime as dt

import numpy as np

from geosave.geodata.attrs import TimeSpec

LAST_MICROSECOND = dict(hour=23, minute=59, second=59, microsecond=999999)


def test_bounds_run_forward_and_end_one_instant_before_next_bucket() -> None:
    spec = TimeSpec(time_freq="MS")
    labels = np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]")

    bounds = spec.bounds(labels)

    assert bounds == [
        (dt(2024, 1, 1), dt(2024, 1, 31, **LAST_MICROSECOND)),
        (dt(2024, 2, 1), dt(2024, 2, 29, **LAST_MICROSECOND)),
    ]


def test_bounds_run_backward_for_end_labels() -> None:
    spec = TimeSpec(time_freq="ME")
    labels = np.array(["2024-01-31", "2024-02-29"], dtype="datetime64[ns]")

    bounds = spec.bounds(labels)

    assert bounds == [
        (dt(2023, 12, 31), dt(2024, 1, 30, **LAST_MICROSECOND)),
        (dt(2024, 1, 31), dt(2024, 2, 28, **LAST_MICROSECOND)),
    ]


def test_bounds_place_each_label_independently_of_gaps() -> None:
    spec = TimeSpec(time_freq="MS")
    labels = np.array(["2024-01-01", "2024-04-01"], dtype="datetime64[ns]")

    bounds = spec.bounds(labels)

    assert bounds == [
        (dt(2024, 1, 1), dt(2024, 1, 31, **LAST_MICROSECOND)),
        (dt(2024, 4, 1), dt(2024, 4, 30, **LAST_MICROSECOND)),
    ]


def test_bounds_step_by_the_full_multiple() -> None:
    spec = TimeSpec(time_freq="5D")
    labels = np.array(["2024-01-01", "2024-01-06"], dtype="datetime64[ns]")

    bounds = spec.bounds(labels)

    assert bounds == [
        (dt(2024, 1, 1), dt(2024, 1, 5, **LAST_MICROSECOND)),
        (dt(2024, 1, 6), dt(2024, 1, 10, **LAST_MICROSECOND)),
    ]
