"""Datetime parsing, calendar calculations, and the spans vector rows state."""

from __future__ import annotations

import re
from datetime import datetime as dt
from datetime import timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

DateRange = tuple[dt, dt]
RegionDatetime = str | tuple[str, str] | DateRange

# Cadence a time axis is bucketed at — any pandas offset alias; the Literal is just IDE suggestions.
type Freq = Literal["D", "W", "MS", "ME", "QS", "QE", "YS", "YE"] | str

# ISO ("2019-05-07T10:30:15") or compact ("20190507T103015") — "-" and ":" separators optional.
_DATETIME_PATTERN = re.compile(
    r"^(?P<year>\d{4})"
    r"(?:-?(?P<month>\d{2})"
    r"(?:-?(?P<day>\d{2})"
    r"(?:[T ](?P<hour>\d{2})"
    r"(?::?(?P<minute>\d{2})"
    r"(?::?(?P<second>\d{2})(?P<fraction>\.\d{1,9})?)?"
    r")?(?P<timezone>Z|[+-]\d{2}:?\d{2})?"
    r")?)?)?$"
)


def naive_utc(value: dt) -> dt:
    """Drop the timezone off a datetime, shifting to UTC when it had one.

    Args:
        value: Aware or naive datetime.

    Returns:
        The same instant with no tzinfo, comparable against a `time` coord's
        own labels, which numpy holds without a timezone.

    Examples:
        >>> naive_utc(dt(2025, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=2))))
        datetime.datetime(2025, 6, 1, 10, 0)
    """
    return (
        value.astimezone(timezone.utc).replace(tzinfo=None)
        if value.tzinfo is not None
        else value
    )


def _parse_timezone(raw: str | None) -> timezone | None:
    """Read a UTC offset off the pattern's timezone group, None for none."""
    if raw is None:
        return None
    if raw == "Z":
        return timezone.utc
    sign = 1 if raw[0] == "+" else -1
    digits = raw[1:].replace(":", "")
    return timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:4])))


def _parse_daterange(value: str) -> DateRange:
    """Read the inclusive period one ISO or compact timestamp covers.

    Raises:
        ValueError: `value` doesn't match the pattern.
    """
    match = _DATETIME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported datetime value: {value!r}")

    year = int(match.group("year"))
    month = int(match.group("month") or 1)
    day = int(match.group("day") or 1)
    hour = int(match.group("hour") or 0)
    minute = int(match.group("minute") or 0)
    second = int(match.group("second") or 0)
    fraction = match.group("fraction")
    microsecond = int(fraction[1:7].ljust(6, "0")) if fraction else 0
    start = dt(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        tzinfo=_parse_timezone(match.group("timezone")),
    )

    if match.group("month") is None:
        end = start.replace(year=year + 1)
    elif match.group("day") is None:
        next_month = (
            start.replace(year=year + 1, month=1)
            if month == 12
            else start.replace(month=month + 1)
        )
        end = next_month
    elif match.group("hour") is None:
        end = start + timedelta(days=1)
    elif match.group("minute") is None:
        end = start + timedelta(hours=1)
    elif match.group("second") is None:
        end = start + timedelta(minutes=1)
    elif fraction is None:
        end = start + timedelta(seconds=1)
    else:
        fraction_digits = len(fraction) - 1  # includes the leading "."
        end = start + timedelta(microseconds=10 ** (6 - fraction_digits))

    return start, end - timedelta(microseconds=1)


def parse_daterange(value: RegionDatetime) -> DateRange:
    """Read one inclusive `(start, end)` range off a datetime input.

    A partial timestamp covers the whole period it names, so `"2019-05"` is
    the whole of May. A pair of datetimes is already a range and passes
    through untouched.

    Args:
        value: ISO or compact timestamp, a `"<start>/<end>"` interval, a pair
            of either, or a pair of datetimes.

    Returns:
        First and last covered instant, the last at microsecond precision.

    Raises:
        ValueError: A string operand doesn't match the pattern.

    Examples:
        >>> parse_daterange("2019-05-07")
        (datetime.datetime(2019, 5, 7, 0, 0),
         datetime.datetime(2019, 5, 7, 23, 59, 59, 999999))
        >>> parse_daterange("2019-05/2019-06")
        (datetime.datetime(2019, 5, 1, 0, 0),
         datetime.datetime(2019, 6, 30, 23, 59, 59, 999999))
    """
    match value:
        case (dt(), dt()):
            return value
        case (str(), str()):
            left, right = value
            return _parse_daterange(left)[0], _parse_daterange(right)[1]
        case str() if "/" in value:
            left, right = value.split("/", 1)
            return _parse_daterange(left)[0], _parse_daterange(right)[1]
        case str():
            return _parse_daterange(value)


def freq_offset(freq: Freq) -> str:
    """Read the canonical pandas offset alias one cadence buckets on.

    Args:
        freq: Any pandas offset alias — `"D"`, `"5D"`, `"W"`, `"ME"`
            (month end), `"MS"` (month start), `"QE"`, `"YE"`.

    Returns:
        The alias pandas resamples and steps the bucket grid with. One
        cadence has one spelling here, whatever spelling came in.

    Raises:
        ValueError: pandas doesn't know `freq`.

    Examples:
        >>> freq_offset("5d"), freq_offset("W")
        ('5D', 'W-SUN')
    """
    try:
        return to_offset(freq).freqstr  # type: ignore[union-attr]
    except ValueError as e:
        raise ValueError(
            f"Unknown freq {freq!r} — needs a pandas offset alias, e.g. 'D', '5D', 'ME'"
        ) from e


def parse_stem_dates(stem: str) -> DateRange | None:
    """Read the period a filename stem names, off the date tokens it carries.

    Reads any stem spelling its dates compactly, as `20190507`. Every token parsing as a datetime counts, a four-digit
    identifier among them, as the year it spells.

    Args:
        stem: Filename stem, its parts separated by `"_"` or `"-"`.

    Returns:
        First and last instant the stem's date tokens cover, or None where it
        carries no date token.

    Examples:
        >>> parse_stem_dates("tile-20190507")
        (datetime.datetime(2019, 5, 7, 0, 0),
         datetime.datetime(2019, 5, 7, 23, 59, 59, 999999))
        >>> parse_stem_dates("20190507-20190509")
        (datetime.datetime(2019, 5, 7, 0, 0),
         datetime.datetime(2019, 5, 9, 23, 59, 59, 999999))
        >>> parse_stem_dates("dem") is None
        True
    """
    covered: list[DateRange] = []
    for token in re.split(r"[_-]", stem):
        try:
            covered.append(_parse_daterange(token))
        except ValueError:
            continue

    if not covered:
        return None
    return min(start for start, _ in covered), max(end for _, end in covered)


# Columns a vector row dates itself by, as STAC names them.
ROW_TIME_COLUMNS = ("datetime", "start_datetime", "end_datetime")


def read_row_spans(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Read each row's first and last instant as naive UTC.

    A row states `datetime`, or `start_datetime` and `end_datetime`. A missing
    bound falls back to `datetime`; without either it stays NaT, which reads
    as open.

    Args:
        rows: Vector rows with any of the STAC time columns.

    Returns:
        Start instants and end instants, indexed as `rows`.

    Examples:
        >>> starts, ends = read_row_spans(items)
        >>> starts.dt.strftime("%Y-%m-%d").tolist()
        ['2024-03-08', '2024-03-28']
    """
    no_instants = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    instants_by_column = {}
    for column in ROW_TIME_COLUMNS:
        if column in rows:
            stamps = pd.to_datetime(rows[column], utc=True, format="mixed")
            instants_by_column[column] = stamps.dt.tz_convert(None)
    nominal_instants = instants_by_column.get("datetime", no_instants)
    start_instants = instants_by_column.get("start_datetime", nominal_instants)
    end_instants = instants_by_column.get("end_datetime", nominal_instants)
    return (
        start_instants.fillna(nominal_instants),
        end_instants.fillna(nominal_instants),
    )


def find_overlapping(
    start_instants: pd.Series, end_instants: pd.Series, timespan: DateRange
) -> np.ndarray:
    """Mark the spans sharing at least one instant with `timespan`.

    Both spans are closed; a NaT bound is open, so an undated span overlaps
    every timespan.

    Args:
        start_instants: First instant of each span, naive UTC.
        end_instants: Last instant of each span, naive UTC.
        timespan: Closed interval, naive UTC.

    Returns:
        Boolean mask in span order.

    Examples:
        >>> find_overlapping(starts, ends, (datetime(2024, 3, 20), datetime(2024, 4, 1)))
        array([False,  True, False])
    """
    first_instant = pd.Timestamp(timespan[0])
    last_instant = pd.Timestamp(timespan[1])
    starts_before_end = start_instants.isna() | (start_instants <= last_instant)
    ends_after_start = end_instants.isna() | (end_instants >= first_instant)
    return (starts_before_end & ends_after_start).to_numpy()
