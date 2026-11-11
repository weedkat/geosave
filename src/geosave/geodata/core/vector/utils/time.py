"""Read and validate temporal coverage on native spatial records."""

import pandas as pd

from geosave.geodata.errors import VectorValidationError
from geosave.geodata.stac.item import read_timespan


def read_times(rows: pd.DataFrame) -> pd.DataFrame:
    """Return nominal time and inclusive start/end coverage, indexed like rows.

    Values are naive UTC. Undated rows and the STAC timeless sentinel produce
    NaT. This reads temporal metadata only; collections and geometry are not
    required. Use gs.validate(temporal=True) to require complete ordered bounds.
    """
    records = []
    for position, (_, row) in enumerate(rows.iterrows()):
        properties = {}
        for name in ("datetime", "start_datetime", "end_datetime"):
            value = row.get(name)
            if bool(pd.notna(value)):
                properties[name] = value

        try:
            span = read_timespan(
                properties.get("datetime"),
                properties.get("start_datetime"),
                properties.get("end_datetime"),
            )
            nominal = start = end = pd.NaT
            if span is not None:
                start, end = span
                if "datetime" in properties:
                    nominal = pd.to_datetime(
                        properties["datetime"], utc=True
                    ).tz_convert(None)
        except (ValueError, TypeError, OverflowError) as error:
            identity = row.get("id", position)
            raise VectorValidationError(
                f"Row at position {position} (Item {identity!r}): {error}"
            ) from error

        records.append((nominal, start, end))

    times = pd.DataFrame(
        records,
        index=rows.index,
        columns=pd.Index(["time", "start", "end"]),
    )
    # Let pandas retain the supplied precision; forcing microseconds would
    # merge distinct nanosecond observations into one sampling slot.
    for name in times:
        times[name] = pd.to_datetime(times[name])
    return times


def validate_time(rows: pd.DataFrame) -> None:
    """Require paired bounds, ordered coverage and a nominal time within it."""
    times = read_times(rows)
    for position, (_, row) in enumerate(rows.iterrows()):
        identity = row.get("id", position)
        prefix = f"Row at position {position} (Item {identity!r})"
        if pd.notna(row.get("start_datetime")) != pd.notna(row.get("end_datetime")):
            raise VectorValidationError(
                f"{prefix}: provide both start_datetime and end_datetime"
            )

        nominal, start, end = times.iloc[position]
        if pd.notna(start) and end < start:
            raise VectorValidationError(
                f"{prefix}: end_datetime precedes start_datetime"
            )
        if pd.notna(nominal) and not start <= nominal <= end:
            raise VectorValidationError(
                f"{prefix}: datetime lies outside the stated coverage"
            )
