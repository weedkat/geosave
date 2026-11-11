"""Plan spatial membership and temporal windows without opening raster assets."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from datetime import datetime
from typing import Literal, get_args, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


type Interval = Literal["raise", "first", "last", "overlap", "drop"]


def validate_options(
    *,
    collections: Sequence[str] | None,
    match: str,
    overlap: float,
    window: int,
    step: int,
    interval: str,
    missing: str,
) -> None:
    """Check the sampling request before inspecting membership or opening assets."""
    if collections is not None:
        if isinstance(collections, str) or not collections:
            raise ValueError("collections must be a nonempty sequence of names")
        if any(not isinstance(name, str) or not name for name in collections):
            raise ValueError("collections must contain nonempty names")
        if len(set(collections)) != len(collections):
            raise ValueError("collections must not repeat")
    if match not in ("exact", "overlap"):
        raise ValueError("match must be 'exact' or 'overlap'")
    if not 0 < overlap <= 1:
        raise ValueError("overlap must be a fraction greater than zero and at most one")
    for name, value in (("window", window), ("step", step)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer counting slots")
    if interval not in get_args(Interval.__value__):
        raise ValueError(
            "interval must be 'raise', 'first', 'last', 'overlap', or 'drop'"
        )
    if missing not in ("drop", "keep"):
        raise ValueError("missing must be 'drop' or 'keep'")


def group_footprints(
    items: gpd.GeoDataFrame, match: str, overlap: float
) -> tuple[gpd.GeoSeries, list[list[int]]]:
    """Return founding footprints and row index labels sharing each footprint.

    Exact matching uses geometric equality, independent of vertex order.
    Overlap retains the smallest-first footprint discovery convention.
    """
    items.gs.validate(area=True)
    geometries = items.geometry.to_numpy()
    founders = []
    order = range(len(items))
    if match == "overlap":
        order = np.argsort(shapely.area(geometries), kind="stable")
    for position in order:
        geometry = geometries[position]
        known = geometries[founders]
        if match == "exact":
            represented = shapely.equals(geometry, known)
        else:
            represented = shapely.area(
                shapely.intersection(geometry, known)
            ) >= overlap * shapely.area(known)
        if not np.any(represented):
            founders.append(int(position))
    founders.sort()

    # Every covering row joins each founder it covers, not just the first.
    groups = []
    for position in founders:
        geometry = geometries[position]
        if match == "exact":
            members = shapely.equals(geometry, geometries)
        else:
            members = shapely.area(
                shapely.intersection(geometry, geometries)
            ) >= overlap * shapely.area(geometry)
        groups.append(items.index[members].tolist())
    footprints = gpd.GeoSeries(geometries[founders], crs=items.crs)
    return footprints, groups


def read_range(
    date_range: tuple[str | datetime | None, str | datetime | None] | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Normalize optional half-open sampling bounds to naive UTC."""
    if date_range is None:
        return None, None
    if len(date_range) != 2:
        raise ValueError("date_range must contain two bounds")
    bounds = []
    for value in date_range:
        bound = None if value is None else pd.Timestamp(value)
        if bound is not None:
            if bool(pd.isna(bound)):
                raise ValueError("date_range bounds must not be NaT")
            if bound.tzinfo is not None:
                bound = bound.tz_convert("UTC").tz_localize(None)
        bounds.append(bound)
    start, end = bounds
    if start is not None and end is not None and start >= end:
        raise ValueError("date_range end must follow its start")
    return start, end


def select_collections(
    rows: gpd.GeoDataFrame, collections: Sequence[str] | None
) -> gpd.GeoDataFrame:
    """Select requested collections, retaining the source index and metadata."""
    if collections is None:
        return rows
    if "collection" not in rows:
        raise ValueError("Collection selection requires a collection column")
    absent = set(collections) - set(rows["collection"])
    if absent:
        raise KeyError(f"No Item belongs to {sorted(absent)}")
    return rows.loc[rows["collection"].isin(collections)]


def select_range(
    times: pd.DataFrame,
    bounds: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> pd.DataFrame:
    """Keep inclusive coverage intersecting a half-open range, including static rows."""
    start, end = bounds
    dated = times["start"].notna()
    if start is not None:
        dated &= times["end"] >= start
    if end is not None:
        dated &= times["start"] < end
    return times.loc[times["start"].isna() | dated]


def read_frequency(
    freq: str | pd.offsets.BaseOffset | None, anchor: pd.Timestamp | None
) -> pd.offsets.BaseOffset | None:
    """Normalize a forward frequency, requiring an origin for relative offsets."""
    if freq is None:
        return None
    offset = pd.tseries.frequencies.to_offset(freq)
    if type(offset) is pd.DateOffset and anchor is None:
        raise ValueError("A relative DateOffset requires a date_range lower bound")
    probe = (
        anchor
        if anchor is not None
        else cast("pd.Timestamp", pd.Timestamp("2000-01-01"))
    )
    if offset.n <= 0 or probe + offset <= probe:
        raise ValueError("freq must advance time")
    return offset


def clip_observations(
    observations: pd.DataFrame,
    bounds: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> pd.DataFrame:
    """Clip assignment coverage and retain nominal times inside the chosen range.

    Input rows have time/start/end columns. No source catalog dates are changed;
    an observation's nominal time takes precedence over its wider coverage.
    """
    start, end = bounds
    observations = select_range(observations, bounds).copy()
    # Clip assignment coverage only; catalog rows retain their original coverage.
    if start is not None:
        observations["start"] = observations["start"].clip(lower=start)
    if end is not None:
        observations["end"] = observations["end"].clip(
            upper=end - pd.Timedelta(1, "ns")
        )
    nominal = observations["time"]
    keep = nominal.isna()
    inside = nominal.notna()
    if start is not None:
        inside &= nominal >= start
    if end is not None:
        inside &= nominal < end
    observations = observations.loc[keep | inside]
    return observations


def build_timeline(
    observations: pd.DataFrame,
    freq: pd.offsets.BaseOffset | None,
    bounds: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> pd.DatetimeIndex:
    """Build exact instants or left-labelled calendar slots from dated coverage.

    Static observations create no timestamps. Calendar slots include empty
    periods; a supplied lower bound anchors relative frequencies across groups.
    """
    start, end = bounds
    instants = []
    for item in observations.to_dict("records"):
        if pd.notna(item["time"]):
            instants.append(item["time"])
        elif pd.notna(item["start"]):
            if freq is None:
                raise ValueError(
                    f"Item {item['id']!r} has interval-only time; supply freq to assign calendar slots"
                )
            instants.extend([item["start"], item["end"]])
    times = pd.DatetimeIndex(sorted(set(instants)))
    if freq is not None and len(times):
        first = start if start is not None else times.min()
        last = end - pd.Timedelta(1, "ns") if end is not None else times.max()
        if type(freq) is pd.DateOffset:
            times = pd.date_range(first, last, freq=freq)
        else:
            times = pd.DatetimeIndex(
                pd.Series(0, index=pd.DatetimeIndex([first, last]))
                .resample(
                    freq,
                    closed="left",
                    label="left",
                    origin=start if start is not None else "epoch",
                )
                .size()
                .index
            )
    return times


def assign_slots(
    observations: pd.DataFrame,
    times: pd.DatetimeIndex,
    interval: Interval,
) -> pd.DataFrame:
    """Assign observations to a supplied timeline without constructing a calendar.

    Input columns are row, id, collection, time, start and end. Output columns
    are time, collection and indices; row identities pass through unchanged.
    NaT denotes static membership. Item coverage includes both endpoints, while
    calendar slots exclude their right edge.
    """
    observations = observations.assign(
        observed=observations["time"].fillna(observations["start"])
    )
    observations = observations.sort_values("observed", kind="stable")
    assignments = []
    for item in observations.to_dict("records"):
        if pd.isna(item["start"]):
            labels = [pd.NaT]
        elif pd.notna(item["time"]):
            position = times.searchsorted(item["time"], side="right") - 1
            labels = [times[position]]
        else:
            first = max(0, int(times.searchsorted(item["start"], side="right")) - 1)
            last = int(times.searchsorted(item["end"], side="right")) - 1
            labels = times[first : last + 1].tolist()
            if len(labels) > 1:
                if interval == "raise":
                    raise ValueError(
                        f"Item {item['id']!r} spans slots {labels[0]} through {labels[-1]}; choose interval='first', 'last', 'overlap', or 'drop'"
                    )
                if interval == "drop":
                    continue
                if interval == "first":
                    labels = labels[:1]
                elif interval == "last":
                    labels = labels[-1:]
        for label in labels:
            assignments.append(
                {"time": label, "collection": item["collection"], "row": item["row"]}
            )
    frame = pd.DataFrame(assignments, columns=pd.Index(["time", "collection", "row"]))
    grouped = cast(
        "pd.Series",
        frame.groupby(["time", "collection"], sort=False, dropna=False)["row"].agg(
            list
        ),
    )
    slots = grouped.rename("indices").reset_index()
    return slots


def select_windows(
    slots: pd.DataFrame,
    times: pd.DatetimeIndex,
    collections: Sequence[str],
    window: int,
    step: int,
    missing: str,
    *,
    trim: bool = True,
) -> list[pd.DataFrame]:
    """Select consecutive windows, retaining explicit empty collection/slot cells.

    Completeness describes available observations, never the validity of pixels.
    Static rows accompany windows once and do not fill dated collection gaps.
    With trim=True, discard unoccupied leading/trailing slots after interval
    assignment. Set False when an explicit date range defines those slots.
    """
    dated = slots.loc[slots["time"].notna()]
    static = slots.loc[slots["time"].isna()]

    # Static-only groups have one raw window, never an invented calendar.
    if dated.empty:
        if (
            window == 1
            and len(static)
            and set(collections) <= set(static["collection"])
        ):
            return [static.copy()]
        return []

    if trim:
        times = pd.DatetimeIndex(
            times[(times >= dated["time"].min()) & (times <= dated["time"].max())]
        )

    lookup = {}
    for time, collection, indices in dated.itertuples(index=False, name=None):
        lookup[(time, collection)] = indices
    static_names = set(static["collection"]) - set(dated["collection"])
    windows = []
    for start in range(0, len(times) - window + 1, step):
        labels = times[start : start + window]
        records = []
        complete = True
        for collection in collections:
            if collection in static_names:
                continue
            present = []
            for label in labels:
                indices = lookup.get((label, collection), [])
                records.append(
                    {"time": label, "collection": collection, "indices": indices}
                )
                present.append(bool(indices))
            if not all(present):
                complete = False
        if not complete and missing == "drop":
            continue
        frame = pd.DataFrame(
            records, columns=pd.Index(["time", "collection", "indices"])
        )
        frame = pd.concat([frame, static], ignore_index=True)
        windows.append(frame)
    return windows
