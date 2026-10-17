"""Resolve model time bins against the acquisitions in each spatial row."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import orjson
import pandas as pd

from geosave.geodata.core.vector.samples import GeoSamples
from geosave.geodata.stac.translate.item import read_instant, read_timespan
from geosave.geodata.warnings import DroppedRowsWarning

if TYPE_CHECKING:
    import geopandas as gpd
    import pystac
    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core.vector.samples import Window

__all__: list[str] = []

_REDUCERS = {"first", "last", "mean", "median", "sum", "min", "max", "std", "var"}


def read_slot(
    slot: str,
) -> tuple[pd.DateOffset | pd.Timedelta, pd.DateOffset | pd.Timedelta]:
    """Read a relative bin: -1W covers the seven days preceding the reference."""
    match = re.fullmatch(r"([+-]?\d+)(min|[shDWMY])", slot)
    if match is None:
        raise ValueError(
            f"invalid slot {slot!r}; use an integer and s, min, h, D, W, M or Y"
        )
    position, unit = match.groups()
    count = int(position)
    if unit == "M":
        return pd.DateOffset(months=count), pd.DateOffset(months=count + 1)
    if unit == "Y":
        return pd.DateOffset(years=count), pd.DateOffset(years=count + 1)
    width = cast("pd.Timedelta", pd.Timedelta(1, unit=unit))
    return count * width, (count + 1) * width


def read_windows(
    windows: Mapping[str, Window],
) -> dict[str, list[tuple[pd.DateOffset | pd.Timedelta, pd.DateOffset | pd.Timedelta]]]:
    """Validate collection settings and parse their relative bin boundaries."""
    if not windows:
        raise ValueError("windows must name at least one collection")
    offsets = {}
    for name, config in windows.items():
        if set(config) - {"slots", "reduce", "fill"}:
            raise ValueError(f"unknown window settings for {name!r}")
        slots = config.get("slots")
        if (
            not isinstance(slots, list)
            or not slots
            or any(not isinstance(slot, str) for slot in slots)
        ):
            raise ValueError("slots must be a nonempty list of time-bin strings")
        if config.get("reduce") is not None and config.get("reduce") not in _REDUCERS:
            raise ValueError(f"unknown reduce method for {name!r}")
        if config.get("fill") not in (None, "interpolate"):
            raise ValueError("fill must be interpolate or omitted")
        offsets[name] = [read_slot(slot) for slot in slots]
    return offsets


def select_bins(
    name: str,
    items: list[pystac.Item],
    bounds: list[tuple[pd.Timestamp, pd.Timestamp]],
    config: Window,
) -> tuple[list[pystac.Item], dict] | None:
    """Select complete bins for one collection, returning Items and a read recipe."""
    for index in range(1, len(bounds)):
        if bounds[index][0] < bounds[index - 1][1]:
            raise ValueError("slots must be ordered and non-overlapping")
    spans = [read_timespan(item) for item in items]
    positions = []
    for start, end in bounds:
        matches = []
        for index, span in enumerate(spans):
            if span is not None and start <= span[0] and span[1] < end:
                matches.append(index)
        if len(matches) > 1 and config.get("reduce") is None:
            raise ValueError(
                f"{name!r} has multiple acquisitions in a bin; specify reduce"
            )
        positions.append(matches)
    present = [index for index, matches in enumerate(positions) if matches]
    if len(present) != len(bounds):
        if (
            config.get("fill") != "interpolate"
            or not present
            or present[0] != 0
            or present[-1] != len(bounds) - 1
        ):
            return None
    used = set()
    for matches in positions:
        used.update(matches)
    order = sorted(used)
    remap = {index: position for position, index in enumerate(order)}
    slots = []
    for (start, end), matches in zip(bounds, positions, strict=True):
        slots.append(
            [
                start.isoformat(),
                end.isoformat(),
                [remap[index] for index in matches],
            ]
        )
    plan = {
        "slots": slots,
        "reduce": config.get("reduce"),
        "fill": config.get("fill"),
    }
    return [items[index] for index in order], plan


def frame(
    samples: gpd.GeoDataFrame,
    *,
    windows: Mapping[str, Window],
    reference: str | None = None,
    stride: int | None = None,
) -> GeoDataFrame[GeoSamples]:
    """Select complete, newest-first frames and persist their lazy read recipe."""
    offsets = read_windows(windows)
    if stride is None:
        stride = max(len(slots) for slots in offsets.values())
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    records = []
    dropped = 0
    for _, row in samples.iterrows():
        items = row.gs.items
        required = set(windows)
        if reference is not None:
            required.add(reference)
        if missing := required - set(items):
            raise ValueError(f"unknown collections: {sorted(missing)}")
        clock = [reference] if reference is not None else list(windows)
        instants = set()
        for name in clock:
            for item in items[name]:
                instant = read_instant(item)
                if instant is not None:
                    instants.add(instant)
        if not instants:
            dropped += 1
        for instant in sorted(instants, reverse=True)[::stride]:
            selected = {}
            recipe = {}
            for name, bins in offsets.items():
                bounds = [(instant + start, instant + end) for start, end in bins]
                match = select_bins(name, items[name], bounds, windows[name])
                if match is None:
                    break
                selected[name], recipe[name] = match
            if len(recipe) != len(windows):
                dropped += 1
                continue
            if reference is not None and reference not in windows:
                matches = [
                    item for item in items[reference] if read_instant(item) == instant
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "reference timestamp has multiple acquisitions; prepare a unique reference series"
                    )
                selected[reference] = matches
            # Timeless auxiliary rasters do not define or constrain the clock.
            for name, entries in items.items():
                if name not in selected and all(
                    read_timespan(item) is None for item in entries
                ):
                    selected[name] = entries
            record = row.gs._to_frame(selected, timespan=(instant, instant))
            record["reference_time"] = instant
            record["frame"] = orjson.dumps(recipe).decode()
            records.append(record)
    if dropped:
        warnings.warn(
            f"dropped {dropped} incomplete frames or samples without reference times",
            DroppedRowsWarning,
            stacklevel=2,
        )
    return GeoSamples.from_rows(
        records, crs=samples.crs, geometry=samples.geometry.name, attrs=samples.attrs
    )
