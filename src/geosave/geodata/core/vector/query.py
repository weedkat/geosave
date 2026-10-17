"""Shared spatial and temporal selection rules for vector rows and frames.

A span is closed at both ends. A missing bound is open, so a row or Item
stating no time overlaps every span.
"""

from __future__ import annotations

# Internal implementation; query through the vector accessor.
__all__: list[str] = []

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geom import Geometry

from geosave.geodata.utils.datetime import AnchorDatetime, parse_daterange
from geosave.geodata.utils.geometry import SomeGeometry, to_shapely

if TYPE_CHECKING:
    from datetime import datetime

    import geopandas as gpd
    import pystac
    from shapely.geometry.base import BaseGeometry

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core.anchor import GeoAnchor

type SpatialPredicate = Literal[
    "intersects", "within", "contains", "covers", "covered_by"
]

_PREDICATES = get_args(SpatialPredicate.__value__)


def select(
    rows: gpd.GeoDataFrame,
    where: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree | SomeGeometry | None,
    *,
    when: AnchorDatetime | None,
    predicate: SpatialPredicate,
) -> GeoDataFrame:
    """Resolve a vector query and select its rows; see `GeoVector.query`."""
    from geosave.geodata.core.anchor import GeoAnchor

    if predicate not in _PREDICATES:
        raise ValueError(
            f"predicate must be one of {sorted(_PREDICATES)}, got {predicate!r}"
        )

    if isinstance(where, xr.DataArray | xr.Dataset | xr.DataTree):
        where = cast("GeoAnchor", where.gs.anchor)
    timespan = None
    if isinstance(where, GeoAnchor):
        ground = where.geobox.extent
        timespan = where.timespan
    elif isinstance(where, Geometry) or where is None:
        ground = where
    else:
        ground = Geometry(to_shapely(where), crs="EPSG:4326")
    if when is not None:
        timespan = parse_daterange(when)

    selected = rows
    if ground is not None and not selected.empty:
        selected = select_by_geometry(
            selected, ground.to_crs(rows.gs.crs).geom, predicate
        )
    if timespan is not None and not selected.empty:
        selected = select_by_timespan(selected, timespan)

    return cast("GeoDataFrame", selected.copy())


# Columns a vector row dates itself by, as STAC names them.
TIME_COLUMNS = ("datetime", "start_datetime", "end_datetime")


def to_naive_utc(instants: pd.Series) -> pd.Series:
    """Read instants as naive UTC timestamps, NaT where absent."""
    stamps = pd.to_datetime(instants, utc=True, format="mixed")
    return stamps.dt.tz_convert(None)


def read_row_spans(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Read each row's first and last instant as naive UTC.

    A row states `datetime`, or `start_datetime` and `end_datetime`. A missing
    bound falls back to `datetime`; without either it stays NaT, open.

    Args:
        rows: Vector rows with any of the STAC time columns.

    Returns:
        Start instants and end instants, indexed as `rows`.

    Examples:
        >>> starts, ends = read_row_spans(items)
        >>> starts.dt.strftime("%Y-%m-%d").tolist()
        ['2024-03-08', '2024-03-28']
    """
    instants_by_column = {}
    for column in TIME_COLUMNS:
        if column in rows:
            instants_by_column[column] = to_naive_utc(cast("pd.Series", rows[column]))
    no_instants = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    nominal_instants = instants_by_column.get("datetime", no_instants)
    start_instants = instants_by_column.get("start_datetime", nominal_instants)
    end_instants = instants_by_column.get("end_datetime", nominal_instants)
    return start_instants.fillna(nominal_instants), end_instants.fillna(
        nominal_instants
    )


def read_item_spans(
    items: Iterable[pystac.Item | Mapping[str, Any]],
) -> tuple[pd.Series, pd.Series]:
    """Read each Item's first and last instant as naive UTC, NaT when timeless.

    Args:
        items: Items or their dicts, in any order.

    Returns:
        Start instants and end instants, in Item order.
    """
    from geosave.geodata.stac.translate import item as stac_item

    start_instants = []
    end_instants = []
    for item in items:
        timespan = stac_item.read_timespan(item)
        start_instants.append(pd.NaT if timespan is None else timespan[0])
        end_instants.append(pd.NaT if timespan is None else timespan[1])
    return (
        pd.Series(pd.to_datetime(start_instants), dtype="datetime64[ns]"),
        pd.Series(pd.to_datetime(end_instants), dtype="datetime64[ns]"),
    )


def find_overlapping(
    start_instants: pd.Series,
    end_instants: pd.Series,
    timespan: tuple[datetime, datetime] | None,
) -> np.ndarray:
    """Mark the spans sharing at least one instant with `timespan`.

    Args:
        start_instants: First instant of each span, NaT where open.
        end_instants: Last instant of each span, NaT where open.
        timespan: Closed interval, naive UTC. None overlaps every span.

    Returns:
        Boolean mask in span order.

    Examples:
        >>> find_overlapping(starts, ends, (datetime(2024, 3, 20), datetime(2024, 4, 1)))
        array([False,  True, False])
    """
    if timespan is None:
        return np.ones(len(start_instants), dtype=bool)
    first_instant = pd.Timestamp(timespan[0])
    last_instant = pd.Timestamp(timespan[1])
    starts_before_end = start_instants.isna() | (start_instants <= last_instant)
    ends_after_start = end_instants.isna() | (end_instants >= first_instant)
    return (starts_before_end & ends_after_start).to_numpy()


def select_by_geometry(
    rows: gpd.GeoDataFrame, geometry: BaseGeometry, predicate: SpatialPredicate
) -> gpd.GeoDataFrame:
    """Keep the rows whose geometry relates to `geometry` as `predicate` says.

    Args:
        rows: Vector rows, on the CRS `geometry` is in.
        geometry: Shape the predicate reads each row against.
        predicate: Exact relation from each row to `geometry`.

    Returns:
        Matching rows in source order.
    """
    candidate_positions = sorted(set(rows.sindex.query(geometry)))
    candidate_rows = rows.iloc[candidate_positions]
    matches = getattr(candidate_rows.geometry, predicate)(geometry)
    return cast("gpd.GeoDataFrame", candidate_rows.loc[matches])


def select_by_timespan(
    rows: gpd.GeoDataFrame, timespan: tuple[datetime, datetime]
) -> gpd.GeoDataFrame:
    """Keep the rows whose span overlaps `timespan`, open bounds included.

    Args:
        rows: Vector rows with any of the STAC time columns.
        timespan: Closed interval, naive UTC.

    Returns:
        Overlapping rows in source order.
    """
    start_instants, end_instants = read_row_spans(rows)
    overlapping = find_overlapping(start_instants, end_instants, timespan)
    return cast("gpd.GeoDataFrame", rows.loc[overlapping])
