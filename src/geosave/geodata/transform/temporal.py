"""Building and collapsing a raster's time axis.

`select` keeps overlapping time steps, `concat` lays rasters end to end along
time, `resample` cuts the axis into
calendar buckets, `reduce` collapses them, and `interpolate` fills the
buckets that observed nothing.

Examples:
    A series assembled from monthly files, then smoothed::

        series = concat([january, february, march])
        monthly = reduce(resample(series, "MS"), "median")
        gapless = interpolate(monthly, "nearest")
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.resample import DataArrayResample, DatasetResample

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import TIME_COORDINATE
from geosave.geodata.core.raster.create import tree
from geosave.geodata.transform import nodata
from geosave.geodata.utils.datetime import AnchorDatetime, freq_offset, parse_daterange

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime as dt, timedelta

    from geosave.geodata import DataArray, Dataset
    from geosave.geodata.utils.datetime import Freq

# How non-time coordinates align, forwarded to `xr.concat`'s own `join`.
type Join = Literal["outer", "inner", "left", "right", "exact", "override"]

# How an absent cell reads its value off the observed buckets around it.
type Interpolation = Literal["nearest", "linear"]

# How long a run of absent cells may be, as pandas reads a duration.
type Gap = str | pd.Timedelta | timedelta | np.timedelta64

# Named reducers a bucket may collapse with.
type Reducer = Literal[
    "mean", "median", "sum", "min", "max", "std", "var", "first", "last"
]

# Named reducers xarray collapses with, mapped to their CF Appendix E cell method.
_CELL_METHODS: Mapping[str, str] = {
    "first": "point",
    "last": "point",
    "min": "minimum",
    "max": "maximum",
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "std": "standard_deviation",
    "var": "variance",
}


def select[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T, when: AnchorDatetime | None
) -> T:
    """Keep time steps whose intervals overlap an inclusive time span.

    Args:
        data: Raster or tree of rasters. Groups without a time dimension stay
            unchanged. Only time coordinates are inspected; pixels remain lazy.
        when: Span accepted by `parse_daterange`, or None to keep every step.

    Returns:
        Same kind of raster with matching steps in source order. Composite
        steps use `start_datetime` and `end_datetime` coordinates when present,
        including scalar bounds; otherwise each step spans its `time` alone.
        Missing bounds are open.

    Examples:
        >>> january = select(observations, "2024-01")
        >>> january.sizes["time"]
        2
    """
    from geosave.geodata.core.raster.datatree import map_groups

    if when is None:
        return data
    if isinstance(data, xr.DataTree):
        return cast("T", map_groups(data, lambda raster: select(raster, when)))
    if "time" not in data.dims:
        return data

    first, last = parse_daterange(when)
    starts = data.coords.get("start_datetime", data["time"])
    ends = data.coords.get("end_datetime", data["time"])
    overlapping = (starts.isnull() | (starts <= np.datetime64(last))) & (
        ends.isnull() | (ends >= np.datetime64(first))
    )
    overlapping = overlapping.broadcast_like(data["time"])

    return cast("T", data.isel(time=np.flatnonzero(overlapping.values)))


def concat[T: xr.DataArray | xr.Dataset | xr.DataTree](
    rasters: Sequence[T],
    *,
    join: Join = "exact",
    fill: float | int | Mapping[str, float | int] | None = None,
) -> T:
    """Extend the time axis by laying rasters end to end.

    Only `time` is extended. `join="exact"` requires every other coordinate
    to already agree; a looser `join` aligns them, padding what one raster
    lacks with `fill`. Other `xr.concat` options: call it directly.

    Args:
        rasters: Datasets, DataArrays, or DataTrees sharing variables, grid and
            dtype, each carrying a `time` coordinate, at least one. A DataTree
            recurses group by group.
        join: How non-time coordinates are aligned, forwarded to `xr.concat`:
            `"exact"`, `"outer"`, `"inner"`, `"left"`, or `"right"`.
        fill: Value written where `join` pads a raster that does not reach a
            position, scalar or per-variable. None reads what each variable
            already carries; only consulted when `join` is not `"exact"`.

    Returns:
        New object of the kind given, spanning every raster's time labels in
        the order the rasters were given. Nothing is reordered, so sort the
        result with `.sortby("time")` where a chronological axis is wanted —
        resampling, interpolation and label-based selection each require one.

    Raises:
        ValueError: `rasters` is empty, the trees hold different groups, they
            lack a time coordinate, carry different variables, dtypes or CRSs,
            `join` pads a variable without
            a fill value and none is given, or the rasters differ in a way
            `join` refuses.

    Examples:
        >>> series = concat([january, february, march])

        Rasters that arrived out of order lay down in that order all the same.
        Sort the result where the axis has to be chronological; a DataTree
        carries no `sortby`, so a tree sorts group by group::

            >>> concat([march, january]).sortby("time")
            >>> tree({n: g.sortby("time") for n, g in scene.gs.rasters.items()})
    """
    if not rasters:
        raise ValueError("no rasters to concatenate; pass at least one")

    first = rasters[0]
    if isinstance(first, xr.DataTree):
        return cast(
            "T", _concat_trees(cast("Sequence[xr.DataTree]", rasters), join, fill)
        )

    _validate_concat(cast("Sequence[xr.DataArray | xr.Dataset]", rasters))

    if join != "exact" and fill is None:
        stated = first.gs.nodata
        values = stated.values() if isinstance(stated, dict) else [stated]
        if None in values:
            raise ValueError(
                "a variable carries no fill value, so a position one raster does "
                "not reach has nothing to hold; pass fill or write one with "
                "raster.gs.write_nodata(...) first"
            )
        fill = cast("float | int | Mapping[str, float | int]", stated)

    # xr.concat overloads on one element type, which a union of both does not select.
    combined = xr.concat(
        cast("Sequence[xr.Dataset]", rasters),
        dim=TIME_COORDINATE,
        join=join,
        fill_value=fill,
        combine_attrs=attrs.combine_attrs,
    )
    return cast("T", combined)


def _concat_trees(
    trees: Sequence[xr.DataTree],
    join: Join,
    fill: float | int | Mapping[str, float | int] | None,
) -> xr.DataTree:
    """Concatenate corresponding groups and combine root metadata."""
    groups = trees[0].gs.groups
    for position, other in enumerate(trees[1:], start=1):
        if set(other.gs.groups) != set(groups):
            raise ValueError(
                f"tree {position} holds {sorted(other.gs.groups)} and tree 0 "
                f"holds {sorted(groups)}, so they name different rasters to "
                f"lay end to end; concatenate trees carrying the same groups"
            )
    series = {}
    for name in groups:
        series[name] = concat(
            [other.gs.rasters[name] for other in trees], join=join, fill=fill
        )
    combined = tree(series)
    combined.attrs = attrs.combine_attrs([other.attrs for other in trees])
    return combined


def _validate_concat(rasters: Sequence[xr.DataArray | xr.Dataset]) -> None:
    """Require dated rasters with matching CRSs, variables, and stored dtypes."""
    first = rasters[0]
    crs_names = sorted({str(raster.gs.crs_name) for raster in rasters})
    if len(crs_names) > 1:
        raise ValueError(
            f"the rasters carry {crs_names}, so laying them end to end would "
            f"stack pixels naming different ground; reproject them onto one CRS "
            f"first"
        )
    # xarray pads variables one raster lacks with NaN, so all must name the same.
    if isinstance(first, xr.Dataset):
        names = set(first.data_vars)
        for raster in rasters[1:]:
            if not isinstance(raster, xr.Dataset) or set(raster.data_vars) != names:
                raise ValueError("concat requires the same variables in every raster")

    for raster in rasters:
        if TIME_COORDINATE not in raster.coords:
            raise ValueError("concat requires a time coordinate on every raster")
        if isinstance(first, xr.Dataset):
            if first.dtypes != raster.dtypes:
                raise ValueError(
                    "concat requires matching variable dtypes; cast explicitly first"
                )
        elif not isinstance(raster, xr.DataArray) or first.dtype != raster.dtype:
            raise ValueError("concat requires matching dtypes; cast explicitly first")


@overload
def resample(
    data: xr.DataArray,
    freq: Freq,
    *,
    closed: Literal["left", "right"] | None = ...,
    label: Literal["left", "right"] | None = ...,
    origin: str | dt = ...,
    offset: str | timedelta | None = ...,
    skip_nodata: bool = ...,
) -> DataArrayResample: ...


@overload
def resample(
    data: xr.Dataset,
    freq: Freq,
    *,
    closed: Literal["left", "right"] | None = ...,
    label: Literal["left", "right"] | None = ...,
    origin: str | dt = ...,
    offset: str | timedelta | None = ...,
    skip_nodata: bool = ...,
) -> DatasetResample: ...


def resample(
    data: xr.DataArray | xr.Dataset,
    freq: Freq,
    *,
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    origin: str | dt = "start_day",
    offset: str | timedelta | None = None,
    skip_nodata: bool = True,
) -> DataArrayResample | DatasetResample:
    """Cut a time axis into calendar buckets, recording how they were cut.

    The `time` coordinate carries a `TimeSpec` from here on, saying what its
    labels stand for. No pixel moves, and no value changes but the nodata
    ones `skip_nodata` leaves NaN.

    Args:
        data: DataArray or Dataset carrying a `time` coordinate.
        freq: Target cadence — any pandas offset alias. The alias fixes where
            bucket edges fall: `"MS"` on month starts, `"5D"` every five days.
        closed: Which bucket edge is inclusive, `"left"` or `"right"`. None
            resolves as pandas does, by alias: a period-end one such as `"ME"`
            or `"W"` closes on the right, every other alias on the left.
        label: Which edge names the bucket, resolved the same way. The
            recorded `TimeSpec` carries whichever edge both resolved to.
        origin: Timestamp the edge grid is phased from, or a strategy name.
        offset: Shift added on top of `origin`.
        skip_nodata: Leave NaN where a variable carries its nodata value, so
            a reducer skips those pixels. False keeps the stored value, which
            every reducer then takes for a reading and blends into the result.

    Returns:
        xarray's resampler over the buckets, holding data whose `time`
        coordinate records the bucketing. Reduce it with any reducer xarray
        offers; a bucket covering no observation survives as absent.

    Raises:
        ValueError: `data` carries no `time` coordinate, or pandas does not
            know `freq`.

    Examples:
        >>> monthly = resample(daily, "MS").median()
        >>> monthly.time.dt.strftime("%Y-%m-%d").values.tolist()
        ['2024-01-01', '2024-02-01', '2024-03-01']
        >>> TimeSpec.from_attrs(monthly.time.attrs).time_freq
        'MS'
    """
    if TIME_COORDINATE not in data.coords:
        raise ValueError(
            f"{type(data).__name__} carries no {TIME_COORDINATE!r} coordinate, "
            f"so it holds no observations to bucket; label its time axis first"
        )

    alias = freq_offset(freq)
    timespec = attrs.TimeSpec.from_resample(
        alias, closed=closed, label=label, origin=origin, offset=offset
    )
    observations = nodata.to_nan(data) if skip_nodata else data
    # The spec rides the time coordinate through grouping onto the reduced axis.
    time_coord = attrs.stamp_attrs(observations[TIME_COORDINATE], timespec)
    return observations.assign_coords({TIME_COORDINATE: time_coord}).resample(
        {TIME_COORDINATE: alias},
        closed=closed,
        label=label,
        origin=origin,
        offset=offset,
    )


@overload
def reduce(resampler: DataArrayResample, method: Reducer) -> DataArray: ...


@overload
def reduce(resampler: DatasetResample, method: Reducer) -> Dataset: ...


@overload
def reduce(resampler: xr.Dataset, method: Reducer) -> Dataset: ...


@overload
def reduce(resampler: xr.DataArray, method: Reducer) -> DataArray: ...


def reduce(
    resampler: DataArrayResample | DatasetResample | xr.Dataset | xr.DataArray,
    method: Reducer,
) -> DataArray | Dataset:
    """Collapse each bucket into one value per cell.

    The resampler carries its own grouping, so nothing else says how to group.

    Args:
        resampler: Buckets from `resample`, or one Dataset/DataArray bin with
            a time dimension. For a single bin, first/last select a whole scene.
        method: Named reducer each bucket collapses with.

    Returns:
        New object of the kind the resampler holds, one position per bucket,
        preserving metadata and carrying the CF cell method it was collapsed with.

    Raises:
        ValueError: An input variable holds class codes and `method` is not
            a point sample of them. Checked before reducing pixels.

    Examples:
        >>> monthly = reduce(resample(daily, "MS"), "median")
        >>> monthly.sizes["time"], monthly.red.attrs["cell_methods"]
        (12, 'time: median')
    """
    cell_method = _CELL_METHODS[method]
    # Every group has the same variable metadata. Public iteration exposes it
    # without reducing pixels or depending on xarray's private resampler state.
    if isinstance(resampler, (xr.Dataset, xr.DataArray)):
        observations = resampler
    else:
        _, observations = next(iter(resampler))
    if isinstance(observations, xr.DataArray):
        variables = {observations.name: observations}
    else:
        variables = observations.data_vars

    # A class code neither averages nor ranks, so only a point sample keeps it.
    categorical = [
        str(name) for name, variable in variables.items() if variable.gs.is_categorical
    ]
    if categorical and cell_method != "point":
        raise ValueError(
            f"{categorical} hold class codes, which {method!r} either blends "
            f"into codes naming no class or orders as if they ranked; reduce "
            f"them with 'first' or 'last', select the variables you mean, or "
            f"drop the Legend first"
        )

    if isinstance(resampler, (xr.Dataset, xr.DataArray)):
        if method in ("first", "last"):
            buckets = resampler.isel(time=0 if method == "first" else -1, drop=True)
        else:
            buckets = getattr(nodata.to_nan(resampler), method)(
                dim="time", keep_attrs=True
            )
    else:
        buckets = getattr(resampler, method)(keep_attrs=True)

    # CF records on each variable how its values were collapsed along time.
    cell_method_attrs = {"cell_methods": f"{TIME_COORDINATE}: {cell_method}"}
    if isinstance(buckets, xr.DataArray):
        return cast("DataArray", buckets.assign_attrs(cell_method_attrs))
    described = {}
    for name, variable in buckets.data_vars.items():
        described[name] = variable.assign_attrs(cell_method_attrs)
    return cast("Dataset", buckets.assign(described))


@overload
def interpolate(
    data: xr.DataArray, method: Interpolation, *, max_gap: Gap | None = ...
) -> DataArray: ...


@overload
def interpolate(
    data: xr.Dataset, method: Interpolation, *, max_gap: Gap | None = ...
) -> Dataset: ...


def interpolate(
    data: xr.DataArray | xr.Dataset,
    method: Interpolation,
    *,
    max_gap: Gap | None = None,
) -> DataArray | Dataset:
    """Fill the cells a bucketed axis left absent, along time.

    `resample` emits every bucket its cadence covers, so one observing nothing
    is already on the axis, holding absence. What that becomes is settled cell
    by cell, so a partly clouded bucket fills only the cells it missed.

    Args:
        data: DataArray or Dataset whose `time` coordinate records the
            bucketing it was cut with, as `reduce` leaves it.
        method: How an absent cell reads its value. `"nearest"` repeats the
            nearest observed bucket, reaching a bucket with none on one side
            of it; `"linear"` interpolates between the observed buckets on
            either side, leaving a cell with only one side absent.
        max_gap: Longest run of absent cells to fill, measured between the
            observations either side of it, e.g. `"45D"`. None fills a run of
            any length. Bounding one needs `bottleneck` installed, which
            xarray reads run lengths through.

    Returns:
        New object of the kind given, on the same axis, its absent cells filled
        as `method` says. A cell observed once stays absent, having nothing to
        read between.

    Raises:
        ValueError: The `time` axis is absent or records no bucketing, or
            `method="linear"` would interpolate class codes.

    Examples:
        A cloudy January observed nothing, so its bucket is absent throughout.
        A sequence model wants a reading there:

        >>> gapless = interpolate(monthly, "nearest")
        >>> gapless.red.isnull().any().item()
        False
    """
    if TIME_COORDINATE not in data.coords:
        raise ValueError(
            f"{type(data).__name__} carries no {TIME_COORDINATE!r} coordinate, so "
            f"it holds no buckets to fill; resample its time axis first"
        )

    timespec = attrs.TimeSpec.from_attrs(data[TIME_COORDINATE].attrs)
    if timespec is None and not {"start_datetime", "end_datetime"} <= set(data.coords):
        raise ValueError(
            f"the {TIME_COORDINATE!r} axis names instants rather than buckets, so "
            f"no bucket observed nothing; resample it to a cadence first"
        )

    variables = {data.name: data} if isinstance(data, xr.DataArray) else data.data_vars
    categorical = [
        str(name) for name, variable in variables.items() if variable.gs.is_categorical
    ]
    if method == "linear" and categorical:
        raise ValueError(
            f"{categorical} hold class codes, which 'linear' blends into codes "
            f"naming no class; fill them with 'nearest', select the variables "
            f"you mean, or drop the Legend first"
        )

    # interpolate_na reads a whole time series per cell, so a lazy axis is one chunk.
    if TIME_COORDINATE in data.chunksizes:
        data = data.chunk({TIME_COORDINATE: -1})
    # Only the nearest reading reaches a bucket with none on one side of it.
    fill_value = "extrapolate" if method == "nearest" else np.nan
    gapless = data.interpolate_na(
        dim=TIME_COORDINATE,
        method=method,
        max_gap=max_gap,
        fill_value=fill_value,
    )
    # interpolate_na keeps every attr; nothing it fills changes what they say.
    return cast("DataArray | Dataset", gapless)
