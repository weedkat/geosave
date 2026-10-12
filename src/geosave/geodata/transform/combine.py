"""Combining several rasters into one.

`merge_bands` joins bands already on one grid, `concat_time` lays
observations end to end, and `mosaic` lays rasters side by side.

Examples:
    An AOI spanning two granules, with radar bands for the same ground::

        scene = mosaic([granule_a, granule_b])
        image = merge_bands([scene, radar])
        series = concat_time([january, february, march])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox, geobox_union_conservative

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import (
    CRS_COORDINATE,
    SPATIAL_DIMENSIONS,
    TIME_COORDINATE,
)
from geosave.geodata.core.constructors import write_cf_axes
from geosave.geodata.core.stack import stack
from geosave.geodata.transform import warp

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from geosave.geodata import Dataset, DataTree

# How non-time coordinates align, forwarded to `xr.concat`'s own `join`.
type Join = Literal["outer", "inner", "left", "right", "exact", "override"]

# How time-constant variables are checked, forwarded to `xr.concat`'s `compat`.
type Compat = Literal[
    "identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"
]

# Which raster an overlapping pixel is taken from — a whole pixel, never a blend.
type MosaicMethod = Literal["first", "last"]


def merge_bands(rasters: Sequence[xr.Dataset]) -> Dataset:
    """Combine uniquely named bands without aligning or broadcasting observations.

    Args:
        rasters: Non-empty rasters on one exact grid, with equal coordinates
            and the same dimension order for every band. Band names must be unique.

    Returns:
        Dataset with bands in input order, attrs joined by `attrs.combine`.

    Raises:
        ValueError: Inputs are empty, bands collide, or grids, coordinates, or
            dimensions differ. Align grids and select observations before merging.

    Examples:
        >>> image = merge_bands([optical[["red", "nir"]], radar[["VV", "VH"]]])
    """
    if not rasters or any(not data.data_vars for data in rasters):
        raise ValueError("merge_bands requires non-empty rasters")

    reference = rasters[0]
    grid = reference.gs.geobox
    if not isinstance(grid, GeoBox):
        raise ValueError("merge_bands requires a regular georeferenced grid")
    reference_coords = reference.coords.to_dataset()
    reference_dims = next(iter(reference.data_vars.values())).dims

    # Nothing is aligned or broadcast, so every raster must already match.
    seen: set[str] = set()
    matched = []
    for data in rasters:
        if data.gs.geobox != grid:
            raise ValueError("merge_bands requires the same grid; align rasters first")
        # The grids are equal, so readers differ only in how they spell it: the
        # CRS coordinate's CF placeholder (0 or the EPSG code) and the x/y attrs
        # (bare from rioxarray, CF from odc-geo). Spell both one way.
        if CRS_COORDINATE in data.coords and CRS_COORDINATE in reference.coords:
            data = data.assign_coords(
                {CRS_COORDINATE: reference[CRS_COORDINATE].variable}
            )
        if grid.crs is not None:
            data = write_cf_axes(data, grid)
        matched.append(data)
        if not data.coords.to_dataset().equals(reference_coords):
            raise ValueError(
                "merge_bands requires equal coordinates; select observations explicitly"
            )
        if any(band.dims != reference_dims for band in data.data_vars.values()):
            raise ValueError(
                "merge_bands requires equal band dimensions; broadcasting is not "
                "supported"
            )

        names = {str(name) for name in data.data_vars}
        duplicates = sorted(seen & names)
        if duplicates:
            raise ValueError(
                f"Duplicate bands {duplicates}; rename bands before merging"
            )
        seen |= names

    merged = xr.merge(
        matched, join="exact", compat="equals", combine_attrs=attrs.combine
    )
    return cast("Dataset", merged)


def concat_time[T: xr.DataArray | xr.Dataset | xr.DataTree](
    rasters: Sequence[T],
    *,
    join: Join = "exact",
    compat: Compat = "equals",
    data_vars: str | Iterable[str] = "all",
    coords: str | Iterable[str] = "different",
    fill_value: float | int | Mapping[str, float | int] | None = None,
) -> T:
    """Extend the time axis by laying rasters end to end.

    Only `time` is extended. `join="exact"` requires every other coordinate
    to already agree, refusing otherwise; a looser `join` instead aligns them,
    padding what one raster lacks with `fill_value`.

    Args:
        rasters: Datasets, DataArrays, or DataTrees sharing variables, grid and dtype,
            each carrying a `time` coordinate, at least one. A DataTree
            recurses group by group.
        join: How non-time coordinates are aligned, forwarded to `xr.concat`:
            `"exact"`, `"outer"`, `"inner"`, `"left"`, or `"right"`.
        compat: How non-time-varying data variables are checked for
            agreement, forwarded to `xr.concat`.
        data_vars: Which data variables concatenate along `time` rather than
            being taken from the first raster, forwarded to `xr.concat`.
        coords: Which coordinates concatenate along `time` rather than being
            taken from the first raster, forwarded to `xr.concat`.
        fill_value: Value written where `join` pads a raster that does not
            reach a position, scalar or per-variable. None reads what each
            variable already carries; only consulted when `join` is not
            `"exact"`.

    Returns:
        New object of the kind given, spanning every raster's time labels in
        the order the rasters were given. Nothing is reordered, so sort the
        result with `.sortby("time")` where a chronological axis is wanted —
        resampling, interpolation and label-based selection each require one.

    Raises:
        ValueError: `rasters` is empty, the stacks hold different groups, they
            carry different variables or CRSs, `join` pads a variable without
            a fill value and none is given, or the rasters differ in a way
            `join`/`compat` refuses.

    Examples:
        >>> series = concat_time([january, february, march])

        Rasters that arrived out of order lay down in that order all the same.
        Sort the result where the axis has to be chronological; a DataTree
        carries no `sortby`, so a stack sorts group by group::

            >>> concat_time([march, january]).sortby("time")
            >>> stack({n: g.sortby("time") for n, g in tree.gs.rasters.items()})
    """
    if not rasters:
        raise ValueError("no rasters to concatenate; pass at least one")

    options: _ConcatOptions = {
        "join": join,
        "compat": compat,
        "data_vars": data_vars,
        "coords": coords,
        "fill_value": fill_value,
    }
    if isinstance(rasters[0], xr.DataTree):
        result = _concat_stacks(cast("Sequence[xr.DataTree]", rasters), options)
    else:
        arrays = cast("Sequence[xr.DataArray | xr.Dataset]", rasters)
        result = _concat_rasters(arrays, options)
    return cast("T", result)


class _ConcatOptions(TypedDict):
    """Options `concat_time` forwards to every concatenation it makes."""

    join: Join
    compat: Compat
    data_vars: str | Iterable[str]
    coords: str | Iterable[str]
    fill_value: float | int | Mapping[str, float | int] | None


def _concat_stacks(trees: Sequence[xr.DataTree], options: _ConcatOptions) -> DataTree:
    """Lay stacks end to end, group by group."""
    groups = trees[0].gs.groups
    for ordinal, tree in enumerate(trees[1:], start=1):
        if set(tree.gs.groups) != set(groups):
            raise ValueError(
                f"stack {ordinal} holds {sorted(tree.gs.groups)} and stack 0 "
                f"holds {sorted(groups)}, so they name different rasters to lay "
                f"end to end; concatenate stacks carrying the same groups"
            )

    rasters = {}
    for name in groups:
        series = [tree.gs.rasters[name] for tree in trees]
        rasters[name] = _concat_rasters(series, options)

    combined = stack(rasters)
    combined.attrs = attrs.combine([tree.attrs for tree in trees])
    return combined


def _concat_rasters[R: xr.DataArray | xr.Dataset](
    rasters: Sequence[R], options: _ConcatOptions
) -> R:
    """Lay rasters or bands end to end along time."""
    # `join` settles the grid axes but never reads a CRS, so check it here.
    crs_names = sorted({str(raster.gs.crs_name) for raster in rasters})
    if len(crs_names) > 1:
        raise ValueError(
            f"the rasters carry {crs_names}, so laying them end to end would "
            f"stack pixels naming different ground; reproject them onto one CRS "
            f"first"
        )

    reference = rasters[0]
    if isinstance(reference, xr.Dataset):
        names = set(reference.data_vars)
        for raster in rasters[1:]:
            if not isinstance(raster, xr.Dataset) or set(raster.data_vars) != names:
                raise ValueError(
                    "concat_time requires the same variables in every raster"
                )

    # A position one raster does not reach takes each variable's own fill.
    fill_value = options["fill_value"]
    if fill_value is None and options["join"] != "exact":
        fill_value = reference.gs.nodata
        stated = fill_value.values() if isinstance(fill_value, dict) else [fill_value]
        if None in stated:
            raise ValueError(
                "a variable carries no fill value, so a position one raster does "
                "not reach has nothing to hold; pass fill_value or write one with "
                "raster.gs.write_nodata(...) first"
            )

    # xr.concat overloads on one element type, which a union of both does not select.
    combined = xr.concat(
        cast("Sequence[xr.Dataset]", rasters),
        dim=TIME_COORDINATE,
        join=options["join"],
        compat=options["compat"],
        data_vars=options["data_vars"],
        coords=options["coords"],
        fill_value=fill_value,
        combine_attrs=attrs.combine,
    )
    return cast("R", combined)


def mosaic(rasters: Sequence[xr.Dataset], *, method: MosaicMethod = "first") -> Dataset:
    """Lay several rasters onto the grid they jointly cover.

    Only `y`/`x` are laid out; every other coordinate must already agree
    across `rasters`. A pixel is always taken whole from one raster, never
    blended across two.

    Args:
        rasters: Datasets in one CRS and resolution, at least one, in
            preference order.
        method: Which raster an overlapping pixel is taken from. `"first"`
            takes the earliest raster carrying a value there, leaving
            absence only where none does; `"last"` takes the latest.

    Returns:
        New Dataset covering every raster's ground, absent only where no
        raster carried a value.

    Raises:
        ValueError: `rasters` is empty, a variable is missing from a raster
            or carries no fill value, one raster sits on no locatable grid, or
            no one grid holds them all because they disagree on CRS,
            resolution or pixel phase, or their non-spatial coordinates differ.

    Examples:
        >>> scene = mosaic([granule_32TNS, granule_32TPS])
    """
    if not rasters:
        raise ValueError("no rasters to mosaic; pass at least one")

    # Only y/x are laid out, so every other coordinate must already agree.
    coordinates = [_non_spatial_coords(raster) for raster in rasters]
    if any(not coords.equals(coordinates[0]) for coords in coordinates[1:]):
        raise ValueError("mosaic requires equal non-spatial coordinates")

    fills = _mosaic_fills(rasters)

    # One raster fills another's gaps, so their band attrs and nodata must agree.
    band_attrs = {}
    for name in fills:
        try:
            band_attrs[name] = attrs.combine([raster[name].attrs for raster in rasters])
        except ValueError as error:
            raise ValueError(f"{name!r} cannot be mosaicked: {error}") from error

    # Each pixel comes whole from the most preferred raster holding a value there.
    grid = _joint_grid(rasters)
    aligned = [warp.reproject(raster, grid) for raster in rasters]
    if method == "last":
        aligned.reverse()
    merged = aligned[0]
    for source in aligned[1:]:
        gaps = {}
        for name, fill in fills.items():
            band = merged[name]
            # NaN never equals itself, so a NaN fill is found through isnull.
            gaps[name] = band.isnull() if np.isnan(fill) else band == fill
        merged = xr.where(xr.Dataset(gaps), source, merged)

    # xr.where drops band attrs, so each band takes back what the rasters agree on.
    bands = {}
    for name, stated in band_attrs.items():
        band = merged[name].copy(deep=False)
        band.attrs = stated
        bands[name] = band
    merged = merged.assign(bands)
    merged.attrs = attrs.combine([raster.attrs for raster in rasters])
    return cast("Dataset", merged.gs.write_crs())


def _non_spatial_coords(raster: xr.Dataset) -> xr.Dataset:
    """Collect the coordinates spanning neither spatial axis, CRS aside."""
    coords = {}
    for name, coord in raster.coords.items():
        spatial = set(coord.dims) & set(SPATIAL_DIMENSIONS)
        if name != CRS_COORDINATE and not spatial:
            coords[name] = coord.variable
    return xr.Dataset(coords=coords)


def _mosaic_fills(rasters: Sequence[xr.Dataset]) -> dict[str, float | int]:
    """Read each variable's fill value, refusing a raster that cannot be laid out.

    Ground no raster covers holds each variable's fill, so every variable of
    the first raster must be in every raster and carry one.

    Returns:
        {
            variable name: its fill value,
        }
    """
    names = rasters[0].gs.variables
    for raster in rasters:
        # A variable one raster lacks has no pixels to lay out there.
        missing = sorted(set(names) ^ set(map(str, raster.data_vars)))
        if missing:
            raise ValueError(
                f"{missing} are not in every raster's variables; select the same "
                f"variables from each raster first"
            )
        unfilled = [name for name in names if raster[name].gs.nodata is None]
        if unfilled:
            raise ValueError(
                f"{unfilled} carries no fill value, so a pixel no raster covers "
                f"has nothing to hold; write one with raster.gs.write_nodata(...) "
                f"first"
            )
    return {name: rasters[0][name].gs.nodata for name in names}


def _joint_grid(rasters: Sequence[xr.Dataset]) -> GeoBox:
    """Read the one grid every raster lays onto without moving a pixel."""
    grids: list[GeoBox] = []
    for ordinal, raster in enumerate(rasters):
        geobox = raster.gs.geobox
        if not isinstance(geobox, GeoBox):
            raise ValueError(
                f"raster {ordinal} sits on no locatable grid, so no ground says "
                f"where it overlaps the rest; open it georeferenced first"
            )
        grids.append(geobox)

    # One lattice holds them only where CRS and resolution agree.
    first = grids[0]
    for ordinal, grid in enumerate(grids[1:], start=1):
        mismatch = None
        if grid.crs != first.crs:
            mismatch = (
                f"CRS {rasters[ordinal].gs.crs_name} against {rasters[0].gs.crs_name}"
            )
        elif grid.resolution != first.resolution:
            mismatch = f"resolution {grid.resolution} against {first.resolution}"
        if mismatch is not None:
            raise ValueError(
                f"raster {ordinal} carries {mismatch} of raster 0, so no one grid "
                f"holds both; align them onto one grid first"
            )

    # Sharing a CRS and resolution leaves only pixel phase to refuse a union over.
    try:
        union = geobox_union_conservative(grids)
    except ValueError as error:
        raise ValueError(
            "the rasters lay their pixel edges on different lines, so no one grid "
            "holds them at that phase; align them onto one grid first"
        ) from error
    return union
