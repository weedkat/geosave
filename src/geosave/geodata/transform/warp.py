"""Moving a raster's pixels onto another grid.

`odc.stac.load` warps at load time, given a geobox. This is the same
operation for a raster already in memory, which has no load to hook into.

Examples:
    A delivered DEM has to reach the imagery's grid before a model reads both,
    and a quicklook wants a CRS rather than an exact grid::

        dem = reproject(srtm, scene, resampling="bilinear")
        webmap = reproject(scene, "EPSG:3857")

    Several rasters reach one grid together, which is the only way to say what
    ground they jointly cover::

        optical, radar, dem = align([optical, radar, srtm], target=optical, extent="intersection")
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from math import isclose, lcm
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.crs import norm_crs
from odc.geo.geobox import (
    GeoBox,
    geobox_intersection_conservative,
    geobox_union_conservative,
)

from geosave import __path__ as _package_paths
from geosave.geodata.conventions import SPATIAL_DIMENSIONS, to_yx
from geosave.geodata.warnings import AssumedFillWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odc.geo import CRS, SomeCRS, SomeResolution

# Kernel names odc forwards to rasterio.warp.Resampling.
type Resampling = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
    "rms",
]

# What names the grid to warp onto, directly or by naming only its CRS.
type Target = GeoBox | xr.DataArray | xr.Dataset | xr.DataTree | SomeCRS

# Ground a set of aligned rasters covers.
type Extent = Literal["union", "intersection", "match"]

# Asking for a raster's own pixel size rather than the one the target names.
type Native = Literal["native"]
NATIVE: Native = "native"

# Kernels that carry a stored value across rather than blending several.
_VALUE_PRESERVING = frozenset({"nearest", "mode"})

# Reprojection arithmetic drifts a pixel size by a hair, which is not a new one.
_RATIO_TOLERANCE = 1e-6


def reproject[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T,
    target: Target,
    *,
    resampling: Resampling | Mapping[str, Resampling] = "nearest",
    resolution: SomeResolution | Native | None = None,
) -> T:
    """Warp pixels onto the grid a target names.

    A target grid is adopted whole — CRS, resolution, and extent — while a bare
    CRS only decides the projection, sizing the grid from the source. Every
    group of a tree lands on it, and data already there is returned untouched.

    Args:
        data: DataArray, Dataset, or DataTree whose pixels sit on a locatable
            grid.
        target: Grid to land on, a raster already on one, or a CRS.
        resampling: One GDAL kernel for every variable, or a mapping naming
            each variable's own, which `"*"` answers the rest of. Inside a
            tree a key names its group, as `"<group>/<variable>"`.
        resolution: Output pixel size. None takes the target's own, or the
            source's ground sampling where the target names only a CRS. A grid
            derived from a CRS snaps to the pixel lattice and covers the source
            whole, so it reaches past the source's own bounds. `"native"` keeps
            each variable's own pixel size instead, nesting it onto the target's
            lattice, so a tree's groups come out at their own resolutions.
            Nesting requires a shared CRS and one integer scale ratio on both axes.
            For a single array or Dataset with a bare CRS target, `"native"`
            uses ODC's automatic resolution. Preserving separate tree
            resolutions requires all groups and the target to share a CRS.

    Returns:
        New object of the same kind on the target grid, carrying its own
        `spatial_ref` and the CF semantics its axes earn. Pixels the source
        does not reach hold each variable's fill value.

    Warns:
        AssumedFillWarning: A variable carries no fill value, so its uncovered
            pixels take NaN where it is float, and its dtype's far end where
            it is integer: the maximum where unsigned, the minimum where
            signed.

    Raises:
        CRSError: `target` names no CRS pyproj recognises.
        KeyError: A `resampling` mapping names neither a variable nor `"*"`.
        ValueError: `data` or `target` sits on no locatable grid, `data` holds
            no data variable, `resolution` contradicts a target grid or stands
            in no whole-numbered ratio to it, a variable of `data` does not
            span its own grid, the kernel is unsupported, or `resampling`
            would blend a variable whose values are class codes.

    Notes:
        ODC's lazy reprojection can differ from eager GDAL results at chunk
        boundaries for interpolating kernels, including bilinear and cubic.
        Increasing chunk size alone does not guarantee identical pixels.

    Examples:
        The target grid comes from whatever already defines it — another
        raster, or the anchor the workspace loaded against:

        >>> dem = reproject(srtm, scene, resampling="bilinear")
        >>> dem.gs.geobox == scene.gs.geobox
        True

        A CRS decides only the projection, so the grid is sized from the
        source unless a resolution says otherwise:

        >>> reproject(scene, "EPSG:3857").gs.crs.epsg
        3857
        >>> reproject(scene, "EPSG:4326", resolution=0.001).gs.resolution.x
        0.001

        Resampling in place names a grid rather than a CRS, since a grid
        derived from a CRS is padded out to the lattice:

        >>> reproject(scene, scene.gs.geobox.zoom_out(2)).gs.geobox.shape
        Shape2d(x=4, y=4)

        Class codes must not be averaged, so a blending kernel refuses. A
        tree holding both imagery and labels names a kernel per variable:

        >>> reproject(landcover, scene, resampling="bilinear")
        Traceback (most recent call last):
        ValueError: ['landcover'] hold class codes, which ['bilinear'] would blend ...
        >>> kernels = {"*": "bilinear", "labels/mask": "mode"}
        >>> reproject(scene, target, resampling=kernels)
    """
    kernels = _read_kernels(data, resampling)
    if isinstance(data, xr.DataTree):
        return cast("T", _reproject_tree(data, target, kernels, resolution))
    source = _read_geobox(data)
    geobox = _resolve_target(source, target, resolution)
    return cast("T", _warp_raster(data, source, geobox, kernels))


def align[T: xr.DataArray | xr.Dataset | xr.DataTree](
    rasters: Sequence[T],
    *,
    target: Target,
    extent: Extent = "match",
    resampling: Resampling | Mapping[str, Resampling] = "nearest",
    resolution: SomeResolution | Native | None = None,
) -> tuple[T, ...]:
    """Warp several rasters onto one grid, so one window reads them all.

    Args:
        rasters: Rasters to align, at least one, each on a locatable grid.
        target: Explicit grid, reference raster, or CRS. A target tree must
            carry one shared grid; select a group when its grids differ.
        extent: Ground the aligned rasters cover. `"union"` reaches every
            raster, holding fill where one does not; `"intersection"` intersects
            their grid-aligned footprint envelopes, retaining partially covered
            boundary pixels; `"match"` adopts the reference's own.
        resampling: One GDAL kernel for every variable, or a mapping naming
            each variable's own, which `"*"` answers the rest of. Pixels move
            only where a raster does not already sit on the grid they align
            onto.
        resolution: Pixel size every raster lands at. None takes the
            reference's own. `"native"` keeps each raster's instead, landing
            them on nesting grids rather than one.

    Returns:
        One raster per input, in the order given, every one on the same exact
        GeoBox — or, under `"native"`, on grids covering one extent at their
        own pixel sizes.

    Raises:
        KeyError: A `resampling` mapping names neither a variable nor `"*"`.
        ValueError: `rasters` is empty, one sits on no locatable grid, `target`
            names no grid or CRS, the grid to align onto carries no CRS,
            `"intersection"` leaves none in common, or `"native"` finds pixel
            sizes in no whole-numbered ratio.

    Examples:
        A delivered DEM and a radar scene onto the optical grid, which one
        tiler then cuts together:

        >>> optical, radar, dem = align([optical, radar, srtm], target=optical)
        >>> {r.gs.geobox for r in (optical, radar, dem)} == {optical.gs.geobox}
        True

        Intersecting the grid-aligned extents trims the output. Partly covered
        pixels and source nodata can still leave fill values:

        >>> [r.gs.geobox.shape for r in align(scenes, target=optical, extent="intersection")]
        [Shape2d(x=180, y=180), Shape2d(x=180, y=180)]

        Keeping every raster's own pixel size lands them on nesting grids,
        which one ground extent still cuts together:

        >>> [r.gs.geobox.shape.x for r in align([b10, b20, b60], target=b10, resolution="native")]
        [384, 192, 64]
    """
    if target is None:
        raise ValueError("alignment requires an explicit target")

    geobox = _resolve_alignment_grid(
        rasters, target=target, extent=extent, resolution=resolution
    )
    # The grid already carries any size asked for; only "native" travels on.
    return tuple(
        reproject(
            raster,
            geobox,
            resampling=resampling,
            resolution=NATIVE if resolution == NATIVE else None,
        )
        for raster in rasters
    )


def _read_variables(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
) -> dict[str, xr.DataArray]:
    """Collect every variable by name, group-qualified inside a tree.

    Returns:
        {
            "<variable>" or "<group>/<variable>": variable,
        }
    """
    if isinstance(data, xr.DataArray):
        return {str(data.name): data}
    variables: dict[str, xr.DataArray] = {}
    if isinstance(data, xr.Dataset):
        for name, variable in data.data_vars.items():
            variables[str(name)] = variable
        return variables

    # A DataTree's indexing can return groups, so walk its rasters.
    for group, raster in data.gs.rasters.items():
        for name, variable in raster.data_vars.items():
            variables[f"{group}/{name}"] = variable
    return variables


def _read_kernels(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
    resampling: Resampling | Mapping[str, Resampling],
) -> dict[str, Resampling]:
    """Read the kernel every variable warps with, refusing a variable it cannot warp.

    Returns:
        {
            "<variable>" or "<group>/<variable>": kernel,
        }

    Raises:
        KeyError: A mapping names neither a variable nor `"*"`.
        ValueError: `data` holds no variable, a variable does not span the grid, or a
            kernel would blend class codes.
    """
    variables = _read_variables(data)
    if not variables:
        raise ValueError(f"{type(data).__name__} holds no data variables to warp")

    kernels: dict[str, Resampling] = {}
    blending_kernels: dict[str, Resampling] = {}
    for name, variable in variables.items():
        if not set(SPATIAL_DIMENSIONS) <= set(variable.dims):
            raise ValueError(
                f"{name!r} spans {list(variable.dims)}, not the grid "
                f"{list(SPATIAL_DIMENSIONS)}; select spatial variables before warping"
            )
        kernel = _read_kernel(resampling, name)
        kernels[name] = kernel
        # A class code averaged with its neighbours names no class.
        if variable.gs.is_categorical and kernel not in _VALUE_PRESERVING:
            blending_kernels[name] = kernel

    if blending_kernels:
        hint = ", ".join(f"{name!r}: 'mode'" for name in sorted(blending_kernels))
        raise ValueError(
            f"{sorted(blending_kernels)} hold class codes, which "
            f"{sorted(set(blending_kernels.values()))} would blend into values "
            f"naming no class; warp them with a value-preserving kernel, as "
            f"resampling={{'*': ..., {hint}}}, or drop the Legend first"
        )
    return kernels


def _read_kernel(
    resampling: Resampling | Mapping[str, Resampling], name: str
) -> Resampling:
    """Read the kernel one variable warps with, `"*"` answering for unnamed variables.

    Raises:
        KeyError: A mapping names neither `name` nor `"*"`.
    """
    if not isinstance(resampling, Mapping):
        return resampling
    if name in resampling:
        return resampling[name]
    if "*" in resampling:
        return resampling["*"]
    raise KeyError(
        f"{name!r} names no resampling kernel and the mapping carries "
        "no '*'; name it, or provide a default kernel"
    )


def _read_geobox(data: xr.DataArray | xr.Dataset) -> GeoBox:
    """Read the grid a raster's pixels sit on.

    Raises:
        ValueError: Nothing says where its pixels land.
    """
    geobox = data.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError(
            f"{type(data).__name__} sits on no locatable grid, so nothing says "
            f"where its pixels start from; open it georeferenced first"
        )
    return geobox


def _resolve_target(
    source: GeoBox, target: Target, resolution: SomeResolution | Native | None
) -> GeoBox:
    """Read the exact grid `source`'s pixels land on.

    A grid target is adopted whole; a bare CRS carries `source` into it.
    `"native"` nests `source`'s pixel size onto a grid target.

    Raises:
        CRSError: `target` names no CRS pyproj recognises.
        ValueError: `target` names no grid or CRS, `resolution` contradicts a
            target grid, or `"native"` stands in no whole-numbered ratio to it.
    """
    if not isinstance(target, GeoBox | xr.DataArray | xr.Dataset | xr.DataTree):
        return _reproject_grid(source, target, resolution)
    grid = _read_target_grid(target)
    _validate_resolution(grid, resolution)
    if resolution == NATIVE:
        return grid.zoom_out(_read_zoom_factor(source, grid))
    return grid


def _read_target_grid(
    target: GeoBox | xr.DataArray | xr.Dataset | xr.DataTree,
) -> GeoBox:
    """Read the shared grid an explicit target names.

    Raises:
        ValueError: The target sits on no locatable grid or carries no CRS.
    """
    if isinstance(target, GeoBox):
        grid = target
    else:
        grid = target.gs.geobox
    if isinstance(target, xr.DataTree) and grid is None:
        raise ValueError(
            "target tree has no shared grid; select a reference group explicitly"
        )
    if not isinstance(grid, GeoBox):
        raise ValueError(
            "the target raster sits on no locatable grid to land on; pass a "
            "GeoBox, a CRS, or a raster placed by a regular grid"
        )
    if grid.crs is None:
        raise ValueError(
            "the target grid carries no CRS; supply a georeferenced target"
        )
    return grid


def _validate_resolution(
    grid: GeoBox, resolution: SomeResolution | Native | None
) -> None:
    """Refuse a pixel size the target grid already fixes."""
    if resolution not in (None, NATIVE):
        raise ValueError(
            f"the target grid already fixes its pixel size at {grid.resolution}, "
            f"so resolution {resolution!r} contradicts it; drop resolution, ask for "
            f"{NATIVE!r} to keep the source's, or name a CRS instead of a grid"
        )


def _reproject_grid(
    source: GeoBox, crs_target: SomeCRS, resolution: SomeResolution | Native | None
) -> GeoBox:
    """Carry a grid into a CRS, at `resolution` or the source's ground sampling.

    Raises:
        ValueError: `crs_target` names no CRS.
    """
    crs = norm_crs(crs_target)
    if crs is None:
        raise ValueError(
            f"{crs_target!r} names no grid and no CRS to land in; pass a GeoBox, "
            f"a raster carrying one, or a CRS such as 'EPSG:3857'"
        )
    # A CRS fixes no pixel size, so keeping the source's is odc's "auto".
    pixel_size = "auto" if resolution in (None, NATIVE) else resolution
    return source.to_crs(
        crs, resolution=cast("Literal['auto', 'fit', 'same']", pixel_size)
    )


def _read_zoom_factor(source: GeoBox, target: GeoBox) -> int:
    """Count how many of a target's pixels span one of the source's.

    Args:
        source: Grid whose own pixel size is kept.
        target: Lattice it nests onto, which must be at least as fine.

    Returns:
        Whole number of target pixels across one source pixel, 1 where the two
        already agree.

    Raises:
        ValueError: The grids have different CRSs, `target` is coarser, or
            the sizes do not share one integer ratio on both axes.

    Examples:
        >>> _read_zoom_factor(sixty_metre, ten_metre)
        6
    """
    if source.crs != target.crs:
        raise ValueError(
            "native resolution requires a shared CRS; reproject onto one CRS "
            "before nesting grids"
        )
    along_x = abs(source.resolution.x) / abs(target.resolution.x)
    along_y = abs(source.resolution.y) / abs(target.resolution.y)
    ratio = round(along_x)
    if ratio < 1:
        raise ValueError(
            f"a pixel of {source.resolution} spans {along_x:.4g} of one of "
            f"{target.resolution}, so keeping it would refine that lattice; nest "
            f"onto a grid at least as fine, or take one resolution for all"
        )
    if not isclose(along_x, ratio, rel_tol=_RATIO_TOLERANCE) or not isclose(
        along_y, ratio, rel_tol=_RATIO_TOLERANCE
    ):
        raise ValueError(
            f"a pixel of {source.resolution} spans {along_x:.4g} by {along_y:.4g} "
            f"of {target.resolution}, which no whole number of pixels nests on; "
            f"take one resolution for all instead"
        )
    return ratio


def _reproject_tree(
    data: xr.DataTree,
    target: Target,
    kernels: Mapping[str, Resampling],
    resolution: SomeResolution | Native | None,
) -> xr.DataTree:
    """Warp every group of a tree onto one grid, or its nesting grids."""
    from geosave.geodata.core.raster.create import tree

    rasters = data.gs.rasters
    # A grid target is adopted whole; a CRS keeps the ground every group covers.
    if isinstance(target, GeoBox | xr.DataArray | xr.Dataset | xr.DataTree):
        grid = _read_target_grid(target)
        _validate_resolution(grid, resolution)
    else:
        grid = _resolve_alignment_grid(
            list(rasters.values()), target=target, extent="union", resolution=resolution
        )

    group_resolution = NATIVE if resolution == NATIVE else None
    groups = {}
    for name, raster in rasters.items():
        source = _read_geobox(raster)
        geobox = _resolve_target(source, grid, group_resolution)
        group_kernels: dict[str, Resampling] = {
            str(variable): kernels[f"{name}/{variable}"]
            for variable in raster.data_vars
        }
        groups[name] = _warp_raster(raster, source, geobox, group_kernels)

    result = tree(groups)
    result.attrs = dict(data.attrs)
    return result


def _warp_raster[R: xr.DataArray | xr.Dataset](
    raster: R, source: GeoBox, geobox: GeoBox, kernels: Mapping[str, Resampling]
) -> R:
    """Warp a variable or Dataset onto `geobox`, untouched where it already sits there."""
    if geobox == source:
        return raster
    if isinstance(raster, xr.DataArray):
        return cast("R", _warp_variable(raster, geobox, kernels[str(raster.name)]))

    # odc's Dataset path keeps the old spatial_ref attrs, so variables warp one by one.
    variables = {}
    for name, variable in raster.data_vars.items():
        variables[name] = _warp_variable(variable, geobox, kernels[str(name)])
    return cast("R", xr.Dataset(variables).gs.write_crs().assign_attrs(raster.attrs))


def _write_default_nodata(variable: xr.DataArray) -> xr.DataArray:
    """Give a variable carrying no fill value one, warning which.

    Returns:
        The variable, carrying NaN where float and its dtype's far end where
        integer: the maximum where unsigned, the minimum where signed.
    """
    if variable.gs.nodata is not None or variable.dtype.kind not in "iuf":
        return variable
    if variable.dtype.kind == "f":
        fill = float("nan")
    else:
        limits = np.iinfo(variable.dtype)
        fill = int(limits.max if variable.dtype.kind == "u" else limits.min)
    warnings.warn(
        f"{str(variable.name)!r} carries no fill value, so the pixels its warp "
        f"does not cover take {fill}; choose one with gs.write_nodata(...) "
        f"before warping",
        AssumedFillWarning,
        skip_file_prefixes=tuple(_package_paths),
    )
    return variable.gs.write_nodata(fill)


def _warp_variable(
    variable: xr.DataArray, geobox: GeoBox, kernel: Resampling
) -> xr.DataArray:
    """Warp one variable, keeping its name, dimension order, and fill value."""
    variable = _write_default_nodata(variable)
    fill = variable.gs.nodata

    # ODC needs the spatial axes last; the caller's order returns afterwards.
    yx_last = variable.transpose(..., *SPATIAL_DIMENSIONS)
    pixels = to_yx(yx_last.odc.reproject(geobox, resampling=kernel))
    pixels = pixels.transpose(*variable.dims).rename(variable.name)

    # odc drops a NaN fill value from attrs and narrows a float one to int.
    return pixels if fill is None else pixels.gs.write_nodata(fill)


def _resolve_alignment_grid(
    rasters: Sequence[xr.DataArray | xr.Dataset | xr.DataTree],
    *,
    target: Target,
    extent: Extent = "match",
    resolution: SomeResolution | Native | None = None,
) -> GeoBox:
    """Resolve an explicit target and optionally adjust its coverage."""
    grids = _read_grids(rasters)
    reference = _pick_reference_grid(grids, target, resolution)
    grid = _cover_extent(reference, grids, extent)
    if resolution == NATIVE:
        grid = _fit_zoom_factors(grid, grids, extent)
    if extent == "intersection" and not (grid.shape.x and grid.shape.y):
        raise ValueError(
            "the rasters cover no ground in common, so an intersection holds no "
            "pixel; align them on 'union' or load them over one footprint"
        )
    return grid


def _read_grids(
    rasters: Sequence[xr.DataArray | xr.Dataset | xr.DataTree],
) -> list[GeoBox]:
    """Read every raster's grid, every group's for a tree.

    Raises:
        ValueError: `rasters` is empty, or one holds no group on a locatable grid.
    """
    if not rasters:
        raise ValueError("no rasters to align; pass at least one")
    grids: list[GeoBox] = []
    for ordinal, data in enumerate(rasters):
        groups = data.gs.rasters.values() if isinstance(data, xr.DataTree) else (data,)
        if not groups:
            raise ValueError(f"raster {ordinal} holds no groups on a locatable grid")
        for raster in groups:
            geobox = raster.gs.geobox
            if not isinstance(geobox, GeoBox):
                raise ValueError(
                    f"raster {ordinal} sits on no locatable grid, so nothing says "
                    f"where its pixels land against the rest; open it georeferenced "
                    f"first"
                )
            grids.append(geobox)
    return grids


def _pick_finest_grid(grids: Sequence[GeoBox]) -> GeoBox:
    """Pick the grid with the smallest pixel area, or the first across CRSs.

    Degrees and metres name no common scale, so pixel areas rank only in one CRS.
    """
    if len({grid.crs for grid in grids}) > 1:
        return grids[0]
    return min(grids, key=lambda grid: abs(grid.transform.determinant))


def _pick_reference_grid(
    grids: Sequence[GeoBox],
    target: Target,
    resolution: SomeResolution | Native | None,
) -> GeoBox:
    """Use a target grid directly; derive a CRS-only target from source coverage."""
    size = None if resolution == NATIVE else resolution
    if isinstance(target, GeoBox | xr.DataArray | xr.Dataset | xr.DataTree):
        grid = _read_target_grid(target)
        _validate_resolution(grid, size)
        return grid

    return _resolve_target(_pick_finest_grid(grids), target, size)


def _cover_extent(reference: GeoBox, grids: Sequence[GeoBox], extent: Extent) -> GeoBox:
    """Lay the ground `extent` names on the reference's pixel lattice."""
    if extent == "match":
        return reference
    # enclosing re-lays a footprint on the reference's own pixel grid.
    footprints = []
    for grid in grids:
        footprints.append(
            reference.enclosing(grid.extent.to_crs(cast("CRS", reference.crs)))
        )
    if extent == "union":
        return geobox_union_conservative(footprints)
    return geobox_intersection_conservative(footprints)


def _fit_zoom_factors(grid: GeoBox, grids: Sequence[GeoBox], extent: Extent) -> GeoBox:
    """Shape a grid so every raster's pixel size divides it.

    An undivided shape makes zoom_out round up, off the shared ground, so it
    grows for "union" and "match" and is trimmed for "intersection".
    """
    step = lcm(*(_read_zoom_factor(source, grid) for source in grids))
    if extent == "intersection":
        height = grid.shape.y - grid.shape.y % step
        width = grid.shape.x - grid.shape.x % step
        return grid[:height, :width]
    return grid.pad_wh(step)
