"""Raster-leaf structure and compatibility rules for native DataTrees."""

from __future__ import annotations

import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata.errors import RasterValidationError
from . import Rule
from .common import georeferenced
from .dataarray import same_encoding_as, same_pixel_meaning_as


def raster_leaves(tree: xr.DataTree) -> None:
    """Require georeferenced raster leaves that own their spatial coordinates.

    This is the default GeoDataTree validation policy. It imposes no shared grid
    or timeline; those relationships belong to additional operation rules.

    Args:
        tree: Native subtree to validate without reading pixels.

    Raises:
        RasterValidationError: There are no rasters, a raster has children,
            its grid coordinates are inherited, or its local grid is invalid.
    """
    rasters = [node for node in tree.subtree if node.data_vars]
    if not rasters:
        raise RasterValidationError(
            "There are no raster variables to validate. Select or load raster data first."
        )

    for node in rasters:
        if node.children:
            raise RasterValidationError(
                "Raster variables belong on leaves; move this node's data into a child.",
                path=node.relative_to(tree),
            )

        try:
            local_coords = set(node.to_dataset(inherit=False).coords)
            effective = node.to_dataset(inherit="all_coords")
            grid_coords = set(effective.gs.spatial_dims or ())
            if effective.odc.crs_coord is not None:
                grid_coords.add(effective.odc.crs_coord.name)
            inherited = grid_coords.intersection(effective.coords) - local_coords
            if inherited:
                raise RasterValidationError(
                    f"Grid coordinates must belong to the raster leaf: {sorted(inherited, key=str)}"
                )
            # No grid coordinate is inherited, so the leaf's own grid is the
            # effective one; ODC locates it once per leaf.
            georeferenced(effective)
        except RasterValidationError as error:
            error.path = node.relative_to(tree)
            raise


def same_grid(tree: xr.DataTree) -> None:
    """Require one grid across the data nodes: CRS, resolution and pixel alignment.

    Extents may differ: nodes on one grid stack by their pixel labels, the
    union of their extents, when datasets are assembled.

        a: x 500000–500040    b: x 500020–500080    10 m, alignment (0, 0)   → one grid
        c: x 500005–500045                          10 m, alignment (5, 0)   → another

    Raises:
        RasterValidationError: Nodes sit on several grids; the error names each.
    """
    nodes_by_grid: dict[tuple | None, list[str]] = {}
    for node in tree.subtree:
        if not node.data_vars:
            continue
        path = node.relative_to(tree)
        geobox = node.to_dataset(inherit=False).gs.geobox
        grid = None
        if geobox is not None:
            grid = (geobox.crs, geobox.resolution, geobox.alignment)
        nodes_by_grid.setdefault(grid, []).append(path)
    if len(nodes_by_grid) <= 1:
        return

    lines = []
    for paths in nodes_by_grid.values():
        lines.append(", ".join(paths))
    raise RasterValidationError(
        f"Nodes are on {len(nodes_by_grid)} grids ({'; '.join(lines)}). Resample "
        "them onto one with tree.gs.reproject(geobox), or select one grid's nodes.",
        path=list(nodes_by_grid.values())[1][0],
    )


def same_geobox(tree: xr.DataTree) -> None:
    """Require one GeoBox across the data nodes: the same grid and the same extent.

    Raises:
        RasterValidationError: Nodes sit on several GeoBoxes; the error names each.
    """
    nodes_by_geobox: dict[GeoBox | None, list[str]] = {}
    for node in tree.subtree:
        if not node.data_vars:
            continue
        geobox = node.to_dataset(inherit=False).gs.geobox
        nodes_by_geobox.setdefault(geobox, []).append(node.relative_to(tree))
    if len(nodes_by_geobox) <= 1:
        return

    lines = []
    for paths in nodes_by_geobox.values():
        lines.append(", ".join(paths))
    raise RasterValidationError(
        f"Nodes are on {len(nodes_by_geobox)} GeoBoxes ({'; '.join(lines)}). Align "
        "them with tree.gs.reproject(geobox).",
        path=list(nodes_by_geobox.values())[1][0],
    )


def same_chunks(tree: xr.DataTree) -> None:
    """Require one Dask chunking of the grid across the data nodes.

    Raises:
        RasterValidationError: Nodes are chunked differently along their grid;
            the error names each.
    """
    nodes_by_chunks: dict[tuple | None, list[str]] = {}
    for node in tree.subtree:
        if not node.data_vars:
            continue
        raster = node.to_dataset(inherit=False)
        chunks = (raster.gs.chunk_heights, raster.gs.chunk_widths)
        nodes_by_chunks.setdefault(chunks, []).append(node.relative_to(tree))
    if len(nodes_by_chunks) <= 1:
        return

    lines = []
    for paths in nodes_by_chunks.values():
        lines.append(", ".join(paths))
    raise RasterValidationError(
        f"Nodes are chunked {len(nodes_by_chunks)} ways ({'; '.join(lines)}). "
        "Rechunk them alike with tree.chunk(y=..., x=...).",
        path=list(nodes_by_chunks.values())[1][0],
    )


def one_raster(tree: xr.DataTree) -> None:
    """Require the selected node itself to be one raster leaf.

    Raises:
        RasterValidationError: The node is a group or holds no raster variables.
    """
    if tree.children or not tree.data_vars:
        raise RasterValidationError("Select one raster leaf first")


def splittable(dim: str) -> Rule[xr.DataTree]:
    """Require one raster leaf that can be split by position along `dim`.

    The dimension must be non-spatial, labelled by a one-dimensional
    coordinate and used by at least one variable. Variables without it go to
    a sibling named `static`, so a dimension of that name cannot sit beside them.

    Args:
        dim: Dimension the split slices by position.

    Returns:
        A metadata-only rule for `tree.gs.validate`.

    Examples:
        >>> leaf.gs.validate(splittable("time"))
    """

    def check(tree: xr.DataTree) -> None:
        try:
            one_raster(tree)
            raster = tree.to_dataset(inherit="all_coords")

            if dim in (raster.gs.spatial_dims or ()):
                raise RasterValidationError(
                    "Splitting a spatial dimension would remove the 2-D grid"
                )
            if dim not in raster.coords or raster[dim].dims != (dim,):
                raise RasterValidationError(
                    f"Dimension {dim!r} needs a one-dimensional coordinate to "
                    "preserve its labels"
                )

            varying = []
            for key, variable in raster.data_vars.items():
                if dim in variable.dims:
                    varying.append(key)
            if not varying:
                raise RasterValidationError(
                    f"No raster variable contains dimension {dim!r}"
                )
            if dim == "static" and len(varying) < len(raster.data_vars):
                raise RasterValidationError(
                    "Rename dimension 'static' before splitting it beside static variables"
                )
        except RasterValidationError as error:
            error.operation = "split"
            raise

    return check


def compatible_merge(tree: xr.DataTree) -> None:
    """Require immediate raster children on one GeoBox with distinct variables.

    Raises:
        RasterValidationError: A child is not a raster leaf, the grids differ,
            or a variable name occurs in several children.

    Examples:
        >>> tree["s2"].gs.validate(compatible_merge)
    """
    try:
        _check_raster_children(tree)
        same_geobox(tree)

        names = set()
        for child in tree.children.values():
            repeated = names.intersection(child.data_vars)
            if repeated:
                raise RasterValidationError(
                    f"Duplicate variables in merge: {sorted(repeated, key=str)}",
                    path=child.relative_to(tree),
                )
            names.update(child.data_vars)
    except RasterValidationError as error:
        error.operation = "merge"
        raise


def _check_raster_children(tree: xr.DataTree) -> None:
    """Require at least one child, each of them a raster leaf."""
    if not tree.children:
        raise RasterValidationError("Combining needs immediate raster-leaf children")
    for child in tree.children.values():
        if child.children or not child.data_vars:
            raise RasterValidationError(
                "Combining needs immediate raster-leaf children; select their group",
                path=child.relative_to(tree),
            )


def _check_concat_variable(first: xr.DataArray, other: xr.DataArray, dim: str) -> None:
    """Reject representation changes before native concatenation can promote data."""
    first_dims = tuple(axis for axis in first.dims if axis != dim)
    other_dims = tuple(axis for axis in other.dims if axis != dim)
    if first_dims != other_dims or first.dtype != other.dtype:
        raise RasterValidationError(
            "Different dimensions or dtype", variable=other.name
        )

    same_pixel_meaning_as(first)(other)
    same_encoding_as(first)(other)


def _check_concat_dataset(first: xr.Dataset, other: xr.Dataset, dim: str) -> None:
    """Check corresponding variables and indexes without constructing a stack."""
    if dim in (other.gs.spatial_dims or ()):
        raise RasterValidationError(
            "concat assembles non-spatial dimensions; prepare grids explicitly"
        )
    if dim not in other.coords or other[dim].dims not in ((), (dim,)):
        raise RasterValidationError(
            f"Each child needs a scalar or axis coordinate {dim!r}"
        )
    if set(other.data_vars) != set(first.data_vars):
        raise RasterValidationError("concat requires matching variable sets")

    for name in first.data_vars:
        _check_concat_variable(first[name], other[name], dim)

    try:
        xr.align(first, other, join="exact", copy=False, exclude={dim})
    except ValueError as error:
        raise RasterValidationError(
            f"Indexes outside {dim!r} must align exactly: {error}"
        ) from error


def compatible_concat(dim: str) -> Rule[xr.DataTree]:
    """Require immediate raster children compatible along a non-spatial dimension.

    Checks grids, coordinate presence, variable sets, dimensions, dtypes, units,
    flags and CF storage recipes. Other dimension indexes must agree exactly.
    Repeated concat labels and different units between distinct variables remain
    valid. Descriptive attrs are handled separately by native `combine_attrs`.

    Args:
        dim: Coordinate used by concatenation; node names do not supply it.

    Returns:
        A metadata-only rule for `tree.gs.validate`. It does not modify inputs
        or execute pixel tasks. Coordinate indexes may be compared.

    Examples:
        >>> observations.gs.validate(compatible_concat("time"))
    """

    def check(tree: xr.DataTree) -> None:
        try:
            _check_raster_children(tree)
            same_geobox(tree)
            first = next(iter(tree.children.values())).to_dataset(inherit="all_coords")
            for child in tree.children.values():
                try:
                    _check_concat_dataset(
                        first, child.to_dataset(inherit="all_coords"), dim
                    )
                except RasterValidationError as error:
                    error.path = child.relative_to(tree)
                    raise
        except RasterValidationError as error:
            error.operation = "concat"
            raise

    return check
