"""Lay rasters out as files: which selection each path format destination holds."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
import xarray as xr
from upath import UPath

from geosave.geodata.utils.template import fields_named

# Path format fields in the order a layout splits their axes.
AXES = ("collection", "res", "time", "band")

# Dates down to the second keep two scenes of one day apart.
TIME_FORMAT = "%Y%m%dT%H%M%S"


def default_path_format(
    member: xr.Dataset | xr.DataTree, collection: str | None
) -> str:
    """Name the layout used when none is given: collection, date, then one file per band.

    A raster of no collection has no `{collection}/` level, and one without
    dates no `{time}/` level. One file per band is one STAC asset per band,
    and band names are unique within a collection, so `{res}` is never needed.

    Args:
        member: One collection's raster, or its tree of grid nodes.
        collection: The collection's name, or None for a raster of none.

    Returns:
        Relative path format ending in `.tif`.

    Examples:
        >>> default_path_format(scene, "s2")
        's2/{time:%Y%m%dT%H%M%S}/{band}.tif'
    """
    levels = []
    if collection is not None:
        levels.append("{collection}")
    for _, raster in split_by_res(member):
        if "time" in raster.coords:
            levels.append("{time:" + TIME_FORMAT + "}")
            break
    levels.append("{band}.tif")
    return "/".join(levels)


def expand_paths(
    raster: xr.Dataset | xr.DataTree,
    root: UPath,
    path_format: str | Mapping[str, str] | None = None,
    *,
    collection: str | None = None,
) -> dict[UPath, xr.Dataset]:
    """Lay a raster out as files under a folder: each destination and the selection it holds.

    One splitter per path format field: `{collection}` a tree's collections,
    `{res}` a collection's grid nodes, `{time}` dates, `{band}` variables.
    An axis without a field stays whole in one file, so selections that
    differ only on it collide, and the error names the field to add.

    Args:
        raster: A tree of collections, one collection's tree of grid nodes,
            or a Dataset. Pixels remain lazy when already lazy.
        root: Folder the files go under, local or remote.
        path_format: Relative path filled with `str.format`, with optional
            `{collection}`, `{res}`, `{time}` and `{band}` fields, or one per
            collection. None takes `default_path_format`.
        collection: Name of a raster that is one collection, filling
            `{collection}`.

    Returns:
        Destinations mapped to Datasets, in collection, grid, time then
        variable order. All coordinates are retained.

    Raises:
        KeyError: A field has no value, such as `{collection}` for a raster
            of no collection.
        ValueError: `root` names a GeoTIFF rather than a folder, a path format
            is not relative, a mapping misses a collection, `{time}` meets a
            raster without dates, or selections collide at one filename.

    Examples:
        >>> files = expand_paths(tree, root, "{collection}/{res}/{time:%Y%m%d}/{band}.tif")
        >>> [path.relative_to(root).as_posix() for path in files]
        ['s2/r10m/20250601/red.tif', ...]
    """
    if root.suffix.lower() in (".tif", ".tiff"):
        raise ValueError(
            f"{root} names a file, but path is the folder files go under; "
            f"write path={str(root.parent)!r} with path_format={root.name!r}"
        )
    collections = split_by_collection(raster, collection)

    # 1. Every selection, with its value on each axis it was split along
    files: dict[UPath, xr.Dataset] = {}
    values_at: dict[UPath, dict[str, object]] = {}
    for name, member in collections:
        pattern = (
            default_path_format(member, name)
            if path_format is None
            else select_path_format(path_format, name)
        )
        _relative_name(pattern)
        fields = fields_named(pattern)
        parts: list[tuple[dict[str, object], xr.Dataset]] = []
        for res, grid_raster in split_by_res(member):
            parts.append(({"collection": name, "res": res}, grid_raster))
        if "time" in fields:
            parts = _split_parts(parts, "time", partial(split_by_time, collection=name))
        if "band" in fields:
            parts = _split_parts(parts, "band", split_by_band)

        # 2. One path per selection; two at one path name the axis they differ on
        for values, part in parts:
            known = {}
            for key, value in values.items():
                if value is not None:
                    known[key] = value
            destination = root / _relative_name(pattern.format(**known))
            if destination in files:
                differing = []
                for key in AXES:
                    if values_at[destination].get(key) != values.get(key):
                        differing.append(f"{{{key}}}")
                raise ValueError(
                    f"path selections collide at {destination}: they differ in "
                    f"{', '.join(differing)}, which the path format must tell apart"
                )
            files[destination] = part
            values_at[destination] = values
    return files


def select_path_format(
    path_format: str | Mapping[str, str], collection: str | None
) -> str:
    """Return one collection's path format: the shared one, or its own from a mapping.

    Raises:
        ValueError: The mapping names no path format for the collection.
    """
    if isinstance(path_format, str):
        return path_format
    if collection is None or collection not in path_format:
        raise ValueError(
            f"the path format mapping names none for collection {collection!r}; "
            f"it names {list(path_format)}"
        )
    return path_format[collection]


def split_by_collection(
    raster: xr.Dataset | xr.DataTree, collection: str | None = None
) -> list[tuple[str | None, xr.Dataset | xr.DataTree]]:
    """Split a tree into its collections; any other raster is one, named `collection`."""
    if isinstance(raster, xr.DataTree):
        members: list[tuple[str | None, xr.Dataset | xr.DataTree]] = []
        for name, child in raster.children.items():
            if child.children:
                members.append((name, child))
        if members:
            return members
    return [(collection, raster)]


def split_by_res(raster: xr.Dataset | xr.DataTree) -> list[tuple[str, xr.Dataset]]:
    """Split a collection into its grid nodes, each named as EOPF names it (`r10m`).

    Each grid node takes the collection's attrs under its own, so a file and
    its Item carry the collection's STAC metadata.
    """
    from geosave.geodata.core.raster.grid import name_grid

    if isinstance(raster, xr.DataTree):
        grids = []
        for path, dataset in raster.gs.rasters.items():
            grids.append(
                (
                    path.rsplit("/", 1)[-1],
                    dataset.assign_attrs({**raster.attrs, **dataset.attrs}),
                )
            )
        return grids
    geobox = raster.gs.geobox
    name = name_grid(geobox) if geobox is not None and geobox.crs is not None else ""
    return [(name, raster)]


def split_by_time(
    raster: xr.Dataset, collection: str | None = None
) -> list[tuple[pd.Timestamp, xr.Dataset]]:
    """Split a raster into its dates, one selection each."""
    if "time" not in raster.coords:
        what = f"collection {collection!r}" if collection is not None else "this raster"
        raise ValueError(f"{what} has no dates, so {{time}} cannot be filled")
    time = raster.coords["time"]
    if time.ndim > 1:
        raise ValueError("a time path field needs scalar or one-dimensional dates")
    dates = pd.DatetimeIndex(np.atleast_1d(time.values))
    if dates.empty or dates.hasnans or not dates.is_unique:
        raise ValueError("path times must be nonempty, unique and not NaT")
    instants = []
    for index, instant in enumerate(dates):
        selected = raster.isel({time.dims[0]: index}) if time.dims else raster
        instants.append((instant, selected))
    return instants


def split_by_band(raster: xr.Dataset) -> list[tuple[str, xr.Dataset]]:
    """Split a raster into its variables, one selection each."""
    bands = []
    for band in raster.data_vars:
        # Selecting with ds[[band]] discards unrelated observation coordinates.
        others = []
        for name in raster.data_vars:
            if name != band:
                others.append(name)
        bands.append((str(band), raster.drop_vars(others)))
    return bands


def _split_parts(parts, field, split):
    """Split every part along one axis, recording each piece's value for `field`."""
    split_parts = []
    for values, part in parts:
        for value, piece in split(part):
            split_parts.append(({**values, field: value}, piece))
    return split_parts


def _relative_name(name: str) -> str:
    """Keep path format output below its folder on local and remote filesystems."""
    windows = PureWindowsPath(name)
    if (
        not name
        or name.startswith(("/", "\\"))
        or windows.drive
        or "://" in name
        or ".." in windows.parts
        or "\\" in name
        or PurePosixPath(name) == PurePosixPath(".")
    ):
        raise ValueError("a path format must name a relative file below its folder")
    return PurePosixPath(name).as_posix()
