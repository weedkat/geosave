"""How a raster is arranged as files, grouped into Items, and written.

The path a raster is written to is a `str.format` template, and the fields it
names decide the files:

    "out/scene.tif"                  one file, every band
    "out/s2_{band}.tif"              one file per band
    "out/{time:%Y%m%d}.tif"          one file per date
    "out/{time:%Y%m%d}/{band}.tif"   one file per date and band
    "out/{collection}.zarr"          one store per tree layer

Three fields are built in, the ones that can split a raster: `time`, each
file's instant, a datetime taking `strftime` directives; `band`, each file's
variable; and `collection`, a tree's layer. Anything else belongs in the
path before it is given, as `f"out/{name}_{{band}}.tif"`.

The suffix names the format. A GeoTIFF holds one `(band, y, x)` array, one
instant of one grid, so its dates must be split; a Zarr or netCDF store holds
any dates it is given.

A raster, or a tree of them, is split only along the fields its path names, and
each piece is written to the path it renders. A tree's layers are `collection`.
The files of one layer and date are one `Scene`. Each file is an asset keyed by
the band it holds, under `{band}` or because it holds one; a file holding
several bands is `"image"`. A catalog reader names a one-band asset by its key,
so the key keeps the band's name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
import xarray as xr
from upath import UPath

from geosave.geodata.conventions import TIME_COORDINATE
from geosave.geodata.utils.template import fields_named

from ..storage import Location, location
from .file import AssetKey, Scene, Writer

# The asset key of a file holding several bands.
SEVERAL_BANDS = "image"

# Files holding one instant of one grid.
_TIFF_SUFFIXES = (".tif", ".tiff")

# Stores holding any dates of one grid.
_STORE_SUFFIXES = (".zarr", ".nc", ".nc4", ".cdf")


class Planned(NamedTuple):
    """One file a raster is about to be written as: a writer's arguments.

    Args:
        raster: Pixels the file holds, still lazy.
        path: Where the file goes; a remote path carries its `storage_options`.
    """

    raster: xr.Dataset
    path: UPath


class PlannedScene(NamedTuple):
    """One scene a raster is about to be written as.

    Args:
        collection: Tree layer the files hold, or None for a lone raster.
        files: Each asset's key mapped to its planned file.
    """

    collection: str | None
    files: dict[AssetKey, Planned]
    coords: xr.Coordinates | None = None


def layout(
    source: xr.Dataset | xr.DataTree,
    template: Location | Mapping[str, Location],
) -> list[PlannedScene]:
    """Say which scenes, and which files, a raster or a tree of rasters is written as.

    The template is read first: `{collection}` splits a tree into its layers,
    `{time}` a raster into its dates and `{band}` into its variables. Each
    piece is rendered to its path, so a field a template leaves out keeps that
    axis together in one file. Layers never share a file, nor do the dates of
    a GeoTIFF. No pixel is read.

    Args:
        source: Raster, or a tree whose layers are rasters.
        template: Path or UPath ending in `.tif`, `.tiff`, `.zarr` or `.nc`,
            naming any of the fields in this module's description. A tree
            also takes layer names mapped to their own templates. A UPath's
            `storage_options` travel to every path it renders.

    Returns:
        One scene per layer and date, layers then time in order.

    Raises:
        ValueError: The suffix names no raster format; a template maps no path
            to a layer; layers, or the dates of a GeoTIFF, would share a file;
            `{time}` is named on a raster without dates; a GeoTIFF's raster
            carries no locatable grid or spans a non-spatial axis other than
            time; or its instants are empty, repeated or NaT.
        KeyError: The template names a field nothing fills.

    Examples:
        >>> [{key: file.path.name for key, file in scene.files.items()} for scene in layout(ds, "out/{time:%Y%m%d}/{band}.tif")]
        [{'red': 'red.tif', 'nir': 'nir.tif'}, ...]
    """
    if isinstance(source, xr.Dataset):
        layers: dict[str | None, xr.Dataset] = {None: source}
    else:
        layers = {
            name: child.to_dataset(inherit="all_coords")
            for name, child in source.children.items()
        }
    if isinstance(template, Mapping):
        absent = [name for name in layers if name not in template]
        if absent:
            raise ValueError(f"the templates name no path for groups {absent}")
        paths = {name: template[name] for name in layers}
    else:
        paths = dict.fromkeys(layers, template)

    scenes: dict[tuple[str | None, pd.Timestamp | None], PlannedScene] = {}
    owners: dict[str, tuple[str | None, pd.Timestamp | None]] = {}
    for layer, ds in layers.items():
        # Rendered as text, so a UPath's options are carried beside it.
        options = getattr(paths[layer], "storage_options", None) or None
        path = str(paths[layer])
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in (*_TIFF_SUFFIXES, *_STORE_SUFFIXES):
            raise ValueError(
                f"{path} names no raster format; end it in .tif, .zarr or .nc"
            )
        named = fields_named(path)
        tiff = suffix in _TIFF_SUFFIXES
        if tiff:
            _require_one_grid(ds)
        # A store keeps its dates together unless the path splits them.
        instants = _scenes(ds) if tiff or TIME_COORDINATE in named else _dated(ds)
        if instants[0][0] is None and TIME_COORDINATE in named:
            what = f"group {layer!r}" if layer else "this raster"
            hint = (
                f'give a template per group, as to_cog({{{layer!r}: "out/{layer}.tif", ...}})'
                if layer
                else "drop it from the path"
            )
            raise ValueError(
                f"{what} has no dates, so {{time}} cannot be filled; {hint}"
            )
        if tiff and len(instants) > 1 and TIME_COORDINATE not in named:
            raise ValueError(
                f"{len(instants)} dates would share {path}; add {{time}} to the "
                f"path, as {{time:%Y%m%d}}, or save with to_zarr or to_netcdf, "
                f"which hold dates in one store"
            )
        for instant, scene in instants:
            planned = scenes.setdefault(
                (layer, instant), PlannedScene(layer, {}, observation_coords(scene))
            ).files
            for band, part in _bands(scene, split="band" in named):
                fields = {"band": band, "time": instant, "collection": layer}
                file = path.format(**{k: v for k, v in fields.items() if v is not None})
                if file in owners:
                    raise _clash(file, owners[file], (layer, instant), named)
                owners[file] = (layer, instant)
                planned[_asset_key(part)] = Planned(part, location(file, options))
    return list(scenes.values())


def save(
    plan: Iterable[PlannedScene], write: Writer, *, compute: bool = True
) -> list[Scene]:
    """Write each scene a `layout` planned.

    Args:
        plan: What `layout` returned.
        write: Writer each planned file is handed to, its options bound.
        compute: False returns every scene pending, written by one
            `dask.compute` of the result.

    Returns:
        The scenes in the order planned.

    Raises:
        FileExistsError: A file exists and the writer may not replace it.

    Examples:
        >>> save(layout(ds, "out/{time:%Y%m%d}.tif"), write_cog)
        [Scene(None, {'image': 20250601.tif}), Scene(None, {'image': 20250611.tif})]
        >>> (scenes,) = dask.compute(save(layout(ds, "out/{band}.zarr"), zarr.write, compute=False))
    """
    if not compute:
        return [
            Scene.defer(scene.collection, scene.files, write, scene.coords)
            for scene in plan
        ]
    return [
        Scene(
            scene.collection,
            {key: write(raster, path) for key, (raster, path) in scene.files.items()},
            coords=scene.coords,
        )
        for scene in plan
    ]


def _asset_key(raster: xr.Dataset) -> AssetKey:
    """Key a file by the band it holds, or `SEVERAL_BANDS` where it holds several."""
    names = list(raster.data_vars)
    return str(names[0]) if len(names) == 1 else SEVERAL_BANDS


def _require_one_grid(ds: xr.Dataset) -> None:
    """Refuse a raster that cannot be one georeferenced `(band, y, x)` array per date."""
    spatial = ds.odc.spatial_dims
    if spatial is None:
        raise ValueError(
            "the raster carries no locatable grid, so its files cannot be "
            "georeferenced; assign a CRS with odc.geo.xr.assign_crs first"
        )
    # Only pixels have to fit a file; an axis only a coordinate spans writes nothing.
    pixel_dims = {str(dim) for array in ds.data_vars.values() for dim in array.dims}
    extra_dims = sorted(pixel_dims - {*spatial, TIME_COORDINATE})
    if extra_dims:
        raise ValueError(
            f"a COG holds one instant of one grid, but the raster also spans "
            f"{extra_dims}; select those axes away before writing"
        )


def _scenes(ds: xr.Dataset) -> list[tuple[pd.Timestamp | None, xr.Dataset]]:
    """Split a raster into its dates; an undated raster is one scene with no instant."""
    # A scalar time coordinate dates the raster as one step of a time axis does.
    if TIME_COORDINATE not in ds.coords or ds[TIME_COORDINATE].ndim > 1:
        return [(None, ds)]
    times = pd.DatetimeIndex(np.atleast_1d(ds[TIME_COORDINATE].values))
    if times.hasnans or not times.is_unique or len(times) == 0:
        raise ValueError("scene times must be nonempty, unique and not NaT")
    # Free of NaT, as just checked.
    instants = cast("list[pd.Timestamp]", list(times))
    if TIME_COORDINATE not in ds.dims:
        return [(instants[0], ds)]
    return [
        (instant, ds.isel({TIME_COORDINATE: index}))
        for index, instant in enumerate(instants)
    ]


def _dated(ds: xr.Dataset) -> list[tuple[pd.Timestamp | None, xr.Dataset]]:
    """Keep a store's raster whole, dated by its instant where it holds one."""
    scenes = _scenes(ds)
    return [(scenes[0][0] if len(scenes) == 1 else None, ds)]


def _bands(scene: xr.Dataset, *, split: bool) -> list[tuple[str | None, xr.Dataset]]:
    """Split a scene into its variables, or keep them together as one piece."""
    if not split:
        return [(None, scene)]
    return [(str(band), scene[[band]]) for band in scene.data_vars]


def _clash(
    path: str,
    first: tuple[str | None, pd.Timestamp | None],
    second: tuple[str | None, pd.Timestamp | None],
    named: Mapping[str, str | None],
) -> ValueError:
    """Say how to keep two pieces from sharing one file."""
    if first[0] != second[0]:
        return ValueError(
            f"2 layers would share {path}; add {{collection}} to the path, as "
            f"out/{{collection}}.tif"
        )
    return ValueError(
        f"{first[1]:%Y-%m-%d} and {second[1]:%Y-%m-%d} would share {path}; make "
        f"{{time:{named.get(TIME_COORDINATE)}}} more precise, as {{time:%Y%m%d}}"
    )


def observation_coords(raster: xr.Dataset) -> xr.Coordinates:
    """Keep nonspatial coordinates independently of a file format's metadata support."""
    spatial = set(raster.odc.spatial_dims or ())
    return xr.Coordinates(
        {
            name: coord.variable.copy(deep=False)
            for name, coord in raster.coords.items()
            if not spatial.intersection(coord.dims)
            and "grid_mapping_name" not in coord.attrs
        }
    )
