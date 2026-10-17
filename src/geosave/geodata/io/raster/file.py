"""Saved raster files, and the scenes one write saves them as.

A `RasterFile` is one saved file or store and the header it states: grid,
variables, date and per-band metadata, without pixels. A `Scene` is the files
of one collection at one instant, keyed by asset; it is what a writer returns,
and observation publication turns it into one STAC Item. Stores may hold
more dates than the observation publisher supports.

A write asked not to compute returns its scenes pending, written by one
`dask.compute` of them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import dask
import dask.array as da
import dask.threaded
import xarray as xr

from ..storage import (
    Location,
    StorageOptions,
    absolute_location,
    is_local,
    local_path,
    location,
)

if TYPE_CHECKING:
    from dask.delayed import Delayed
    from upath import UPath

# An Asset's key within its Item, as "red" or "image".
type AssetKey = str

# Writes one raster to one location, returning the file it wrote.
type Writer = Callable[[xr.Dataset, UPath], RasterFile]


@dataclass(frozen=True)
class RasterFile:
    """One saved raster file or store and the header it states; one STAC Asset.

    Args:
        path: Where it is; a remote path carries its `storage_options`.
        header: The raster as `open_raster` opens it, its pixels replaced by
            empty lazy arrays, so holding it keeps no data alive.

    Examples:
        >>> file = RasterFile.open("out/scene.tif")
        >>> file.name, file.header.gs.variables
        ('scene.tif', ('red', 'nir'))
    """

    path: UPath
    header: xr.Dataset

    @classmethod
    def from_xarray(cls, path: Location, raster: xr.Dataset) -> RasterFile:
        """Describe the file at `path` by the raster it holds; no pixel is kept.

        Args:
            path: Location of the file.
            raster: The file as `open_raster` opens it.

        Returns:
            The file, named absolutely as a catalog stores it.
        """
        header = raster.map(
            lambda band: band.copy(
                data=da.empty(band.shape, dtype=band.dtype, chunks=-1)
            )
        )
        options = getattr(path, "storage_options", None) or None
        return cls(location(absolute_location(path), options), header)

    @classmethod
    def open(
        cls, path: Location, storage_options: StorageOptions | None = None
    ) -> RasterFile:
        """Read the header of a saved file; no pixel is read.

        Args:
            path: Location of a raster file or store.
            storage_options: Options for the filesystem a URL names.

        Returns:
            The file with the header it states.

        Raises:
            ValueError: The suffix names no raster format.
        """
        # Deferred: openers imports the raster formats this module sits among.
        from ..openers import open_raster

        found = location(path, storage_options)
        with open_raster(found, storage_options=storage_options) as raster:
            return cls.from_xarray(found, raster)

    @property
    def name(self) -> str:
        """Return the file's name."""
        return self.path.name

    def __fspath__(self) -> str:
        """Return the local path, as the file's own path does.

        Raises:
            TypeError: The file is remote, whose path is not `os.PathLike`.
        """
        if not is_local(self.path):
            raise TypeError(f"{self.path} is remote and has no local path")
        return str(local_path(self.path))

    def __str__(self) -> str:
        """Return the path as text, as a path prints."""
        return str(self.path)

    def __repr__(self) -> str:
        """Describe the file by its path, variables and date."""
        times = self.header.gs.times
        when = "undated" if times is None else f"{times[0]:%Y-%m-%d}"
        return f"RasterFile({self.path}, {list(self.header.gs.variables)}, {when})"


@dataclass(frozen=True)
class Scene:
    """The files of one collection, keyed by asset, with observation coordinates.

    Args:
        collection: Tree layer the files were written from, or None for a lone
            raster.
        files: Each asset's key mapped to its file, as
            `{"red": RasterFile(...), "nir": RasterFile(...)}`. None while
            pending.
        coords: Nonspatial observation coordinates captured before file splitting.
            They need not be representable by the file format.
        pending: The write producing the files, resolving to this scene
            written, or None once written. `dask.compute` computes a pending
            scene as it does any dask collection.

    Examples:
        >>> ds.gs.to_cog("out/s_{time:%Y%m%d}_{band}.tif")[0]
        Scene(None, {'red': s_20250601_red.tif, 'nir': s_20250601_nir.tif})
        >>> later = ds.gs.to_cog("out/{time:%Y%m%d}.tif", compute=False)
        >>> written = dask.compute(later)[0]
    """

    collection: str | None
    files: Mapping[AssetKey, RasterFile] | None
    pending: Delayed | None = None
    coords: xr.Coordinates | None = field(default=None, compare=False)

    @classmethod
    def open(
        cls,
        paths: Mapping[AssetKey, Location],
        *,
        collection: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> Scene:
        """Describe files saved elsewhere as one scene, reading their headers only.

        Args:
            paths: Each asset's key mapped to its file.
            collection: Collection the files belong to.
            storage_options: Options for the filesystem a URL names.

        Returns:
            The scene, its files holding their headers.

        Examples:
            >>> Scene.open({"optical": "s0/optical.tif", "label": "s0/label.tif"})
            Scene(None, {'optical': optical.tif, 'label': label.tif})
        """
        return cls(
            collection,
            {
                key: RasterFile.open(path, storage_options)
                for key, path in paths.items()
            },
        )

    @classmethod
    def defer(
        cls,
        collection: str | None,
        planned: Mapping[AssetKey, tuple[xr.Dataset, UPath]],
        write: Writer,
        coords: xr.Coordinates | None = None,
    ) -> Scene:
        """Return the scene `write` will make of each raster and path, once computed.

        Args:
            collection: Collection the files belong to.
            planned: Each asset's key mapped to its raster and where it goes.
            write: Writer called with each raster and path on compute.
            coords: Nonspatial coordinates carried into the written scene.

        Returns:
            Pending scene.
        """
        return cls(
            collection,
            None,
            dask.delayed(_write, pure=False)(collection, planned, write, coords),
            coords,
        )

    def compute(self) -> Scene:
        """Write a pending scene, or return a written one as it is.

        Returns:
            The written scene.
        """
        return self if self.pending is None else self.pending.compute()

    @property
    def written(self) -> Mapping[AssetKey, RasterFile]:
        """Return the files of a written scene.

        Raises:
            RuntimeError: The scene is still pending.
        """
        if self.files is None:
            raise RuntimeError("this scene is not written yet; call .compute()")
        return self.files

    # A pending scene is a dask collection by handing the protocol to its write,
    # which resolves to the written scene.
    def __dask_graph__(self) -> Any:
        return None if self.pending is None else self.pending.__dask_graph__()

    def __dask_keys__(self) -> Any:
        return self._delayed.__dask_keys__()

    def __dask_postcompute__(self) -> Any:
        return self._delayed.__dask_postcompute__()

    def __dask_postpersist__(self) -> Any:
        return self._delayed.__dask_postpersist__()

    def __dask_tokenize__(self) -> Any:
        if self.pending is None:
            return (
                type(self).__name__,
                self.collection,
                tuple(map(str, self.written.values())),
            )
        return self.pending.key

    def __dask_layers__(self) -> Any:
        return self._delayed.__dask_layers__()

    __dask_scheduler__ = staticmethod(dask.threaded.get)

    @property
    def _delayed(self) -> Delayed:
        if self.pending is None:
            raise TypeError("this scene is written, not pending")
        return self.pending

    def __repr__(self) -> str:
        """Describe the scene by its collection and the file of each asset."""
        if self.files is None:
            return f"Scene({self.collection!r}, pending)"
        files = ", ".join(f"{key!r}: {file.name}" for key, file in self.files.items())
        return f"Scene({self.collection!r}, {{{files}}})"


def _write(
    collection: str | None,
    planned: Mapping[AssetKey, tuple[xr.Dataset, UPath]],
    write: Writer,
    coords: xr.Coordinates | None = None,
) -> Scene:
    """Write each planned raster, returning the scene they make."""
    return Scene(
        collection,
        {key: write(raster, path) for key, (raster, path) in planned.items()},
        coords=coords,
    )
