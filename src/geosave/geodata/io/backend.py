"""Let xarray open GeoSave rasters and trees by the engine name `geosave`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import xarray as xr
from xarray.backends import BackendEntrypoint

from .openers import RASTER_TREE_OPENERS, find_opener, open_raster, open_raster_tree


class GeoSaveBackendEntrypoint(BackendEntrypoint):
    """Expose stored rasters and groups through native xarray opening functions.

    The engine runs only when named. It claims no file on its own, so
    `xr.open_dataset("scene.tif")` keeps choosing among xarray's own engines.

    Examples:
        >>> xr.open_dataset("scene.zarr", engine="geosave").gs.variables
        ('red', 'nir')
        >>> tuple(xr.open_datatree("prepared/s1.zarr", engine="geosave").children)
        ('label', 'sentinel_2_l2a')
    """

    description = "GeoSave rasters and raster trees (COG, Zarr, NetCDF)"
    url = "https://github.com/weedkat/geosave"
    open_dataset_parameters = ("filename_or_obj", "drop_variables")

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **options: Any,
    ) -> xr.Dataset:
        """Open what `open_raster` reads.

        Args:
            filename_or_obj: One raster file or store. Open split files through
                their Item catalog with stac.load_dataset(rows.gs.to_items(), ...).
            drop_variables: Variable name or names to leave out.
            **options: Read options the format's reader accepts, as
                `mask_and_scale=True`.

        Returns:
            Raster Dataset, lazily indexed; xarray applies `chunks` itself.
        """
        options.setdefault("chunks", None)
        raster = open_raster(filename_or_obj, **options)
        if drop_variables is None:
            return raster
        result = raster.drop_vars(drop_variables)
        result.set_close(raster.close)
        return result

    def open_datatree(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **options: Any,
    ) -> xr.DataTree:
        """Open a stored hierarchy, or a banded raster as a root-only tree.

        Args:
            filename_or_obj: One raster file or store, preserving stored groups.
            drop_variables: Unused; drop variables from the groups instead.
            **options: Read options passed to every group's reader.

        Returns:
            DataTree holding one group per raster.
        """
        options.setdefault("chunks", None)
        try:
            find_opener(filename_or_obj, RASTER_TREE_OPENERS)
        except ValueError:
            pass
        else:
            return open_raster_tree(filename_or_obj, **options)
        result = open_raster(filename_or_obj, **options)
        tree = xr.DataTree(result)
        tree.set_close(result.close)
        return tree
