"""The ingestor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from geosave.geodata.io import storage, write_zarr_tree
from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import xarray as xr
    from upath import UPath

    from geosave.geodata import GeoRegion
    from geosave.geodata.io.storage import Location


class RasterIngestor(ABC):
    """Load what a model predicts over a region; inference only.

    A subclass states where the rasters come from in `__call__`; writing
    them to files is the same for every source and lives here.
    """

    @abstractmethod
    def __call__(self, region: GeoRegion) -> list[xr.DataTree]:
        """Return the lazy raw rasters of each acquisition over the region.

        Args:
            region: Place and timespan to predict.

        Returns:
            One tree per acquisition, in acquisition order. A tree holds one
            node per raw raster under the name the preprocessor reads it by,
            in stored values with their nodata and packing stated; every
            node states the acquisition's scalar `time`.
        """

    def ingest_to_zarr(self, region: GeoRegion, destination: Location) -> list[UPath]:
        """Write each acquisition over a region as one Zarr store in a folder.

        The source is read once, chunk by chunk, so a region too large for
        memory can be predicted from the stores.

        Args:
            region: Place and timespan to predict.
            destination: Folder the stores are written in, each named by its
                acquisition time, such as `20240701T030551.zarr`.

        Returns:
            Path of each store, in acquisition order; `open_raster_tree`
            opens one as the lazy rasters calling the ingestor would return.

        Raises:
            FileExistsError: A store of an acquisition already exists.
        """
        folder = storage.create_upath(destination)

        stores = []
        for rasters in self(region):
            first = next(iter(rasters.children.values()))
            time = pd.Timestamp(first["time"].item())
            store = folder / f"{time:%Y%m%dT%H%M%S}.zarr"
            stores.append(write_zarr_tree(rasters, store))
        return stores


INGESTORS = Registry(
    RasterIngestor, imports=("geosave.ml.processing.raster.ingestion.sentinel2",)
)
