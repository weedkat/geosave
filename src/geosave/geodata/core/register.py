"""Register GeoSave accessors on native xarray and pandas containers."""

from __future__ import annotations

import pandas as pd
import xarray as xr

from .raster.dataarray import GeoDataArray
from .raster.dataset import GeoDataset
from .raster.datatree import GeoDataTree
from .vector.catalog import GeoCatalog
from .vector.item import GeoCatalogItem

xr.register_dataarray_accessor("gs")(GeoDataArray)
xr.register_dataset_accessor("gs")(GeoDataset)
xr.register_datatree_accessor("gs")(GeoDataTree)
pd.api.extensions.register_dataframe_accessor("gs")(GeoCatalog)
pd.api.extensions.register_series_accessor("gs")(GeoCatalogItem)
