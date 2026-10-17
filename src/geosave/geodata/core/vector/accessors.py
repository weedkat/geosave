"""Register `gs` on DataFrames and Series, choosing the accessor from the columns."""

from __future__ import annotations

import pandas as pd

from .catalog import GeoCatalog
from .item import GeoItem
from .sample import GeoSample
from .samples import GeoSamples
from .vector import GeoVector


# pandas registers a class and calls it with the object, so `__new__` returns
# the accessor the data calls for; `__init__` is skipped for a foreign instance.
@pd.api.extensions.register_dataframe_accessor("gs")
class FrameAccessor:
    """Read a sample table through `GeoSamples`, an item table through `GeoCatalog`, any other frame through `GeoVector`."""

    def __new__(cls, data: pd.DataFrame) -> GeoVector:
        if all(column in data.columns for column in GeoSamples.FIELDS):
            return GeoSamples(data)
        if all(column in data.columns for column in GeoCatalog.COLUMNS):
            return GeoCatalog(data)
        return GeoVector(data)


@pd.api.extensions.register_series_accessor("gs")
class RowAccessor:
    """Read a sample-table row through `GeoSample`, an item-table row through `GeoItem`."""

    def __new__(cls, row: pd.Series) -> GeoSample | GeoItem:
        if all(column in row.index for column in GeoSamples.FIELDS):
            return GeoSample(row)
        if all(column in row.index for column in GeoCatalog.COLUMNS):
            return GeoItem(row)
        # pandas reads AttributeError as "this object has no such accessor".
        raise AttributeError(
            "gs reads a row of an item table or a sample table; this row has "
            "neither table's columns"
        )
