"""Group an Item table's rows by their original positions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame


class GeoSamples:
    """Group native catalog rows by their original positions.

    The catalog table is copied to preserve row positions after source edits.
    Nested Python values such as asset dictionaries remain shared.
    Indexing and iteration return GeoDataFrames. Optional ``slots`` records
    temporal membership as ordinary DataFrames with time, collection and indices
    columns. Empty lists denote missing observations; NaT denotes static rows.
    Grouping metadata is never attached to selected rows as execution settings.

    Examples:
        >>> samples = items.gs.sample(collections=["optical"], freq="MS", window=3)
        >>> rows = samples[0]
        >>> rows.equals(items.iloc[samples.indices[0]])
        True
        >>> with rows.gs.to_datatree() as tree:
        ...     print(tree.gs.groups)
        ('optical',)
    """

    def __init__(
        self,
        catalog: gpd.GeoDataFrame,
        indices: list[list[int]],
        *,
        slots: list[pd.DataFrame] | None = None,
        footprints: gpd.GeoSeries | None = None,
    ) -> None:
        """Keep source row positions and optional spatial/temporal membership."""
        self.catalog = catalog.copy()
        self.indices = indices
        self.slots = slots
        self._footprints = footprints

    @property
    def times(self) -> list[pd.DatetimeIndex]:
        """Return each window's ordered slot labels; spatial-only samples have none."""
        if self.slots is None:
            return []
        times = []
        for frame in self.slots:
            times.append(
                pd.DatetimeIndex(frame["time"].dropna().unique()).sort_values()
            )
        return times

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.indices)

    def __getitem__(self, position: int) -> GeoDataFrame:
        """Return the item table's rows of one sample."""
        return cast("GeoDataFrame", self.catalog.iloc[self.indices[position]])

    def __iter__(self) -> Iterator[GeoDataFrame]:
        """Yield each sample's rows in order."""
        for position in range(len(self)):
            yield self[position]

    @property
    def footprints(self) -> GeoDataFrame:
        """Return Item counts and the founding footprint of each planned group.

        For manually constructed groups without footprints, use the Items'
        intersection, falling back to the smallest footprint for zero-area overlap.
        """
        if self._footprints is not None:
            return cast(
                "GeoDataFrame",
                gpd.GeoDataFrame(
                    {"items": [len(rows) for rows in self.indices]},
                    geometry=self._footprints.reset_index(drop=True),
                    crs=self.catalog.crs,
                ),
            )
        item_counts = []
        footprints = []
        for rows in self:
            item_geometries = rows.geometry.to_numpy()
            footprint = shapely.intersection_all(item_geometries)
            if shapely.area(footprint) == 0:
                smallest_position = np.argmin(shapely.area(item_geometries))
                footprint = item_geometries[smallest_position]
            item_counts.append(len(rows))
            footprints.append(footprint)
        frame = gpd.GeoDataFrame(
            {"items": item_counts}, geometry=footprints, crs=self.catalog.crs
        )
        return cast("GeoDataFrame", frame)
