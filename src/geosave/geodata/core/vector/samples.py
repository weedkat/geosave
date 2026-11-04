"""Group an item table's rows into samples, each opened as one DataTree."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import shapely

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame


class GeoSamples:
    """Hold an item table and, for each sample, the positions of its rows.

    `GeoCatalog.sample` builds them. Samples may share rows: a swath belongs
    to every footprint it covers, and a timeless Item to every window, which a
    pandas GroupBy cannot hold. A sample is the table's rows at its positions,
    so it opens as the item table does.

    Args:
        catalog: Item table the positions select from.
        indices: For each sample, the positions of its rows.

    Examples:
        >>> samples = items.gs.sample(pair=["s2", "s1"], tolerance="2D")
        >>> samples.indices
        [[0, 5, 9], [2, 6, 9], [3, 7, 9]]
        >>> samples[0].gs.to_datatree().gs.groups
        ('s2', 's1', 'dem')
    """

    def __init__(self, catalog: gpd.GeoDataFrame, indices: list[list[int]]) -> None:
        """Keep the item table and each sample's row positions."""
        self.catalog = catalog
        self.indices = indices

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
        """Return one row per sample: its Item count, and where its Items overlap.

        Items that only touch share a line, which no map or grid can use, so
        such a sample takes its smallest Item's footprint instead.
        """
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
