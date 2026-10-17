"""The `gs` accessor of one item-table row: one STAC Item, read as a Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd

if TYPE_CHECKING:
    import pandas as pd
    import pystac

    from geosave.geodata import Dataset
    from geosave.geodata.core.anchor import GeoAnchor
    from geosave.geodata.io.raster.stac import ChunkSpec


class GeoItem:
    """Read one row of an item table: one STAC Item.

    Args:
        row: One row of an item table, in WGS84 as item tables are.

    Examples:
        >>> row = items.iloc[0]
        >>> row.gs.anchor.timespan
        (datetime.datetime(2024, 3, 8, 0, 0), datetime.datetime(2024, 3, 8, 0, 0))
        >>> row.gs.to_xarray()
        <xarray.Dataset> Dimensions: (y: 50, x: 70) Data variables: red
    """

    def __init__(self, row: pd.Series) -> None:
        """Bind the accessor to one Item row."""
        self._row = row

    @property
    def anchor(self) -> GeoAnchor | None:
        """Return the stated grid and span, or warn and return None without a grid."""
        from geosave.geodata.core.anchor import GeoAnchor
        from geosave.geodata.stac.translate import item as stac_item

        item = self.to_item()
        grid = stac_item.read_grid(item)
        return None if grid is None else GeoAnchor(grid, stac_item.read_timespan(item))

    def to_item(self) -> pystac.Item:
        """Build the pystac Item this row states.

        Returns:
            New Item holding the row's assets and properties.
        """
        from geosave.geodata.stac.translate import table as stac_table

        row_table = gpd.GeoDataFrame([self._row], geometry="geometry", crs="EPSG:4326")
        (item,) = stac_table.to_items(row_table)
        return item

    def to_xarray(self, *, chunks: ChunkSpec = {}) -> Dataset:
        """Open the Item as a lazy xarray Dataset, on its native grid and dates.

        Args:
            chunks: File-opening chunk sizes. An empty mapping uses native
                Dask chunks; ``"auto"`` chooses sizes; None disables Dask.

        Returns:
            Dataset of the Item's variables, named and described as STAC
            states them, its root carrying the Collection's ACDD attrs when
            the table holds the Collection's definition.

        Examples:
            >>> items.iloc[0].gs.to_xarray().gs.variables
            ('red',)
        """
        from geosave.geodata.io.raster.stac import open_raster

        item = self.to_item()
        collection_fields = self._row.attrs.get("stac:collections", {})
        return cast(
            "Dataset",
            open_raster(
                [item],
                chunks=chunks,
                collection_fields=collection_fields.get(item.collection_id),
            ),
        )
