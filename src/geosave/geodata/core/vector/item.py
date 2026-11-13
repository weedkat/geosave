"""Read one spatial catalog record from a pandas Series."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
import pandas as pd
from shapely.geometry.base import BaseGeometry


if TYPE_CHECKING:
    import pystac
    from geosave.geodata import Dataset, DataTree
    from geosave.geodata.core.region import GeoRegion
    from geosave.geodata.core.vector.catalog import Rule


class GeoCatalogItem:
    """Read one spatial catalog record, optionally holding STAC assets.

    A Series represents a record when it carries a geometry field. It does
    not inherit its parent GeoDataFrame's CRS. STAC conversion and raster
    loading require the additional Item fields.

    Args:
        row: Spatial record with a `geometry` field; STAC records use WGS84.

    Examples:
        >>> row = items.iloc[0]
        >>> row.gs.region.timespan
        (datetime.datetime(2024, 3, 8, 0, 0), datetime.datetime(2024, 3, 8, 0, 0))
    """

    def __init__(self, row: pd.Series) -> None:
        """Bind a spatial record; ordinary field columns have no accessor."""
        if "geometry" not in row.index or (
            row["geometry"] is not None
            and not isinstance(row["geometry"], BaseGeometry)
        ):
            raise AttributeError("gs reads a spatial record with a geometry field")
        self._row = row

    def validate(self, *rules: Rule) -> None:
        """Require every table rule of this record, as a one-row table.

        Args:
            *rules: Table rules from `core.vector.catalog`, such as `stac_items`.

        Raises:
            VectorValidationError: A rule is broken.

        Examples:
            >>> rows.iloc[0].gs.validate(stac_items)
        """
        import geopandas as gpd

        rows = gpd.GeoDataFrame([self._row], geometry="geometry")
        rows.attrs = self._row.attrs
        for rule in rules:
            rule(rows)

    @property
    def region(self) -> GeoRegion:
        """Return STAC geometry and time without opening assets.

        A standalone STAC row must use WGS84. For projected or generic rows,
        select a one-row GeoDataFrame to retain its CRS and use its gs.region.
        """
        from geosave.geodata.core.region import GeoRegion
        from geosave.geodata.core.vector.utils.time import read_timespan
        from geosave.geodata.stac.catalog import to_item

        from geosave.geodata.core.vector.catalog import valid_geometry

        self.validate(valid_geometry)
        item = to_item(self._row)
        return GeoRegion.from_geometry(
            cast("dict[str, Any]", item.geometry),
            timespan=read_timespan(
                item.datetime,
                item.properties.get("start_datetime"),
                item.properties.get("end_datetime"),
            ),
        )

    def to_item(self) -> pystac.Item:
        """Build the pystac Item this row states, its asset hrefs absolute.

        Raises:
            VectorValidationError: The record lacks the STAC Item fields.
        """
        from geosave.geodata.stac.catalog import to_item

        return to_item(self._row)

    def read_pixels(self, *args: Any, **kwargs: Any) -> Dataset:
        """Read the row's pixels onto a grid.

        Raises:
            NotImplementedError: Always, while it is being rebuilt with catalog-tree loading and publication.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.read_pixels is being rebuilt with catalog-tree loading and publication"
        )

    def to_datatree(self, **options: Any) -> DataTree:
        """Load the Item lazily as a STAC tree: collection, Item, asset.

        Args:
            **options: Forwarded to `stac.load_datatree`, such as `chunks` or `bands`.

        Returns:
            DataTree as a one-row catalog loads, with its Collection on the
            collection node.
        """
        import pystac

        from geosave.geodata.stac.catalog import FOOTER_KEY
        from geosave.geodata.stac.load import load_datatree

        item = self.to_item()
        # pandas carries the table's footer onto each of its rows.
        stored = self._row.attrs.get(FOOTER_KEY, {}).get("collections", {})
        collections = []
        if item.collection_id in stored:
            collections.append(pystac.Collection.from_dict(stored[item.collection_id]))
        return cast(
            "DataTree", load_datatree([item], collections=collections, **options)
        )
