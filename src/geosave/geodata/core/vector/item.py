"""Read one spatial catalog record from a pandas Series."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry


if TYPE_CHECKING:
    import pystac
    from geosave.geodata import Dataset
    from geosave.geodata.core.region import GeoRegion
    from geosave.geodata.core.vector.catalog import Rule


class GeoCatalogItem:
    """Read one spatial catalog record, optionally a STAC Item row.

    A Series represents a record when it carries a geometry field. It does
    not inherit its parent GeoDataFrame's CRS, so a STAC row is read as WGS84;
    select a one-row GeoDataFrame to keep another CRS.

    Args:
        row: Spatial record with a `geometry` field; STAC records use WGS84.

    Examples:
        >>> row = catalog.iloc[0]
        >>> row.gs.region.timespan
        (datetime.datetime(2024, 3, 8, 0, 0), datetime.datetime(2024, 3, 8, 0, 0))
        >>> scene = stac.load_dataset([row.gs.to_item()], chunks={})
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
            *rules: Table rules, such as `stac.catalog.stac_items`.

        Raises:
            VectorValidationError: A rule is broken.

        Examples:
            >>> catalog.iloc[0].gs.validate(stac_items)
        """
        one_row_catalog = gpd.GeoDataFrame([self._row], geometry="geometry")
        # pandas carries the table's attrs onto each of its rows.
        one_row_catalog.attrs = self._row.attrs
        for rule in rules:
            rule(one_row_catalog)

    @property
    def region(self) -> GeoRegion:
        """Return the Item's footprint and time without opening assets.

        Raises:
            VectorValidationError: The geometry is broken, or the record is
                not a STAC Item row.
        """
        from geosave.geodata.core.region import GeoRegion
        from geosave.geodata.core.vector.catalog import valid_geometry
        from geosave.geodata.core.vector.utils.time import read_timespan

        self.validate(valid_geometry)
        item = self.to_item()
        return GeoRegion.from_geometry(
            cast("dict[str, Any]", item.geometry),
            timespan=read_timespan(
                item.datetime,
                item.properties.get("start_datetime"),
                item.properties.get("end_datetime"),
            ),
        )

    def to_item(self) -> pystac.Item:
        """Convert this Item row to a native pystac Item, as `catalog.gs.to_items` does.

        Returns:
            The Item, with absolute asset hrefs and its Collection linked
            where the table defines it.

        Raises:
            VectorValidationError: The record is not a STAC Item row.

        Examples:
            >>> catalog.iloc[0].gs.to_item().id
            '20250601T103000'
        """
        import shapely
        from stac_geoparquet import to_dict

        from geosave.geodata.stac.catalog import (
            build_item,
            read_collections,
            stac_items,
        )

        self.validate(stac_items)
        item_dict = to_dict(self._row.to_dict())
        # The table keeps no bbox column; the geometry states it.
        item_dict["bbox"] = shapely.bounds(self._row["geometry"]).tolist()

        # pandas carries the table's attrs onto each of its rows.
        return build_item(item_dict, read_collections(self._row.attrs))

    def read_pixels(self, *args: Any, **kwargs: Any) -> Dataset:
        """Read the row's pixels onto a grid from the overview that fits it.

        The explorer's viewport reads need it. Load whole Items with
        `stac.load_dataset([row.gs.to_item()], ...)` instead.

        Raises:
            NotImplementedError: Always, until the overview-aware reader is rebuilt.
        """
        raise NotImplementedError(
            "read_pixels awaits the overview-aware reader; load whole Items "
            "with stac.load_dataset([row.gs.to_item()], ...)"
        )
