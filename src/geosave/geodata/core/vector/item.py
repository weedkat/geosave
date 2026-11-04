"""Read one spatial catalog record from a pandas Series."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
import pandas as pd
from shapely.geometry.base import BaseGeometry

from geosave.geodata.stac.catalog import FOOTER_KEY

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pystac
    from odc.geo.geobox import GeoBox
    from geosave.geodata import Dataset, DataTree
    from geosave.geodata.core.anchor import GeoAnchor


class GeoCatalogItem:
    """Read one spatial catalog record, optionally holding STAC assets.

    A Series represents a record when it carries a geometry field. It does
    not inherit its parent GeoDataFrame's CRS. STAC conversion and raster
    loading require the additional Item fields.

    Args:
        row: Spatial record with a `geometry` field; STAC records use WGS84.

    Examples:
        >>> row = items.iloc[0]
        >>> row.gs.anchor.timespan
        (datetime.datetime(2024, 3, 8, 0, 0), datetime.datetime(2024, 3, 8, 0, 0))
        >>> row.gs.to_datatree().gs.to_dataset()
        <xarray.Dataset> Dimensions: (y: 50, x: 70) Data variables: red
    """

    def __init__(self, row: pd.Series) -> None:
        """Bind a spatial record; ordinary field columns have no accessor."""
        if "geometry" not in row.index or (
            row["geometry"] is not None
            and not isinstance(row["geometry"], BaseGeometry)
        ):
            raise AttributeError("gs reads a spatial record with a geometry field")
        self._row = row

    @property
    def anchor(self) -> GeoAnchor:
        """Return STAC geometry and time without opening assets.

        A standalone STAC row must use WGS84. For projected or generic rows,
        select a one-row GeoDataFrame to retain its CRS and use its gs.anchor.
        """
        from geosave.geodata.core.anchor import GeoAnchor
        from geosave.geodata.stac import item as stac_item

        item = self.to_item()
        if item.geometry is None:
            raise ValueError("the Item has no geographic geometry")
        return GeoAnchor.from_geometry(
            item.geometry, timespan=stac_item.read_timespan(item)
        )

    def to_item(self) -> pystac.Item:
        """Build the pystac Item this row states.

        Returns:
            New Item holding the row's assets and properties.

        Raises:
            ValueError: The record lacks the required STAC Item fields.
        """
        from geosave.geodata.stac.catalog import to_item

        return to_item(self._row)

    def read_pixels(
        self, geobox: GeoBox, *, variables: Sequence[str] | None = None
    ) -> Dataset:
        """Read the row's pixels inside `geobox`, each file from the overview it needs.

        Args:
            geobox: Grid to read onto, such as a map's view.
            variables: Variables to read. None reads every data asset.

        Returns:
            Dataset on `geobox`, float32 with nodata as NaN.

        Raises:
            KeyError: A requested variable is not in the row.

        Examples:
            >>> row.gs.read_pixels(view, variables=["red"]).red.shape
            (600, 900)
        """
        from geosave.geodata.io.stac import read_scene_pixels

        return cast("Dataset", read_scene_pixels(self.to_item(), geobox, variables))

    def to_datatree(self, **options: Any) -> DataTree:
        """Open the Item lazily as a scene tree, one node per grid, at its date.

        Args:
            **options: Forwarded to each asset's GDAL reader, such as
                `chunks` or `mask_and_scale`.

        Returns:
            DataTree of grid nodes (`r10m`, `r20m`), each a Dataset of the
            variables on that grid, named and attributed as STAC states them,
            with the Item's properties on its `stac_properties` coordinate.
            The root carries the Collection, where the table holds it, and
            the extension schemas as `StacMetadata` attrs, so `to_catalog`
            writes the scene as this row does. Closing it closes every file.

        Raises:
            ValueError: A file has no grid, in itself or in its asset.

        Examples:
            >>> scene = items.iloc[0].gs.to_datatree()
            >>> list(scene.gs.rasters)
            ['r10m', 'r20m']
            >>> scene.gs.to_dataset().gs.variables
            ('red', 'swir')
        """
        import pystac

        from geosave.geodata.io.stac import open_scene
        from geosave.geodata.stac.item import build_stac_metadata

        item = self.to_item()
        # pandas carries the table's footer onto each of its rows.
        stored = self._row.attrs.get(FOOTER_KEY, {}).get("collections", {})
        fields = stored.get(item.collection_id)
        collection = None if fields is None else pystac.Collection.from_dict(fields)
        scene = open_scene(item, **options)
        scene.attrs.update(build_stac_metadata([item], collection).to_attrs())
        return cast("DataTree", scene)
