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
    from geosave.geodata.core.region import GeoRegion


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

    def validate(self, *, geometry: bool = True, stac_item: bool | None = None) -> None:
        """Check a spatial record without opening its assets.

        Args:
            geometry: Require a present, nonempty, valid geometry.
            stac_item: True requires STAC Item fields. False rejects STAC
                fields or metadata to require an ordinary feature. None skips
                this check. Does not validate full schemas or asset access.

        A Series has no inherited GeoDataFrame CRS. Use a one-row GeoDataFrame
        to validate CRS or work with projected geometry.

        Raises:
            VectorValidationError: Geometry or required Item fields are missing
                or invalid. Geometry errors include the record's index label.

        Examples:
            >>> rows.iloc[0].gs.validate(stac_item=True)
        """
        from geosave.geodata.core.vector.utils.geometry import validate_geometry
        from geosave.geodata.errors import VectorValidationError
        from geosave.geodata.stac.catalog import FOOTER_KEY, ITEM_COLUMNS

        if geometry:
            try:
                validate_geometry(cast("BaseGeometry | None", self._row["geometry"]))
            except VectorValidationError as error:
                raise VectorValidationError(
                    f"Record {self._row.name!r}: {error}"
                ) from error

        if stac_item is True:
            missing = set(ITEM_COLUMNS) - set(self._row.index)
            if missing:
                raise VectorValidationError(
                    f"STAC Item operations need columns {sorted(missing)}; item tables come "
                    "from to_catalog, StacClient.search, or reading a stac-geoparquet file"
                )

        if stac_item is False:
            if FOOTER_KEY in self._row.attrs or all(
                column in self._row.index for column in ITEM_COLUMNS
            ):
                raise VectorValidationError(
                    "ordinary features required; write catalog rows with "
                    "to_geoparquet to preserve STAC metadata"
                )

    @property
    def region(self) -> GeoRegion:
        """Return STAC geometry and time without opening assets.

        A standalone STAC row must use WGS84. For projected or generic rows,
        select a one-row GeoDataFrame to retain its CRS and use its gs.region.
        """
        from geosave.geodata.core.region import GeoRegion
        from geosave.geodata.stac import item as stac_item

        item = self.to_item()
        self.validate()
        return GeoRegion.from_geometry(
            cast("dict[str, Any]", item.geometry),
            timespan=stac_item.read_timespan(
                item.datetime,
                item.properties.get("start_datetime"),
                item.properties.get("end_datetime"),
            ),
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
        from geosave.geodata.stac.publish import read_scene_pixels

        return cast("Dataset", read_scene_pixels(self.to_item(), geobox, variables))

    def to_datatree(self, **options: Any) -> DataTree:
        """Load the Item lazily, one node per STAC asset on its own grid.

        Args:
            **options: Forwarded to `stac.load_datatree`, such as `groupby`,
                `chunks` or `bands`.

        Returns:
            DataTree of ``{collection}/{asset}`` nodes, as a one-row
            catalog loads, with the Collection metadata and the Item's time on
            the collection node. The Item's geometry is its region, beside the
            whole grids.

        Raises:
            ValueError: A band states no grid in `proj:*`.

        Examples:
            >>> scene = rows.iloc[0].gs.to_datatree()
            >>> list(scene.gs.datasets)
            ['s2/red', 's2/swir']
            >>> scene.gs.group_grid()["s2"].gs.to_dataset(grid="r10m").gs.variables
            ('red', 'swir')
        """
        import pystac

        from geosave.geodata.stac.load import load_datatree
        from geosave.geodata.stac.item import read_footprint

        item = self.to_item()
        # pandas carries the table's footer onto each of its rows.
        stored = self._row.attrs.get(FOOTER_KEY, {}).get("collections", {})
        fields = stored.get(item.collection_id)
        collection = None if fields is None else pystac.Collection.from_dict(fields)
        tree = load_datatree([item], collection=collection, **options)
        # The Item's geometry is its region; the grid stays whole.
        footprint = read_footprint([item])
        if footprint is not None:
            tree = tree.gs.write_extent(footprint)
        return cast("DataTree", tree)
