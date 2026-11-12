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
        >>> items.iloc[[0]].gs.to_dataset(grid)
        <xarray.Dataset> Dimensions: (time: 1, y: 50, x: 70) Data variables: red
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

        Note:
            Catalog-to-layer ownership is deferred for review in the CF/xarray
            adoption plan. RasterLayer still calls this row wrapper, but that
            coupling is not the settled catalog design. Keep behavior unchanged
            until the layer's source contract is reviewed.

        Args:
            geobox: Grid to read onto, such as a map's view.
            variables: ODC band names or aliases, as in `load_dataset`. None
                reads every data asset.

        Returns:
            Dataset on `geobox`, float32 with nodata as NaN.

        Raises:
            ValueError: A requested band name or alias is unknown.

        Examples:
            >>> row.gs.read_pixels(view, variables=["red"]).red.shape
            (600, 900)
        """
        import numpy as np

        from geosave.geodata.stac import load_dataset

        pixels = load_dataset(
            [self.to_item()],
            geobox,
            bands=variables,
            groupby="id",
            chunks=None,
            dtype="float32",
            nodata=np.nan,
        )
        if "time" in pixels.dims:
            pixels = pixels.isel(time=0, drop=True)
        return cast("Dataset", pixels)

    def to_datatree(self, **options: Any) -> DataTree:
        """Load the Item lazily as ``{collection}/{item}/{asset}`` nodes, as stated.

        Args:
            **options: Forwarded to `stac.load_datatree`, such as `chunks` or `bands`.

        Returns:
            DataTree as a one-row catalog loads, with the Collection metadata
            on its collection node.

        Raises:
            ValueError: A band states no grid in `proj:*`.

        Examples:
            >>> scene = rows.iloc[0].gs.to_datatree()
            >>> list(scene.gs.datasets)
            ['s2/S2B_33UUP_20250115/red', 's2/S2B_33UUP_20250115/swir']
        """
        import pystac

        from geosave.geodata.stac.load import load_datatree

        item = self.to_item()
        # pandas carries the table's footer onto each of its rows.
        stored = self._row.attrs.get(FOOTER_KEY, {}).get("collections", {})
        fields = stored.get(item.collection_id)
        collection = None if fields is None else pystac.Collection.from_dict(fields)
        collections = [] if collection is None else [collection]
        return cast(
            "DataTree", load_datatree([item], collections=collections, **options)
        )
