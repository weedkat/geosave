"""Place, draw, write and convert spatial records on native GeoDataFrames."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import Geometry

from geosave.geodata.errors import VectorValidationError


if TYPE_CHECKING:
    import holoviews as hv
    import pystac
    from typing_extensions import Unpack
    from upath import UPath
    from geosave.geodata import Dataset
    from geosave.geodata.io.storage import Location
    from geosave.geodata.io.vector.ogr import OGRWriteOptions
    from geosave.geodata.io.vector.geoparquet import GeoParquetWriteOptions

    from geosave.geodata.core.region import GeoRegion
    from geosave.geodata.stac.load import StacLoadOptions
    from geosave.geodata.viz.explore import Explorer

# A rule raises VectorValidationError when the table breaks it.
type Rule = Callable[[gpd.GeoDataFrame], None]


# Rules a table can be asked to meet; `rows.gs.validate(valid_geometry)` runs one.
# The rules of a STAC Item table live in `stac.catalog`.


def valid_geometry(rows: gpd.GeoDataFrame) -> None:
    """Require a present, nonempty, valid geometry in every row.

    Raises:
        VectorValidationError: A geometry is broken; the error names its row.
    """
    from geosave.geodata.core.vector.utils.geometry import validate_geometry

    for position, shape in enumerate(rows.geometry):
        try:
            validate_geometry(shape)
        except VectorValidationError as error:
            identity = rows.iloc[position].get("id", position)
            raise VectorValidationError(
                f"Row at position {position} (Item {identity!r}): {error}"
            ) from error


def dated(rows: gpd.GeoDataFrame) -> None:
    """Require each row's time stated whole: a datetime, or an ordered start/end pair.

    Raises:
        VectorValidationError: A row's time is incomplete or out of order.
    """
    from geosave.geodata.core.vector.utils.time import validate_time

    validate_time(rows)


class GeoCatalog:
    """Place, draw and write spatial records, and convert STAC Item rows.

    Native GeoPandas masks select rows. Spatial operations request the geometry
    and CRS guarantees they need through `validate`. The region combines
    geometry and temporal coverage without opening assets.

    A table of STAC Items, as `stac.create_catalog` lays it out, additionally
    converts back to Items and loads as a Dataset. Its Collection definitions
    travel in `attrs["stac-geoparquet"]`; ordinary features need neither Item
    columns nor Collection metadata.

    Raises:
        AttributeError: `data` is not a GeoDataFrame, or has no active
            geometry column.

    Examples:
        >>> catalog = stac.create_catalog([published_item])
        >>> recent = catalog[catalog["datetime"] >= pd.Timestamp("2025-06-01", tz="UTC")]
        >>> scene = recent.gs.to_dataset(chunks={})
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Bind the accessor to one GeoDataFrame.

        Args:
            data: GeoDataFrame with an active geometry column.

        Raises:
            AttributeError: The input is not a GeoDataFrame or has no active
                geometry column.
        """
        # pandas reads AttributeError as "this object has no such accessor".
        if not isinstance(data, gpd.GeoDataFrame):
            raise AttributeError(
                f"gs reads a GeoDataFrame, got {type(data).__name__}; build one "
                f"with gpd.GeoDataFrame(...)"
            )
        if data.active_geometry_name is None:
            raise AttributeError("gs needs an active geometry column")
        self._data = data

    def validate(self, *rules: Rule) -> None:
        """Require every rule, without changing rows or opening assets.

        Args:
            *rules: Table rules the operation needs, such as `stac_items` or
                `valid_geometry`.

        Raises:
            VectorValidationError: A rule is broken; the error names the row.

        Examples:
            >>> rows.gs.validate(valid_geometry, stac.catalog.stac_items)
        """
        for rule in rules:
            rule(self._data)

    @property
    def crs(self) -> OdcCRS:
        """Return the coordinate reference system every geometry shares.

        Raises:
            VectorValidationError: The frame declares no CRS.
        """
        if self._data.crs is None:
            raise VectorValidationError(
                "The vector has no CRS. Assign the known coordinate reference "
                "system with frame.set_crs(...); this does not reproject geometries."
            )
        return OdcCRS(self._data.crs)

    @property
    def footprint(self) -> Geometry:
        """Return the union of all geometries in the vector CRS.

        Returns:
            ODC geometry carrying the vector CRS.

        Raises:
            ValueError: The frame holds no row.
        """
        if self._data.empty:
            raise VectorValidationError(
                "An empty vector has no footprint. Select at least one row."
            )
        self.validate(valid_geometry)
        return Geometry(self._data.geometry.union_all(), crs=self.crs)

    @property
    def region(self) -> GeoRegion:
        """Return feature union and the temporal envelope of dated rows.

        Timeless rows do not widen the envelope. No assets are opened and no
        pixel grid is inferred. Invalid geometry or incomplete dates raise.
        """
        from geosave.geodata.core.region import GeoRegion
        from geosave.geodata.core.vector.utils.time import read_times

        region = GeoRegion.from_geometry(self.footprint)
        self.validate(dated)
        times = read_times(self._data)
        dated_times = times.loc[times["start"].notna()]
        if dated_times.empty:
            return region

        return region.set_time((dated_times["start"].min(), dated_times["end"].max()))

    def plot(self, *, color: str = "tab:blue", alpha: float = 0.3) -> hv.Overlay:
        """Draw this vector's geometry in its own coordinates.

        Polygons draw outlined and filled, lines and points in `color`. The
        drawing overlays a raster plot with `*` where both share a CRS.

        Args:
            color: Colour of outlines, lines and points.
            alpha: Opacity of the polygon fill.

        Returns:
            Overlay of one element per geometry type the vector holds.

        Raises:
            ValueError: The vector holds no geometry or a GeometryCollection.

        Examples:
            >>> plots.gs.plot()
            >>> scene.gs.plot() * plots.to_crs(region.crs).gs.plot()
        """
        from geosave.geodata.viz.plot import draw_vector

        return draw_vector(self._data, color=color, alpha=alpha)

    def explore(
        self,
        *,
        column: str | None = None,
        basemap: str | None = None,
        color: str = "tab:blue",
        alpha: float = 0.3,
    ) -> Explorer:
        """Construct a web map viewer with a sidebar to explore this vector.

        Features have hover and click identification. Property and date filters
        narrow the table and map. Rows start unselected. Selecting a row
        draws its geometry; the Raster control opens Item assets separately.

        Args:
            column: Column that colours the features, or None for one colour.
            basemap: Basemap under the layers, as `viz.explore.page.map.BASEMAPS` names it.
                None uses the light basemap.
            color: Colour of every feature when no column colours them.
            alpha: Opacity of polygon fill; 0 draws boundaries only.

        Returns:
            Explorer, shown by a notebook or started with `.serve()`.

        Raises:
            KeyError: `column` is not a column of the frame.
            ValueError: The vector has no CRS, holds no geometry, or holds a
                GeometryCollection.

        Examples:
            >>> fields.gs.explore(column="crop")
            >>> fields.gs.explore(alpha=0.0, basemap="Imagery")  # boundaries only
        """
        from matplotlib.colors import to_hex
        from geosave.geodata.viz.explore import Explorer

        viewer = Explorer(tables={"Features": self._data}, basemap=basemap)
        layer = viewer.tables.tables["Features"].footprints
        layer.column = column
        layer.color = to_hex(color)
        layer.alpha = alpha
        return viewer

    def rasterize(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Burn geometries or one property onto an exact grid.

        Raises:
            NotImplementedError: Always, until rasterization is rebuilt.
        """
        raise NotImplementedError("rasterize is not rebuilt yet")

    @property
    def collections(self) -> dict[str, pystac.Collection]:
        """Return the Collections the table's footer metadata holds, by id."""
        from geosave.geodata.stac.catalog import read_collections

        return read_collections(self._data.attrs)

    def to_geojson(
        self,
        path: Location,
        *,
        overwrite: bool = False,
        **options: Unpack[OGRWriteOptions],
    ) -> UPath:
        """Write this vector as one GeoJSON file.

        GeoJSON states coordinates in WGS84, so a projected vector is
        reprojected on the way out. Reach for `to_geopackage` or
        `to_geoparquet` to keep the vector's own CRS.

        Args:
            path: Output path or URL ending in `.geojson` or `.json`.
            overwrite: Replace an existing file when true.
            **options: GeoJSON write options.

        Returns:
            Local writes return a PosixUPath; URL writes return a UPath carrying
            its storage options.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.gs.to_geojson("plots.geojson")
            PosixUPath('plots.geojson')
        """
        from geosave.geodata.io.vector import ogr

        return ogr.write_geojson(self._data, path, overwrite=overwrite, **options)

    def to_geopackage(
        self,
        path: Location,
        *,
        layer: str | None = None,
        overwrite: bool = False,
        **options: Unpack[OGRWriteOptions],
    ) -> UPath:
        """Write this vector as one layer in a GeoPackage.

        Args:
            path: Output path or URL ending in `.gpkg`.
            layer: Layer name. None names the layer after the file stem.
            overwrite: Replace an existing file when true.
            **options: GeoPackage write options.

        Returns:
            Local writes return a PosixUPath; URL writes return a UPath carrying
            its storage options.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.gs.to_geopackage("survey.gpkg", layer="plots")
            PosixUPath('survey.gpkg')
        """
        from geosave.geodata.io.vector import ogr

        return ogr.write_geopackage(
            self._data, path, layer=layer, overwrite=overwrite, **options
        )

    def to_geoparquet(
        self,
        path: Location,
        *,
        collections: Iterable[pystac.Collection] | None = None,
        overwrite: bool = False,
        **options: Unpack[GeoParquetWriteOptions],
    ) -> UPath:
        """Write features or a STAC catalog to one GeoParquet file.

        Frames carrying `attrs["stac-geoparquet"]` use STAC encoding,
        preserving Collection definitions and absolute asset hrefs. Ordinary
        features use GeoPandas encoding. Catalog metadata is checked at write
        time, so missing fields raise instead of losing their STAC encoding.

        Args:
            path: Output path or URL ending in `.parquet` or `.geoparquet`.
            collections: Override catalog Collection definitions by id. None
                preserves the frame's definitions; feature tables take none.
            overwrite: Replace an existing file when true.
            **options: GeoParquet/storage options. STAC catalogs require WKB
                geometry, a bbox and no serialized pandas index.

        Returns:
            UPath of the file, carrying its filesystem options.

        Raises:
            FileExistsError: The path exists and overwrite is false.
            ValueError: Catalog metadata or fields are invalid, the rows are
                samples, or options conflict with the STAC encoding.
            TypeError: An option is unsupported or repeated.

        Examples:
            >>> catalog.gs.to_geoparquet("data/catalog.parquet")
            PosixUPath('data/catalog.parquet')
            >>> plots.gs.to_geoparquet("plots.parquet", compression="zstd")
            PosixUPath('plots.parquet')
        """
        from geosave.geodata.io.vector import geoparquet, stac_geoparquet
        from geosave.geodata.stac.catalog import FOOTER_KEY

        if FOOTER_KEY in self._data.attrs:
            return stac_geoparquet.write_stac_geoparquet(
                self._data,
                path,
                collections=collections,
                overwrite=overwrite,
                **options,
            )
        if collections is not None:
            raise ValueError(
                "Collection definitions require a catalog with STAC metadata"
            )
        return geoparquet.write_geoparquet(
            self._data, path, overwrite=overwrite, **options
        )

    def to_dataset(self, **options: Unpack[StacLoadOptions]) -> Dataset:
        """Load one collection's Item rows onto one grid, as `stac.load_dataset` does.

        A shortcut for `stac.load_dataset(rows.gs.to_items(), ...)`, with its
        default stamps. Use `stac.load_native_datasets` on the Items to keep
        each native grid.

        Args:
            **options: Options of `odc.stac.load`, such as `geobox`, `like`,
                `chunks` and `with_properties`.

        Returns:
            Dataset with the rows' bands, period and Collection.

        Raises:
            VectorValidationError: The rows are not STAC Items, or name
                several collections.

        Examples:
            >>> scene = catalog.gs.to_dataset(chunks={}, with_properties=["eo:cloud_cover"])
        """
        from geosave.geodata.stac.catalog import one_collection
        from geosave.geodata.stac.load import load_dataset

        self.validate(one_collection)
        dataset = load_dataset(self.to_items(), **options)
        return cast("Dataset", dataset)

    def to_items(self) -> list[pystac.Item]:
        """Convert Item rows to native pystac Items, ready for `stac.load_dataset`.

        The stac-geoparquet library decodes the rows, and each Item links to
        the Collection the table defines for it.

        Returns:
            One Item per row, in row order, with WGS84 geometry.

        Raises:
            VectorValidationError: The rows are not STAC Items.

        Examples:
            >>> scene = stac.load_dataset(catalog.gs.to_items(), chunks={})
        """
        from geosave.geodata.stac.catalog import to_items

        return to_items(self._data)
