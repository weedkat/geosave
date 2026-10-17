"""The `gs` GeoDataFrame accessor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from upath import UPath
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo import SomeCRS
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import Geometry

from geosave.geodata.utils.datetime import AnchorDatetime
from geosave.geodata.utils.geometry import SomeGeometry, to_shapely

if TYPE_CHECKING:
    import holoviews as hv

    from geosave.geodata.viz.explorer import Explorer, Overlay
    from numpy.typing import DTypeLike
    from typing_extensions import Unpack

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.io.storage import Location
    from geosave.geodata.io.vector.geojson import GeoJSONWriteOptions
    from geosave.geodata.io.vector.geopackage import GeoPackageWriteOptions
    from geosave.geodata.io.vector.geoparquet import GeoParquetWriteOptions

    from geosave.geodata.core.anchor import GeoAnchor

from .query import SpatialPredicate


class GeoVector:
    """GeoSave operations on a GeoDataFrame, read through `gs`.

    A vector is geometry with information attached: an active geometry column,
    a CRS, and any other columns. Querying uses geometry and, for dated
    targets, the optional `datetime`, `start_datetime` and `end_datetime`
    columns. A frame of STAC Items reads through `GeoCatalog`, and a frame
    of samples through `GeoSamples`.

    Args:
        data: GeoDataFrame with an active geometry column.

    Raises:
        AttributeError: `data` is not a GeoDataFrame, or has no active
            geometry column.

    Examples:
        >>> plots = open_vector("plots.geojson")
        >>> plots.gs.crs.to_epsg()
        4326
        >>> mask = plots.gs.rasterize(scene, column="class")
        >>> plots.gs.to_geoparquet("plots.parquet")
        PosixUPath('plots.parquet')
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

    @property
    def crs(self) -> OdcCRS:
        """Return the coordinate reference system every geometry shares."""
        if self._data.crs is None:
            raise ValueError("this vector has no CRS")
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
            raise ValueError("an empty vector has no footprint")
        return Geometry(self._data.geometry.union_all(), crs=self.crs)

    def to_geojson(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
        **options: Unpack[GeoJSONWriteOptions],
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
        from geosave.geodata.io import geojson

        return geojson.write(self._data, path, overwrite=overwrite, **options)

    def to_geopackage(
        self,
        path: str | PathLike[str],
        *,
        layer: str | None = None,
        overwrite: bool = False,
        **options: Unpack[GeoPackageWriteOptions],
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
        from geosave.geodata.io import geopackage

        return geopackage.write(
            self._data, path, layer=layer, overwrite=overwrite, **options
        )

    def to_geoparquet(
        self,
        path: Location,
        *,
        overwrite: bool = False,
        **options: Unpack[GeoParquetWriteOptions],
    ) -> UPath:
        """Write this vector as one GeoParquet file.

        Args:
            path: Output path or URL ending in `.parquet` or `.geoparquet`.
            overwrite: Replace an existing file when true.
            **options: GeoParquet write options.

        Returns:
            Local writes return a PosixUPath; URL writes return a UPath carrying
            its storage options.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.gs.to_geoparquet("plots.parquet")
            PosixUPath('plots.parquet')
            >>> plots.gs.to_geoparquet(
            ...     "hf://buckets/fatmur/test/plots.parquet",
            ...     storage_options={"token": token},
            ... )
            'hf://buckets/fatmur/test/plots.parquet'
        """
        from geosave.geodata.io import geoparquet

        return geoparquet.write(self._data, path, overwrite=overwrite, **options)

    @classmethod
    def concat(cls, vectors: Iterable[gpd.GeoDataFrame]) -> GeoDataFrame:
        """Concatenate spatial records without interpreting their identity.

        Args:
            vectors: Frames already expressed on one CRS plane.

        Returns:
            New frame with unioned property columns, a fresh RangeIndex, and
            its geometry column named `geometry`.

        Raises:
            ValueError: No frames are supplied, they read through different
                accessors (an item table beside a sample table), their CRSs
                differ, or a frame holds a second column named `geometry`
                beside its active one.
        """
        items = list(vectors)
        if not items:
            raise ValueError("concat needs at least one GeoDataFrame")
        kinds = {type(item.gs).__name__ for item in items}
        if len(kinds) > 1:
            raise ValueError(
                f"cannot concatenate tables of different kinds: {sorted(kinds)}"
            )
        crs = OdcCRS(items[0].crs)
        mismatches = [str(item.crs) for item in items if OdcCRS(item.crs) != crs]
        if mismatches:
            raise ValueError(
                f"the vectors' CRSs differ from {crs}: {mismatches}; "
                "call to_crs explicitly"
            )
        frames: list[gpd.GeoDataFrame] = []
        for item in items:
            frame = item.copy()
            if frame.active_geometry_name != "geometry":
                if "geometry" in frame.columns:
                    raise ValueError(
                        "cannot concatenate an active geometry with another "
                        "column named 'geometry'"
                    )
                frame = frame.rename_geometry("geometry")
            frames.append(frame)
        return cast(
            "GeoDataFrame",
            gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs
            ),
        )

    def query(
        self,
        where: GeoAnchor
        | xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | SomeGeometry
        | None = None,
        *,
        when: AnchorDatetime | None = None,
        predicate: SpatialPredicate = "intersects",
    ) -> GeoDataFrame:
        """Select rows related to a place and a time span.

        One rule serves every vector, Items and samples alike: the predicate
        reads from each row to `where`, and a row's span, from
        `start_datetime` and `end_datetime` each falling back to `datetime`,
        must share an instant with `when`. A missing bound is open, so an
        undated row matches any time.

        Args:
            where: Anchor or geolocated xarray object, whose ground and span
                are used; or a geometry, an odc Geometry carrying its CRS and
                any other spelling read as WGS84. None selects by time alone.
            when: Time span, as `parse_daterange` reads it, such as `"2024-03"`.
                None takes the span of an anchor `where`, else selects by
                place alone.
            predicate: Exact relation from each row to `where`.

        Returns:
            Matching original rows, preserving their index, order and metadata.

        Raises:
            ValueError: The predicate is unsupported, `where` carries no
                locatable grid, or a date column holds invalid values.

        Examples:
            >>> items.gs.query(scene)["id"].tolist()
            ['s2_20240308', 'label']
            >>> items.gs.query(field, when="2024-03")["id"].tolist()
            ['s2_20240308', 's2_20240328', 'label']
        """
        from .query import select

        return select(self._data, where, when=when, predicate=predicate)

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
            >>> scene.gs.plot() * plots.to_crs(anchor.crs).gs.plot()
        """
        from geosave.geodata.viz.plot import draw_vector

        return draw_vector(self._data, color=color, alpha=alpha)

    def explore(
        self,
        *,
        column: str | None = None,
        overlay: Overlay | None = None,
        basemap: str | None = None,
        color: str = "tab:blue",
        alpha: float = 0.3,
    ) -> Explorer:
        """Open this vector on a web map with a sidebar to explore it.

        Features have hover and click identification. Property and date filters
        narrow the table and map. Checking catalog rows opens their rasters;
        unchecking removes them.

        Args:
            column: Column that colours the features, or None for one colour.
            overlay: Rasters, GeoDataFrames or item tables drawn on top; a
                mapping names each layer.
            basemap: Basemap under the layers, as `viz.map.BASEMAPS` names it.
                None uses the light basemap.
            color: Colour of every feature when no column colours them.
            alpha: Opacity of polygon fill; 0 draws boundaries only.

        Returns:
            Explorer, shown by a notebook or served with `.servable()`.

        Raises:
            KeyError: `column` is not a column of the frame.
            ValueError: The vector has no CRS, holds no geometry, or holds a
                GeometryCollection.

        Examples:
            >>> fields.gs.explore(column="crop")
            >>> fields.gs.explore(alpha=0.0, basemap="Imagery")  # boundaries only
        """
        from geosave.geodata.viz.explorer import Explorer, create_layer, create_layers

        layer = create_layer(self._data, column=column, color=color, alpha=alpha)
        return Explorer(layer, *create_layers(overlay), basemap=basemap)

    def rasterize(
        self,
        like: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree,
        *,
        column: str | None = None,
        fill: int | float | bool = 0,
        dtype: DTypeLike | None = None,
        all_touched: bool = False,
    ) -> xr.DataArray:
        """Burn geometries or one property onto an existing exact grid.

        Args:
            like: Anchor or geolocated xarray object supplying the target grid.
            column: Property to burn. None returns a boolean presence mask.
            fill: Value outside geometries. A presence mask takes only 0 or
                False, since it is False outside them.
            dtype: Output dtype for property values.
            all_touched: Burn every touched pixel instead of pixel centres.

        Returns:
            Eager geolocated array on the exact target grid. A dated anchor
            dates it with a scalar `time` at the start of its timespan.

        Raises:
            KeyError: The selected property is absent.
            ValueError: A geometry is null, empty, or invalid, values or fill
                cannot be represented by the dtype, or `column` is None with a
                truthy `fill`.
        """
        from geosave.geodata.transform.vector import rasterize

        return rasterize(
            self._data,
            like,
            column=column,
            fill=fill,
            dtype=dtype,
            all_touched=all_touched,
        )

    @classmethod
    def from_geometry(
        cls,
        geometry: SomeGeometry,
        *,
        crs: SomeCRS | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> GeoDataFrame:
        """Build a one-feature frame from a common geometry representation.

        Args:
            geometry: GeoJSON mapping, WKT string, Shapely geometry, or ODC
                geometry.
            crs: CRS of the supplied coordinates. None uses an ODC geometry's
                own CRS, or WGS84 for other representations.
            properties: Scalar columns stored beside the geometry.

        Returns:
            One-feature frame carrying the supplied property columns.

        Raises:
            ValueError: The geometry cannot be parsed, carries a CRS different
                from `crs`, or a property is named `geometry`.
        """
        properties = properties or {}
        if "geometry" in properties:
            raise ValueError(
                "geometry is the frame's own column; do not pass it as a property"
            )
        geometry_crs = geometry.crs if isinstance(geometry, Geometry) else None
        if geometry_crs is not None and crs is not None and OdcCRS(crs) != geometry_crs:
            raise ValueError(
                f"geometry is in {geometry_crs} but crs= names {crs}; "
                "transform the geometry or provide its actual CRS"
            )
        frame = gpd.GeoDataFrame(
            {name: [value] for name, value in properties.items()},
            geometry=[to_shapely(geometry)],
            crs=geometry_crs or crs or "EPSG:4326",
        )
        return cast("GeoDataFrame", frame)
