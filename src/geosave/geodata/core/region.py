"""Geographic location and temporal coverage, independent of pixel grids."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from odc.geo.crs import CRS
from odc.geo.geom import BoundingBox, Geometry, point

from geosave.geodata.utils.crs import validate_wgs84_bbox, validate_wgs84_coordinate
from geosave.geodata.utils.datetime import RegionDatetime, DateRange, parse_daterange
from geosave.geodata.utils.geometry import SomeGeometry

if TYPE_CHECKING:
    import geopandas as gpd
    from odc.geo import SomeCRS
    from odc.geo.geobox import GeoBox
    from geosave.geodata.utils.geocode import Place


@dataclass(frozen=True, eq=False)
class GeoRegion:
    """Describe a CRS-aware location and optional temporal coverage.

    Args:
        extent: Nonempty, valid ODC geometry with a CRS. Points and polygons
            retain their shape; no pixel grid is inferred.
        timespan: Inclusive pair of start/end datetimes, or None for a timeless
            location. The factory methods also accept date strings and intervals.
    """

    extent: Geometry
    timespan: DateRange | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.extent, Geometry):
            raise TypeError("extent must be an odc.geo Geometry; use from_geometry")
        if self.extent.crs is None:
            raise ValueError("region extent needs a CRS")
        if self.extent.is_empty or not self.extent.is_valid:
            raise ValueError("region extent must be nonempty and valid")
        if self.timespan is not None:
            object.__setattr__(self, "timespan", parse_daterange(self.timespan))

    @property
    def crs(self) -> CRS:
        """Return the geometry's coordinate reference system."""
        assert self.extent.crs is not None
        return self.extent.crs

    @property
    def lonlat(self) -> tuple[float, float]:
        """Return the centroid in WGS84 longitude/latitude order."""
        longitude, latitude = self.extent.centroid.to_crs("EPSG:4326").coords[0]
        return float(longitude), float(latitude)

    def locate(self) -> Place | None:
        """Reverse geocode the centroid through Nominatim.

        May make a network request; results are cached by rounded coordinates.

        Returns:
            Place with the available address fields, or None when the service
            has no result or cannot be reached.
        """
        from geosave.geodata.utils.geocode import Place

        longitude, latitude = self.lonlat
        return Place.from_coordinate(latitude, longitude)

    def replace(self, *, timespan: RegionDatetime | None) -> Self:
        """Return a new region with the same geometry and another timespan.

        Args:
            timespan: ISO or compact date string (such as "2025-06" or
                "20250601"), a "start/end" interval, a pair of date strings,
                or a pair of datetimes. Partial dates cover their entire period;
                string range endpoints are inclusive. None makes the new region
                timeless.

        Returns:
            New region; the original is unchanged.
        """
        return dataclasses.replace(
            self, timespan=None if timespan is None else parse_daterange(timespan)
        )

    def format(self, template: str, *, geobox: GeoBox | None = None) -> str:
        """Format location and time fields with a Python format string.

        Args:
            template: Text containing named replacement fields:
                `lon` and `lat` are centroid coordinates in WGS84 degrees;
                `epsg` is the integer CRS code, or None if unavailable;
                `width` and `height` are geometry bounds spans in the geometry's
                CRS units, not pixel counts. `start` and `end` are inclusive
                datetimes, available only when the region has a timespan.
                Use numeric formats such as `{lon:.4f}`, datetime formats such
                as `{start:%Y%m%d}`, and `{{` or `}}` for literal braces.
            geobox: Optional grid supplying `epsg`, `width`, and `height`
                instead of the region extent, plus `res` (absolute x-axis
                resolution). Spans and resolution use the grid's CRS units.
                The region still supplies `lon`, `lat`, `start`, and `end`.
                Without a grid, `res` is unavailable.

        Returns:
            Text with the fields substituted.

        Raises:
            ValueError: A field is unavailable, the template is malformed, or
                a format specification is invalid for its value, or the supplied
                grid has no CRS.

        Examples:
            >>> region = GeoRegion.from_coordinates(52, 13, timespan="2025-06")
            >>> region.format("{lat:.2f}_{lon:.2f}_{start:%Y%m%d}")
            '52.00_13.00_20250601'
        """
        longitude, latitude = self.lonlat
        bounds = self.extent.boundingbox
        fields: dict[str, object] = {
            "lon": longitude,
            "lat": latitude,
            "epsg": self.crs.epsg,
            "width": bounds.span_x,
            "height": bounds.span_y,
        }
        if geobox is not None:
            if geobox.crs is None:
                raise ValueError("grid carries no CRS")
            fields.update(
                epsg=geobox.crs.epsg,
                width=geobox.boundingbox.span_x,
                height=geobox.boundingbox.span_y,
                res=abs(geobox.resolution.x),
            )
        if self.timespan is not None:
            fields["start"], fields["end"] = self.timespan
        try:
            return template.format(**fields)
        except KeyError as error:
            raise ValueError(
                f"region has no {error.args[0]!r} to format; it offers {sorted(fields)}"
            ) from None

    @classmethod
    def from_bbox(
        cls,
        bbox: BoundingBox | tuple[float, float, float, float],
        *,
        timespan: RegionDatetime | None = None,
    ) -> Self:
        """Create a region from bounding coordinates.

        Args:
            bbox: ODC BoundingBox in its own CRS, or a WGS84
                `(west, south, east, north)` tuple in degrees. A BoundingBox
                without a CRS is also interpreted as WGS84.
            timespan: ISO or compact date string, "start/end" interval, pair
                of date strings, or pair of datetimes. Partial dates cover
                their whole period, including both endpoints. None is timeless.

        Returns:
            Region whose geometry is the bounding polygon.

        Raises:
            ValueError: WGS84 bounds are out of range, south exceeds north,
                the resulting geometry is invalid or empty, or a date string
                cannot be parsed.
        """
        if not isinstance(bbox, BoundingBox) or bbox.crs is None:
            validate_wgs84_bbox(tuple(bbox))
            bbox = BoundingBox(*bbox, crs="EPSG:4326")
        return cls.from_geometry(bbox.polygon, timespan=timespan)

    @classmethod
    def from_geometry(
        cls,
        geometry: SomeGeometry | gpd.GeoDataFrame | gpd.GeoSeries,
        *,
        crs: SomeCRS | None = None,
        timespan: RegionDatetime | None = None,
    ) -> Self:
        """Preserve a geometry or union a frame's features, without creating a grid.

        Args:
            geometry: ODC geometry, GeoDataFrame/GeoSeries, GeoJSON geometry
                mapping, WKT string, or Shapely geometry. Frames and series
                must carry a CRS and contain valid, nonempty geometries without
                missing values; their features are unioned.
            crs: CRS of bare geometry inputs (WGS84 when omitted). For inputs
                carrying a CRS, a different CRS must be applied explicitly first.
            timespan: ISO or compact date string, "start/end" interval, pair
                of date strings, or pair of datetimes. Partial dates cover
                their whole period, including both endpoints. None is timeless.

        Returns:
            Region preserving the geometry's shape and CRS.

        Raises:
            TypeError: The geometry representation is unsupported.
            ValueError: Geometry is invalid or empty, a frame or series lacks
                a CRS, an explicit CRS conflicts with the existing CRS, or a
                date string cannot be parsed.
        """
        import geopandas as gpd
        from geosave.geodata.utils.geometry import to_shapely

        if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            shapes = geometry.geometry
            if geometry.crs is None:
                raise ValueError("geometry frame needs a CRS")
            if (
                shapes.empty
                or shapes.isna().any()
                or shapes.is_empty.any()
                or not shapes.is_valid.all()
            ):
                raise ValueError("region extent must be nonempty and valid")
            geometry = Geometry(shapes.union_all(), geometry.crs)
        if isinstance(geometry, Geometry):
            if crs is not None and geometry.crs != CRS(crs):
                raise ValueError(
                    "geometry already has a different CRS; reproject explicitly"
                )
        else:
            geometry = Geometry(to_shapely(geometry), crs or "EPSG:4326")
        return cls(geometry, None if timespan is None else parse_daterange(timespan))

    @classmethod
    def from_coordinates(
        cls,
        latitude: float,
        longitude: float,
        *,
        timespan: RegionDatetime | None = None,
    ) -> Self:
        """Create a WGS84 point region without defining a raster window.

        Args:
            latitude: Latitude in degrees, in [-90, 90].
            longitude: Longitude in degrees, wrapped into [-180, 180).
            timespan: ISO or compact date string, "start/end" interval, pair
                of date strings, or pair of datetimes. Partial dates cover
                their whole period, including both endpoints. None is timeless.

        Returns:
            Point region in EPSG:4326. Its `lonlat` property returns longitude
            first, unlike this method's latitude-first arguments.

        Raises:
            ValueError: Latitude is outside [-90, 90], the point is invalid,
                or a date string cannot be parsed.
        """
        latitude, longitude = validate_wgs84_coordinate(latitude, longitude)
        return cls.from_geometry(
            point(longitude, latitude, "EPSG:4326"), timespan=timespan
        )
