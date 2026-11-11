"""Declared spatial extent, stored using ACDD WKT and CRS attributes."""

from __future__ import annotations

from typing import Self

from odc.geo.geom import Geometry
from pydantic import Field, model_validator
from pyproj import CRS
from shapely import from_wkt, to_wkt
from shapely.errors import GEOSException

from .base import AttrsGroup


class Extent(AttrsGroup):
    """Preserve a declared place independently of the raster's pixel grid.

    Attributes:
        wkt: Nonempty, valid two-dimensional geometry in CRS axis order.
        crs: CRS of the WKT coordinates. ACDD defaults to EPSG:4326, whose
            axis order is latitude/longitude, unlike ODC's x/y geometries.
    """

    wkt: str = Field(alias="geospatial_bounds")
    crs: str = Field(default="EPSG:4326", alias="geospatial_bounds_crs")

    @model_validator(mode="after")
    def _validate_extent(self) -> Self:
        geometry = self.to_geometry()
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("extent must be nonempty and valid")
        return self

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> Self:
        """Encode a CRS-aware ODC geometry in ACDD coordinate order.

        Args:
            geometry: Declared spatial extent, independent of pixel placement.

        Returns:
            Validated WKT and CRS group.

        Raises:
            ValueError: Geometry lacks a CRS or is not valid, nonempty and 2D.
        """
        if geometry.crs is None:
            raise ValueError("extent needs a CRS")
        if geometry.geom.has_z or geometry.geom.has_m:
            raise ValueError("extent must be two-dimensional")
        crs = str(geometry.crs)
        if cls._swap_axes(crs):
            geometry = geometry.transform(lambda x, y: (y, x))
        return cls(wkt=to_wkt(geometry.geom, rounding_precision=-1), crs=crs)

    def to_geometry(self) -> Geometry:
        """Return the extent as a CRS-aware ODC geometry in x/y order."""
        try:
            geometry = Geometry(from_wkt(self.wkt), self.crs)
        except GEOSException as error:
            raise ValueError("extent must be valid WKT") from error
        if geometry.geom.has_z or geometry.geom.has_m:
            raise ValueError("extent must be two-dimensional")
        if self._swap_axes(self.crs):
            geometry = geometry.transform(lambda x, y: (y, x))
        return geometry

    def clip(self, geometry: Geometry) -> Self | None:
        """Narrow this extent to a geometry, as cropping narrows coverage.

        Args:
            geometry: CRS-aware geometry the pixels now cover at most.

        Returns:
            Extent of the intersection, in this extent's CRS, or None when
            the two do not meet.
        """
        area = self.to_geometry()
        clipped = area.intersection(geometry.to_crs(self.crs))
        if clipped.is_empty:
            return None
        return self.from_geometry(clipped)

    def to_attrs(self) -> dict[str, str]:
        """Write WKT and an explicit CRS, including the ACDD default."""
        return {"geospatial_bounds": self.wkt, "geospatial_bounds_crs": self.crs}

    @staticmethod
    def _swap_axes(crs: str) -> bool:
        axes = CRS(crs).axis_info
        if len(axes) != 2:
            raise ValueError("extent CRS must have two axes")
        return axes[0].direction == "north" and axes[1].direction == "east"
