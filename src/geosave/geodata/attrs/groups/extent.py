"""The place a raster declares it covers, independently of its pixel grid."""

from __future__ import annotations

from typing import Self

from odc.geo.geom import Geometry
from pydantic import Field, model_validator
from shapely import from_wkt, to_wkt
from shapely.errors import GEOSException

from .base import AttrsGroup


class Extent(AttrsGroup):
    """Preserve the place a raster declares it covers, independently of its pixel grid.

    A crop to a polygon keeps the polygon here while the pixels keep their
    rectangular grid. It lives on a Dataset's and its spatial variables' attrs.

    Attributes:
        wkt: Nonempty, valid two-dimensional geometry in x/y order.
        crs: CRS of the WKT coordinates.

    Examples:
        >>> attrs.Extent.from_geometry(city).to_attrs()
        {'extent': 'POLYGON ((13 52, 14 52, 13 53, 13 52))', 'extent_crs': 'EPSG:4326'}
    """

    wkt: str = Field(alias="extent")
    crs: str = Field(alias="extent_crs")

    @model_validator(mode="after")
    def _validate_extent(self) -> Self:
        geometry = self.to_geometry()
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("extent must be nonempty and valid")
        return self

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> Self:
        """Encode a CRS-aware ODC geometry.

        Args:
            geometry: Declared extent, independent of pixel placement.

        Returns:
            Validated WKT and CRS group.

        Raises:
            ValueError: Geometry lacks a CRS or is not valid, nonempty and 2D.
        """
        if geometry.crs is None:
            raise ValueError("extent needs a CRS")
        if geometry.geom.has_z or geometry.geom.has_m:
            raise ValueError("extent must be two-dimensional")
        wkt = to_wkt(geometry.geom, rounding_precision=-1)
        return cls.model_validate({"extent": wkt, "extent_crs": str(geometry.crs)})

    def to_geometry(self) -> Geometry:
        """Return the extent as a CRS-aware ODC geometry."""
        try:
            geometry = Geometry(from_wkt(self.wkt), self.crs)
        except GEOSException as error:
            raise ValueError("extent must be valid WKT") from error
        if geometry.geom.has_z or geometry.geom.has_m:
            raise ValueError("extent must be two-dimensional")
        return geometry

    def clip(self, geometry: Geometry) -> Self | None:
        """Narrow this extent to a geometry, as cropping narrows coverage.

        Args:
            geometry: CRS-aware geometry the pixels now cover at most.

        Returns:
            Extent of the intersection, in this extent's CRS, or None when
            the two do not meet.
        """
        clipped = self.to_geometry().intersection(geometry.to_crs(self.crs))
        if clipped.is_empty:
            return None
        return self.from_geometry(clipped)

    def to_attrs(self) -> dict[str, str]:
        """Write the WKT and its CRS."""
        return {"extent": self.wkt, "extent_crs": self.crs}
