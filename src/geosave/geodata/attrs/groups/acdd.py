"""ACDD discovery attributes: what a dataset is, and the place it declares."""

from __future__ import annotations

from typing import Self

import numpy as np
from odc.geo.geom import Geometry
from pydantic import Field, model_validator
from pyproj import CRS
from shapely import from_wkt, to_wkt
from shapely.errors import GEOSException

from .base import AttrsGroup


class ACDD(AttrsGroup):
    """Hold a whole dataset's ACDD keys, the ones catalogs search on.

    It lives on a Dataset's root attrs, where a STAC Collection's fields land
    when a source loads it.

    Attributes:
        id: Identifier of the dataset, a STAC Collection's id.
        title: Short human-readable name.
        summary: Paragraph about the data, a Collection's description.
        keywords: Comma-separated keywords.
        institution: Organization that produced the data.
        license: License of the data, an SPDX identifier where one applies.
        time_coverage_start: ISO 8601 start of the period the data covers.
        time_coverage_end: ISO 8601 end of that period; equal to the start
            for a single instant.

    Raises:
        ValueError: A coverage bound is not an ISO 8601 instant, or the end
            comes before the start.

    Examples:
        >>> ds.gs.acdd.title
        'Sentinel-2 Level-2A'
    """

    id: str | None = None
    title: str | None = None
    summary: str | None = None
    keywords: str | None = None
    institution: str | None = None
    license: str | None = None
    time_coverage_start: str | None = None
    time_coverage_end: str | None = None

    @model_validator(mode="after")
    def _check_time_coverage_is_ordered(self) -> Self:
        """Require ISO 8601 coverage bounds with the end not before the start."""
        bounds = {
            "time_coverage_start": self.time_coverage_start,
            "time_coverage_end": self.time_coverage_end,
        }
        instants = {}
        for key, text in bounds.items():
            if text is None:
                continue
            try:
                instant = np.datetime64(text)
            except ValueError:
                instant = np.datetime64("NaT")
            if np.isnat(instant):
                raise ValueError(
                    f"{key} {text!r} is not an ISO 8601 date or time. Write it "
                    f"like '2024-03-05' or '2024-03-05T06:07:08'."
                )
            instants[key] = instant
        if (
            len(instants) == 2
            and instants["time_coverage_end"] < instants["time_coverage_start"]
        ):
            raise ValueError(
                f"time_coverage_end {self.time_coverage_end!r} is before "
                f"time_coverage_start {self.time_coverage_start!r}. Swap them, "
                f"or give the same value for a single instant."
            )
        return self


class ACDDExtent(AttrsGroup):
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
        wkt = to_wkt(geometry.geom, rounding_precision=-1)
        return cls.model_validate(
            {"geospatial_bounds": wkt, "geospatial_bounds_crs": crs}
        )

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
