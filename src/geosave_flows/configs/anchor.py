"""Serializable inputs that open native geospatial anchors."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, PositiveInt, model_validator

from geosave.geodata.core import GeoAnchor
from geosave.geodata import io

from .base import ConfigModel


class CoordinateAnchorConfig(ConfigModel):
    """Build a grid around one WGS84 coordinate.

    Attributes:
        kind: Discriminator; must be `"coordinates"`.
        latitude: Grid centre latitude in degrees, from -90 to 90.
        longitude: Grid centre longitude in degrees, from -180 to 180.
        shape: Positive pixel count for a square, or `(height, width)`.
        resolution: Positive pixel size in the target CRS's units.
        crs: Target CRS. None selects the local UTM/UPS grid.
        timespan: Date, period, or start/end pair. None leaves the grid timeless.
    """

    kind: Literal["coordinates"]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    shape: PositiveInt | tuple[PositiveInt, PositiveInt]
    resolution: Annotated[float, Field(gt=0)]
    crs: str | None = None
    timespan: str | tuple[str, str] | None = None

    def open(self) -> GeoAnchor:
        """Build the configured native anchor.

        Returns:
            GeoAnchor with the configured pixel grid and temporal coverage.
        """
        return GeoAnchor.from_coordinates(
            **self.model_dump(exclude={"kind"}, exclude_none=True)
        )


class GeoJSONAnchorConfig(ConfigModel):
    """Build a grid covering one local GeoJSON file.

    Attributes:
        kind: Discriminator; must be `"geojson"`.
        path: Local vector file whose footprint defines the area.
        shape: Pixel count along the longest axis, or `(height, width)`.
            Supply exactly one of `shape` and `resolution`.
        resolution: Positive pixel size in target CRS units; alternative to shape.
        crs: Target CRS. None keeps a projected source CRS or chooses local
            UTM/UPS for a geographic source.
        timespan: Date, period, or start/end pair. None leaves the grid timeless.
        pad: Nonnegative distance added to each side in target CRS units.
    """

    kind: Literal["geojson"]
    path: Path
    shape: PositiveInt | tuple[PositiveInt, PositiveInt] | None = None
    resolution: Annotated[float, Field(gt=0)] | None = None
    crs: str | None = None
    timespan: str | tuple[str, str] | None = None
    pad: Annotated[float, Field(ge=0)] = 0

    @model_validator(mode="after")
    def require_grid_size(self) -> Self:
        """Require one grid sizing method.

        Returns:
            This configuration after validating its grid sizing fields.

        Raises:
            ValueError: Both or neither of `shape` and `resolution` are set.
        """
        if (self.shape is None) == (self.resolution is None):
            raise ValueError("GeoJSON anchor needs shape or resolution")
        return self

    def open(self) -> GeoAnchor:
        """Read the geometry and build its native anchor.

        Returns:
            GeoAnchor covering the file's footprint with configured padding,
            grid sizing, CRS, and temporal coverage.
        """
        return GeoAnchor.from_geometry(
            io.open_vector(self.path).gs.footprint,
            **self.model_dump(exclude={"kind", "path"}, exclude_none=True),
        )


class RasterAnchorConfig(ConfigModel):
    """Read the exact grid and time from one raster.

    Attributes:
        kind: Discriminator; must be `"raster"`.
        path: Raster path or URL accepted by `geosave.open_raster`.
    """

    kind: Literal["raster"]
    path: str | Path

    def open(self) -> GeoAnchor:
        """Read the raster's native anchor.

        Returns:
            GeoAnchor using the raster's existing grid and temporal coverage.
            The raster is closed after its metadata is read.
        """
        with io.open_raster(self.path) as raster:
            return raster.gs.anchor


AnchorConfig = Annotated[
    CoordinateAnchorConfig | GeoJSONAnchorConfig | RasterAnchorConfig,
    Field(discriminator="kind"),
]
