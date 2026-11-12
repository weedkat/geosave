"""Properties, pixel operations and summaries shared by every `gs` raster accessor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from geosave.geodata.conventions import END_DATETIME, START_DATETIME, TIME_COORDINATE

if TYPE_CHECKING:
    from affine import Affine
    from odc.geo import CRS, BoundingBox, Resolution, SomeCRS, SomeResolution
    from odc.geo.geobox import GeoBox

    from geosave.geodata.utils.datetime import DateRange

    from geosave.geodata.core.region import GeoRegion


class RasterTime[DataT: xr.DataArray | xr.Dataset]:
    """Read and write temporal coordinates on one native raster.

    A label that covers a period carries it as `start_datetime` and
    `end_datetime` coordinates, the names STAC Items use. They follow the
    temporal dimension, or are scalar endpoints shared by every observation.
    """

    _data: DataT

    @property
    def time_coordinate(self) -> xr.DataArray | None:
        """Read the `time` coordinate, the name ODC and odc-stac give it."""
        return self._data.coords.get(TIME_COORDINATE)

    def write_time_bounds(
        self,
        freq: str,
        *,
        dim: str | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | datetime = "start_day",
        offset: str | timedelta | None = None,
    ) -> DataT:
        """Write bounds for existing resample labels without reading pixels.

        Coordinates must already hold bin labels produced with this sampling
        configuration, rather than raw observation timestamps.

        Args:
            freq: Pandas offset alias the cells span, such as "MS".
            dim: Coordinate holding the labels; defaults to the temporal
                coordinate.
            closed: Inclusive bin edge, following native resample defaults.
            label: Bin label edge, following native resample defaults.
            origin: Origin used for resampling.
            offset: Shift applied to the resample origin. Existing labels
                already include this shift.

        Returns:
            New raster with `start_datetime` and `end_datetime` coordinates on
            the temporal coordinate's dimensions; scalar ones for a scalar label.

        Raises:
            KeyError: No time coordinate exists.
            ValueError: A label is not on an edge of `freq`.

        Examples:
            >>> monthly.gs.write_time_bounds("MS").end_datetime.values[0]
            numpy.datetime64('2025-02-01T00:00:00.000000000')
        """
        from pandas.tseries.frequencies import to_offset

        labels = self.time_coordinate if dim is None else self._data.coords[dim]
        if labels is None:
            raise KeyError("No temporal coordinate is declared")
        step = to_offset(freq)
        label_edge = pd.Grouper(
            freq=freq, closed=closed, label=label, origin=origin, offset=offset
        ).label
        values = pd.DatetimeIndex(np.atleast_1d(labels.values))
        # Fixed-length steps such as "10D" accept every instant as an edge.
        for value in values:
            if not step.is_on_offset(value):
                raise ValueError(f"time label {value} is not on a {freq} edge")
        if label_edge == "left":
            starts, ends = values, values + step
        else:
            starts, ends = values - step, values
        starts = starts.to_numpy().reshape(labels.shape)
        ends = ends.to_numpy().reshape(labels.shape)
        return self._data.assign_coords(
            {START_DATETIME: (labels.dims, starts), END_DATETIME: (labels.dims, ends)}
        )

    @property
    def timespan(self) -> DateRange | None:
        """Return the first start and last end the cells state, else the first and last label.

        Returns None without time or for an empty time axis. Reads coordinates
        only, never raster pixels.

        Examples:
            >>> monthly.gs.write_time_bounds("MS").gs.timespan
            (datetime.datetime(2025, 1, 1, 0, 0), datetime.datetime(2025, 3, 1, 0, 0))
        """
        coords = self._data.coords
        if START_DATETIME in coords:
            starts = coords[START_DATETIME].values
            ends = coords[END_DATETIME].values
        else:
            labels = self.time_coordinate
            if labels is None:
                return None
            starts = ends = labels.values
        first = pd.DatetimeIndex(np.atleast_1d(starts)).min()
        last = pd.DatetimeIndex(np.atleast_1d(ends)).max()
        if pd.isna(first) or pd.isna(last):
            return None
        return first.to_pydatetime(), last.to_pydatetime()


class GeoRaster[DataT: xr.Dataset | xr.DataArray | xr.DataTree](ABC):
    """Spatial, attrs, and pixel members every `gs` raster accessor shares.

    Concrete accessors bind their xarray object to `_data`; this class is
    never built directly. Concrete accessors own pixel operations and traversal.
    """

    _data: DataT

    def __init__(self, data: DataT) -> None:
        """Bind the xarray object.

        Args:
            data: Band, raster, or tree to read through this accessor.
        """
        self._data = data

    @property
    @abstractmethod
    def spatial_dims(self) -> tuple[str, str] | None:
        """Return spatial axes in vertical, horizontal order when all rasters agree."""

    @property
    def ydim(self) -> str | None:
        """Name the vertical spatial dimension, as `.odc.ydim` does."""
        dims = self.spatial_dims
        return None if dims is None else dims[0]

    @property
    def xdim(self) -> str | None:
        """Name the horizontal spatial dimension, as `.odc.xdim` does."""
        dims = self.spatial_dims
        return None if dims is None else dims[1]

    def _require_spatial_dims(self) -> tuple[str, str]:
        """Return the spatial dimensions pixel operations need, placed or not."""
        from geosave.geodata.errors import RasterValidationError

        dims = self.spatial_dims
        if dims is None:
            raise RasterValidationError(
                "ODC finds no spatial dimensions. Name them as ODC does, "
                "y/x or latitude/longitude."
            )
        return dims

    def validate(self) -> None:
        """Require a grid ODC recognizes, with a CRS, without reading pixels.

        Raises:
            RasterValidationError: ODC finds no grid, or the grid has no CRS.
            NotImplementedError: Pixels are placed by ground control points.

        Examples:
            >>> scene.gs.validate()
        """
        from geosave.geodata.errors import RasterValidationError

        geobox = self.geobox
        if geobox is None:
            raise RasterValidationError(
                "ODC finds no grid. Name the spatial dimensions as ODC does, "
                "y/x or latitude/longitude, with their coordinates."
            )
        if geobox.crs is None:
            raise RasterValidationError(
                "The coordinate reference system (CRS) is missing. Declare the "
                "CRS of the existing coordinates with raster.odc.assign_crs(crs)."
            )

    @property
    @abstractmethod
    def geobox(self) -> GeoBox | None:
        """Return the common grid, or None for missing or differing placement."""

    @property
    def is_georeferenced(self) -> bool:
        """Whether this object carries a locatable grid.

        Returns:
            True when `geobox` resolves to a grid with a CRS. A tree with
            differing grids has no shared grid and returns False.
        """
        grid = self.geobox
        return grid is not None and grid.crs is not None

    @property
    def crs(self) -> CRS | None:
        """Read the coordinate reference system, if this object carries one.

        Returns:
            CRS the grid mapping coordinate names, or None if this object
            carries no locatable grid or the grid itself carries no CRS.
        """
        geobox = self.geobox
        return None if geobox is None else geobox.crs

    @property
    def crs_name(self) -> str | None:
        """Name the coordinate reference system briefly, for a reader.

        A CRS read back off `spatial_ref` prints its whole WKT, and odc's own
        `crs_str` prints whatever the CRS was built from, so neither is short
        enough to name one in a message.

        Returns:
            Its EPSG code as `"EPSG:32633"`, the projection's own name where
            it carries no code, or None if this object carries no CRS.

        Examples:
            >>> scene.gs.crs_name
            'EPSG:32633'
        """
        crs = self.crs
        if crs is None:
            return None
        return f"EPSG:{crs.epsg}" if crs.epsg else crs.proj.name

    @property
    def bounds(self) -> BoundingBox | None:
        """Read the grid's extent in its own CRS, if this object carries one.

        Returns:
            Bounding box covering every pixel, or None if this object
            carries no locatable grid.
        """
        geobox = self.geobox
        return None if geobox is None else geobox.boundingbox

    @property
    def resolution(self) -> Resolution | None:
        """Read the pixel size in CRS units, if this object carries one.

        Returns:
            Signed resolution along x and y, or None if this object carries
            no locatable grid.
        """
        geobox = self.geobox
        return None if geobox is None else geobox.resolution

    @property
    def transform(self) -> Affine | None:
        """Read the pixel-to-world affine transform, as `.odc.transform` does."""
        geobox = self.geobox
        return None if geobox is None else geobox.transform

    @property
    def aspect(self) -> float | None:
        """Read the width-to-height ratio in pixels, as `.odc.aspect` does."""
        geobox = self.geobox
        return None if geobox is None else geobox.aspect

    def map_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Read south-west and north-east corners in lat/lon, as `.odc.map_bounds` does.

        Returns:
            ``((lat_s, lon_w), (lat_n, lon_e))``, the order folium and
            ipyleaflet take.

        Raises:
            ValueError: This object carries no grid.

        Examples:
            >>> scene.gs.map_bounds()
            ((45.0, 13.0), (45.1, 13.1))
        """
        geobox = self.geobox
        if geobox is None:
            raise ValueError("Not geo registered")
        return geobox.map_bounds()

    def output_geobox(
        self,
        crs: SomeCRS,
        *,
        resolution: SomeResolution | Literal["auto", "fit", "same"] = "auto",
    ) -> GeoBox:
        """Compute the grid this raster lands on in another CRS, as `.odc.output_geobox` does.

        Args:
            crs: Destination CRS.
            resolution: Destination pixel size, or ODC's ``"auto"``, ``"fit"``
                or ``"same"``.

        Returns:
            Grid covering this raster's footprint, to pass to `reproject`.

        Raises:
            ValueError: This object carries no grid.

        Examples:
            >>> scene.gs.output_geobox("EPSG:3857", resolution=10).crs.epsg
            3857
        """
        from odc.geo.overlap import compute_output_geobox

        geobox = self.geobox
        if geobox is None:
            raise ValueError("Not geo registered")
        return compute_output_geobox(geobox, crs, resolution=resolution)

    @property
    @abstractmethod
    def timespan(self) -> DateRange | None:
        """Read inclusive temporal coverage using the concrete raster's rules."""

    @property
    @abstractmethod
    def time_coordinate(self) -> xr.DataArray | None:
        """Read the temporal coordinate owned by this raster or tree node."""

    @property
    def times(self) -> pd.DatetimeIndex | None:
        """Read the time coordinate labels.

        Returns:
            Labels in axis order, or None for timeless data.

        Examples:
            >>> ds.gs.times.strftime("%Y%m%d").tolist()
            ['20250601', '20250611']
        """
        labels = self.time_coordinate
        if labels is None:
            return None
        return pd.DatetimeIndex(np.atleast_1d(labels.values))

    @property
    @abstractmethod
    def region(self) -> GeoRegion:
        """Read spatial extent and temporal coverage using the concrete raster's rules."""
