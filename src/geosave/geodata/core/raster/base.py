"""Properties, pixel operations and summaries shared by every `gs` raster accessor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from geosave.geodata.conventions import END_DATETIME, START_DATETIME, TIME_COORDINATE

if TYPE_CHECKING:
    from collections.abc import Callable

    from odc.geo import CRS, BoundingBox, Resolution
    from odc.geo.gcp import GCPGeoBox
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
        """Read the unambiguous CF temporal coordinate, falling back to `time`.

        Raises:
            ValueError: Multiple coordinates declare temporal identity.
        """
        names = set(self._data.cf.coordinates.get("time", []))
        names.update(self._data.cf.axes.get("T", []))
        if len(names) > 1:
            raise ValueError(f"Multiple temporal coordinates are declared: {names}")
        if names:
            return self._data.coords[names.pop()]
        return self._data.coords.get(TIME_COORDINATE)

    def write_time_bounds(
        self,
        freq: str,
        *,
        label: Literal["left", "right"] = "left",
        dim: str | None = None,
    ) -> DataT:
        """State each time label's cell as one frequency step, without reading pixels.

        Args:
            freq: Pandas offset alias the cells span, such as "MS".
            label: "left" when each label starts its cell, "right" when it
                ends it.
            dim: Coordinate holding the labels; defaults to the temporal
                coordinate.

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
        values = pd.DatetimeIndex(np.atleast_1d(labels.values))
        # Fixed-length steps such as "10D" accept every instant as an edge.
        for value in values:
            if not step.is_on_offset(value):
                raise ValueError(f"time label {value} is not on a {freq} edge")
        if label == "left":
            starts, ends = values, values + step
        else:
            starts, ends = values - step, values
        starts = starts.to_numpy().reshape(labels.shape)
        ends = ends.to_numpy().reshape(labels.shape)
        return self._data.assign_coords(
            {START_DATETIME: (labels.dims, starts), END_DATETIME: (labels.dims, ends)}
        )

    @property
    def time_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return each label's start and end as 1-D arrays, None for instants.

        A scalar period gives arrays of length one. NaT marks a label that
        covers no observation.

        Raises:
            ValueError: Bounds are incomplete, reversed, or on unrelated dimensions.
        """
        coords = self._data.coords
        if START_DATETIME not in coords and END_DATETIME not in coords:
            return None
        if START_DATETIME not in coords or END_DATETIME not in coords:
            raise ValueError(
                "a period needs both start_datetime and end_datetime "
                "coordinates; add the missing one or drop both"
            )
        start = coords[START_DATETIME]
        end = coords[END_DATETIME]
        labels = self.time_coordinate
        for endpoint in (start, end):
            if endpoint.ndim and (
                labels is None or endpoint.dims != labels.dims or labels.ndim != 1
            ):
                raise ValueError(
                    "observation endpoints must follow the temporal coordinate's dimension"
                )

        starts = np.atleast_1d(start.values)
        ends = np.atleast_1d(end.values)
        starts, ends = np.broadcast_arrays(starts, ends)
        if np.any(np.isnat(starts) != np.isnat(ends)) or np.any(starts > ends):
            raise ValueError("time bounds must be ordered pairs or paired NaT")
        return starts, ends

    def _read_coverage(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Read observation coverage for whole-object summaries."""
        return self.time_bounds

    @property
    def timespan(self) -> DateRange | None:
        """Return inclusive coverage from bounds, falling back to time labels.

        Empty time axes, absent temporal information and NaT coverage return
        None. Only temporal coordinates are read, never raster pixels.
        """
        labels = self.time_coordinate
        if labels is not None and labels.size == 0:
            return None

        bounds = self._read_coverage()
        if bounds is None:
            if labels is None:
                return None
            starts = ends = np.atleast_1d(labels.values)
        else:
            starts, ends = bounds

        if starts.size == 0 or ends.size == 0:
            return None

        present = ~np.isnat(starts) & ~np.isnat(ends)
        if not present.any():
            return None
        first, last = starts[present].min(), ends[present].max()

        # Microseconds convert to datetime; nanoseconds would convert to int.
        return first.astype("datetime64[us]").item(), last.astype(
            "datetime64[us]"
        ).item()


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

    @abstractmethod
    def validate(self, *, georeferenced: bool = False) -> None:
        """Check raster validity and optionally require geographic placement."""

    @property
    @abstractmethod
    def geobox(self) -> GeoBox | GCPGeoBox | None:
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
    def grid_mapping_coords(self) -> tuple[str, ...]:
        """Name every scalar coordinate that carries a CRS, as CF grid mappings do.

        A raster normally carries one; one written by several tools can carry
        stale extras, which placing it on a new grid must drop.

        Examples:
            >>> scene.gs.grid_mapping_coords
            ('spatial_ref',)
        """
        markers = {"crs_wkt", "spatial_ref", "grid_mapping_name"}
        names = []
        for name, coord in self._data.coords.items():
            if not coord.dims and not markers.isdisjoint(coord.attrs):
                names.append(str(name))
        return tuple(names)

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

    @abstractmethod
    def _map_variables(self, change: Callable[[xr.DataArray], xr.DataArray]) -> DataT:
        """Apply one per-variable change to every data variable this object holds."""

    def mask_and_scale(self) -> DataT:
        """Decode CF scale, offset and missing values using xarray.

        Returns:
            Same kind of object with decoded values and native per-variable
            storage encoding. Already decoded variables pass through. Lazy
            arrays remain lazy; coordinates and descriptive attrs are retained.
        """
        from .transform import packing

        return self._map_variables(packing.mask_and_scale)

    def scale_and_fill(self) -> DataT:
        """Restore stored values from each variable's CF encoding without I/O.

        Returns:
            Same kind of object with stored pixels and CF attrs. Consumed
            conversion encoding is removed; unrelated encoding is retained.
            Variables without conversion instructions pass through.

        Raises:
            ValueError: Attrs and encoding declare conflicting packing, or
                encoded integer values overflow or collide with missing codes.
                For lazy arrays, pixel checks run when evaluated.
        """
        from .transform import packing

        return self._map_variables(packing.scale_and_fill)
