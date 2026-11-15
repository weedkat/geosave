"""Properties, pixel operations and summaries shared by every `gs` raster accessor."""

from __future__ import annotations

import math

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

import geosave.geodata.attrs as attrs
from geosave.geodata.core.conventions import (
    END_DATETIME,
    START_DATETIME,
    TIME_COORDINATE,
)
from geosave.geodata.errors import RasterValidationError

if TYPE_CHECKING:
    from affine import Affine
    from odc.geo import CRS, BoundingBox, Resolution, SomeCRS, SomeResolution
    from odc.geo.geobox import GeoBox

    from geosave.geodata.utils.datetime import DateRange, Freq

    from .utils.windows import Keep

    from geosave.geodata.core.region import GeoRegion


class RasterStacProperties[DataT: xr.DataArray | xr.Dataset]:
    """Read and write STAC Item properties held as marked coordinates.

    A coordinate is a property exactly when its attrs hold `stac_property`.
    A per-Item property follows the Item dimension; a shared one is scalar.
    """

    _data: DataT

    @property
    def stac_properties(self) -> dict[str, Hashable]:
        """Map each STAC property name to the coordinate that holds it.

        Examples:
            >>> ds.gs.stac_properties
            {'eo:cloud_cover': 'eo:cloud_cover'}
        """
        found: dict[str, Hashable] = {}
        for key, coord in self._data.coords.items():
            if "stac_property" not in coord.attrs:
                continue
            group = attrs.StacProperty.from_attrs(coord.attrs)
            found[group.name] = key
        return found

    def read_stac_properties(self) -> dict[str, Any]:
        """Read one Item's STAC property values from scalar marked coordinates.

        Returns:
            Each STAC property name mapped to its value. A missing value,
            None or NaN, is left out, since JSON holds neither.

        Raises:
            ValueError: A property coordinate holds one value per Item; select
                one Item first.

        Examples:
            >>> scene.isel(time=0).gs.read_stac_properties()
            {'eo:cloud_cover': 12.5, 'platform': 'sentinel-2b'}
        """
        values: dict[str, Any] = {}
        for property_name, coord_name in self.stac_properties.items():
            value = self._data.coords[coord_name].item()
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            values[property_name] = value
        return values

    def write_stac_property(
        self, name: str, values: Any, *, item_dim: str = TIME_COORDINATE
    ) -> DataT:
        """Write one STAC property as a marked coordinate.

        Args:
            name: STAC property name, such as "eo:cloud_cover".
            values: A scalar shared by every Item, or one scalar per Item. None
                marks a missing value.
            item_dim: Dimension the per-Item values follow.

        Returns:
            New raster with the coordinate; pixels are not read.

        Raises:
            KeyError: Per-Item values are given and `item_dim` is absent.
            ValueError: The length differs from `item_dim`, a value is not a
                scalar, or `name` is a dimension or variable name.

        Examples:
            >>> ds.gs.write_stac_property("platform", "Sentinel-2B")["platform"].dims
            ()
        """
        taken = set(self._data.dims)
        if isinstance(self._data, xr.Dataset):
            taken.update(self._data.data_vars)
        else:
            taken.add(self._data.name)
        if name in taken:
            raise ValueError(f"{name!r} is a dimension or variable, not a property")

        if isinstance(values, (tuple, dict, set)):
            raise ValueError(f"Property {name!r} values must be scalar")

        if isinstance(values, (list, np.ndarray)):
            items = list(values)
            for item in items:
                if isinstance(item, (list, tuple, dict, set, np.ndarray)):
                    raise ValueError(f"Property {name!r} values must be scalar")
            if item_dim not in self._data.dims:
                raise KeyError(f"Dimension {item_dim!r} is absent")
            if len(items) != self._data.sizes[item_dim]:
                raise ValueError(
                    f"Property {name!r} has length {len(items)}, "
                    f"but {item_dim!r} has {self._data.sizes[item_dim]}"
                )
            # An array keeps its dtype, so object strings are not re-inferred.
            coord = xr.Variable(item_dim, np.asarray(values))
        else:
            coord = xr.Variable((), values)

        coord = attrs.stamp_attrs(coord, attrs.StacProperty(name=name))
        return self._data.assign_coords({name: coord})


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

    def sample(
        self,
        freq: Freq,
        *,
        size: int = 1,
        stride: int = 1,
        keep: Keep = "last",
    ) -> list[DataT]:
        """Sample the time axis into windows of consecutive slots, lazily.

        Each window is one model input in time: this raster at `size` slots
        of length `freq` in a row. Cut it in space with `RasterChipper`.
        A slot holding several observations keeps one of them; blend them
        with `composite` first when that is wanted. A window holding an empty
        slot is left out. Only the time labels are read.

        Args:
            freq: Length of one slot, as a pandas offset alias such as "MS".
            size: Slots per window, the length of each window's time axis.
            stride: Slots between the starts of consecutive windows.
            keep: The observation a slot keeps when several fall in it.

        Returns:
            One raster per window, in time order; `start_datetime` and
            `end_datetime` follow their time labels.

        Raises:
            ValueError: The raster has no datetime `time` dimension, `size`
                or `stride` is below 1, or `keep` is unknown.

        Examples:
            >>> pairs = scene.gs.sample("MS", size=2, keep="last")
            >>> pairs[0].time.dt.strftime("%m-%d").values.tolist()
            ['06-28', '07-23']
        """
        from .utils.windows import window_times

        times = self.times
        if times is None or TIME_COORDINATE not in self._data.dims:
            raise ValueError("sample needs a datetime `time` dimension")

        windows = []
        for positions in window_times(times, freq, size=size, stride=stride, keep=keep):
            windows.append(self._data.isel({TIME_COORDINATE: positions}))
        return windows

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


# A rule raises RasterValidationError when the raster breaks it; `gs.validate`
# runs rules without reading pixels.
type Rule[DataT] = Callable[[DataT], None]


def georeferenced(raster: xr.DataArray | xr.Dataset | xr.DataTree) -> None:
    """Require a grid ODC recognizes, with a CRS.

    Raises:
        RasterValidationError: ODC finds no grid, or the grid has no CRS.
        NotImplementedError: Pixels are placed by ground control points.
    """
    geobox = raster.gs.geobox
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

    def validate(self, *rules: Rule[DataT]) -> None:
        """Require a grid ODC recognizes, with a CRS, then every rule, without reading pixels.

        Args:
            *rules: Rules the operation needs beyond a georeferenced grid, such
                as `decoded` for a band or `same_pixel_meaning` for a Dataset.

        Raises:
            RasterValidationError: The grid or a rule is broken. The error
                names the variable or node and how to fix it.

        Examples:
            >>> scene.gs.validate()
            >>> band.gs.validate(decoded)
            >>> scene.gs.validate(same_pixel_meaning)
        """
        georeferenced(self._data)

        for rule in rules:
            rule(self._data)

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
    def region(self) -> GeoRegion:
        """Read spatial extent and temporal coverage using the concrete raster's rules."""
