"""Properties, pixel operations and summaries shared by every `gs` raster accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import xarray as xr

from geosave.geodata.conventions import TIME_COORDINATE, require_yx
from geosave.geodata.core.raster.time import read_bounds

if TYPE_CHECKING:
    import geopandas as gpd
    from collections.abc import Mapping

    from odc.geo import CRS, BoundingBox, Resolution
    from odc.geo.gcp import GCPGeoBox
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import Geometry

    from odc.geo import SomeResolution

    from geosave.geodata.transform.warp import Resampling, Target
    from geosave.geodata.utils.datetime import DateRange

    from geosave.geodata.core.anchor import GeoAnchor


class GeoRaster[DataT: xr.Dataset | xr.DataArray | xr.DataTree]:
    """Spatial, attrs, and pixel members every `gs` raster accessor shares.

    Concrete accessors bind their xarray object to `_data`; this class is
    never built directly. Each pixel operation delegates to `transform`,
    which treats a band, a raster, and a tree alike.
    """

    _data: DataT

    def __init__(self, data: DataT) -> None:
        """Bind the xarray object.

        Args:
            data: Band, raster, or tree to read through this accessor.

        Raises:
            ValueError: Its grid, or a group's, spans dimensions other than
                `y` and `x`.
        """
        require_yx(data)
        self._data = data

    @property
    def geobox(self) -> GeoBox | GCPGeoBox | None:
        """Read the pixel grid, if this object carries one.

        Same lookup as `.odc.geobox`, since `gs` is built on odc-geo: None is
        a normal answer, not a refused one. A caller that needs ground
        position checks for None and a regular `GeoBox` itself.

        Returns:
            Grid including CRS, transform, bounds, and shape; a `GCPGeoBox`
            if placed by ground control points rather than a transform; or
            None if this object carries no CRS or spatial dims. A tree's is
            the one grid all its grid nodes share, or None where they differ.
        """
        if not isinstance(self._data, xr.DataTree):
            return self._data.odc.geobox
        # A DataTree carries no `.odc` accessor, and its root holds no grid.
        shared = None
        for node in self._data.leaves:
            grid = node.to_dataset(inherit="all_coords").odc.geobox
            if grid is None or (shared is not None and grid != shared):
                return None
            shared = grid
        return shared

    @property
    def is_georeferenced(self) -> bool:
        """Whether this object carries a locatable grid.

        Returns:
            True when `geobox` resolves to one.
        """
        return self.geobox is not None

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
    def timespan(self) -> DateRange | None:
        """Read inclusive temporal coverage.

        A composite carries `start_datetime` and `end_datetime`, so its span runs from the earliest
        start to the latest end. Plain instants span from the oldest to the
        newest label.

        Returns:
            First and last covered instant, or None for timeless data.
        """
        coords = getattr(self._data, "coords", None)
        if coords is None or TIME_COORDINATE not in coords:
            return None
        bounds = read_bounds(coords)
        if bounds is None:
            starts = ends = np.atleast_1d(coords[TIME_COORDINATE].values)
        else:
            starts, ends = bounds
        first, last = starts.min(), ends.max()
        if np.isnat(first) or np.isnat(last):
            return None
        # Microsecond units convert to datetime; nanosecond ones would give an int.
        return first.astype("datetime64[us]").item(), last.astype(
            "datetime64[us]"
        ).item()

    @property
    def times(self) -> pd.DatetimeIndex | None:
        """Read the time coordinate labels.

        Returns:
            Labels in axis order, or None for timeless data.

        Examples:
            >>> ds.gs.times.strftime("%Y%m%d").tolist()
            ['20250601', '20250611']
        """
        # ty misbinds the coords property on the bounded type parameter.
        data = cast(xr.Dataset | xr.DataArray | xr.DataTree, self._data)
        coords = data.coords
        if TIME_COORDINATE not in coords:
            return None
        return pd.DatetimeIndex(np.atleast_1d(coords[TIME_COORDINATE].values))

    @property
    def anchor(self) -> GeoAnchor:
        """Read exact spatial and temporal coverage.

        Returns:
            Anchor over this object's coverage and time span, which names its
            centroid and place and formats both into a filename.

        Raises:
            ValueError: This object carries no locatable grid.

        Examples:
            >>> ds["ndvi"].gs.anchor.format("{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}")
            '45.00N_13.00E_20250601'
        """
        from odc.geo.geobox import GeoBox

        from geosave.geodata.core.anchor import GeoAnchor

        geobox = self.geobox
        if not isinstance(geobox, GeoBox):
            raise ValueError(f"{type(self._data).__name__} carries no locatable grid")
        return GeoAnchor(geobox.extent, timespan=self.timespan)

    def unpack(self) -> DataT:
        """Read physical values out of stored digital numbers.

        Returns:
            New object of physical values, a variable unchanged where it
            carries neither `scale_factor` nor `add_offset`.

        Examples:
            >>> ds.gs.unpack().B04.max().item()
            0.09
        """
        from geosave.geodata.transform import nodata

        return nodata.unpack(self._data)

    def mask_and_scale(self) -> DataT:
        """Decode stored nodata, scale, and offset into physical values.

        Nodata is masked before scaling so its stored value cannot become
        an ordinary reading. Already decoded variables pass through.

        Returns:
            New object of the same kind, preserving coordinates, descriptive
            metadata, and laziness. Decoded storage metadata leaves attrs.

        Examples:
            >>> reflectance = ds.gs.mask_and_scale()
            >>> reflectance.red.dtype
            dtype('float32')
        """
        from geosave.geodata.transform import nodata

        return nodata.unpack(nodata.to_nan(self._data))

    def mask(
        self, valid: xr.DataArray | np.ndarray, *, fill: float | int | None = None
    ) -> DataT:
        """Make nodata every pixel `valid` does not keep.

        Args:
            valid: Boolean array, True where a pixel is real data. A DataArray
                names its own axes and may span fewer, broadcasting over the
                rest; a bare numpy array is read as the grid alone.
            fill: Value the blanked pixels take, replacing nodata metadata
                on each affected variable. None reads what each one carries.

        Returns:
            New object, its unkept pixels holding their variable's fill value.

        Raises:
            ValueError: `valid` is not boolean, does not span the grid, spans an
                axis this object does not, `fill` does not fit a dtype, or a
                variable carries no fill value and none is given.

        Examples:
            >>> clear = ds.gs.mask(ds.scl.isin([4, 5, 6, 7]))
        """
        from geosave.geodata.transform import nodata

        return nodata.mask(self._data, valid, fill=fill)

    def to_nan(self) -> DataT:
        """Replace each variable's fill value with NaN.

        Returns:
            New object holding NaN where the pixels were nodata, a variable
            unchanged where it carries no fill value.

        Examples:
            >>> ds.gs.to_nan().red.dtype
            dtype('float32')
        """
        from geosave.geodata.transform import nodata

        return nodata.to_nan(self._data)

    def reproject(
        self,
        target: Target,
        *,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | None = None,
    ) -> DataT:
        """Warp pixels onto the grid a target names.

        A target grid is adopted whole — CRS, resolution, and extent — while a
        bare CRS only decides the projection, sizing the grid from this object.

        Args:
            target: Grid to land on, a raster already on one, or a CRS.
            resampling: One GDAL kernel for every variable, or a mapping naming
                each variable's own, which `"*"` answers the rest of.
            resolution: Output pixel size, taken only for a CRS target. None
                keeps the ground sampling as closely as the new CRS allows.

        Returns:
            New object on the target grid, carrying its own `spatial_ref` and
            the CF semantics its axes earn.

        Raises:
            ValueError: This object or `target` sits on no locatable grid,
                `resolution` contradicts a target grid, a variable does not
                span the grid, or `resampling` would blend class codes.

        Examples:
            >>> ds.gs.reproject("EPSG:3857").gs.crs.epsg
            3857
            >>> srtm.gs.reproject(scene, resampling="bilinear")
        """
        from geosave.geodata.transform import warp

        return warp.reproject(
            self._data, target, resampling=resampling, resolution=resolution
        )

    def crop(
        self, region: Geometry | gpd.GeoSeries | gpd.GeoDataFrame, *, mask: bool = True
    ) -> DataT:
        """Cut this object down to the extent of a region.

        Args:
            region: Geometry, or geometries, to cut against, reprojected onto
                this object's CRS where they sit in another. Disconnected
                geometries share one cut.
            mask: Also make nodata the pixels outside the geometries, which
                then take their own variable's fill value.

        Returns:
            New object covering the region's extent, its dtype unchanged.

        Raises:
            TypeError: `region` carries no CRS, as a bare shapely geometry.
            ValueError: This object carries no CRS, `region` is empty or does
                not overlap it, or `mask` is set while a variable carries no
                fill value.

        Examples:
            >>> ds.gs.crop(field_boundaries)
            >>> ds.gs.crop(box(10, 10, 30, 30, "EPSG:32633"))
        """
        from geosave.geodata.transform.extent import crop

        return crop(self._data, region, mask=mask)
