"""Properties, pixel operations and summaries shared by every `gs` raster accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import xarray as xr
from dask.base import compute

import geosave.geodata.attrs as attrs
from geosave.geodata.utils.datetime import parse_daterange

from geosave.geodata.conventions import TIME_COORDINATE, require_yx

if TYPE_CHECKING:
    import geopandas as gpd
    import torch
    from collections.abc import Callable, Hashable, Mapping

    from odc.geo import CRS, BoundingBox, Resolution
    from odc.geo.gcp import GCPGeoBox
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import Geometry

    from odc.geo import SomeResolution

    from geosave.geodata.transform.warp import Resampling, Target
    from geosave.geodata.utils.datetime import DateRange

    from .anchor import GeoAnchor


def as_tensor(
    to_numpy: Callable[..., np.ndarray], dtype: str | torch.dtype | None
) -> torch.Tensor:
    """Read an accessor's model-input array as a writable, contiguous tensor.

    Args:
        to_numpy: The accessor's own `to_numpy`, which fixes the axis order.
        dtype: Torch dtype or its string name. None keeps the pixel dtype. A
            dtype NumPy cannot hold, such as bfloat16, reads as float32 first.

    Returns:
        Tensor laid out as `to_numpy` lays out its array.

    Raises:
        ValueError: A dtype name is unknown.
        TypeError: `dtype` is neither a torch dtype nor a name, or the pixel
            dtype has no tensor equivalent.
    """
    # Imported here so that importing geodata never pays for torch.
    import torch

    if isinstance(dtype, str):
        target = getattr(torch, dtype, None)
        if not isinstance(target, torch.dtype):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
    elif dtype is None or isinstance(dtype, torch.dtype):
        target = dtype
    else:
        raise TypeError(f"{dtype!r} is not a torch dtype or the name of one")

    # Read-only pixel buffers are copied, so tensor operations can safely write.
    if target is None:
        values = np.require(to_numpy(), requirements=["C", "W"])
        try:
            return torch.as_tensor(values)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Cannot convert numpy dtype {values.dtype} to a torch tensor"
            ) from error

    try:
        reading = torch.empty(0, dtype=target).numpy().dtype
    except TypeError:
        reading = np.dtype("float32")
    return torch.as_tensor(
        np.require(to_numpy(dtype=reading), requirements=["C", "W"]), dtype=target
    )


def statistics(variables: Mapping[Hashable, xr.DataArray]) -> pd.DataFrame:
    """Summarise variables while computing shared pixel sources only once.

    Args:
        variables: Row names mapped to variables. Each variable is reduced
            across all axes, excluding NaN and its own fill value.

    Returns:
        DataFrame in input order with minimum, maximum, mean, population
        stddev, and valid_percent columns. An empty input keeps these columns.

    Raises:
        ValueError: A variable holds no present pixels.
    """
    reductions = []
    for band in variables.values():
        fill = band.gs.nodata
        values = band if fill is None else band.where(band != fill)
        present = values.notnull().sum()
        reductions.append(
            xr.Dataset(
                {
                    "present": present,
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "mean": values.mean(),
                    "stddev": values.std(),
                    "valid_percent": 100.0 * present / values.size,
                }
            )
        )

    columns = ["minimum", "maximum", "mean", "stddev", "valid_percent"]
    rows = {}
    for name, summary in zip(variables, compute(*reductions), strict=True):
        if not int(summary["present"]):
            raise ValueError(
                f"{name} holds no present pixel, so it summarises to "
                f"nothing; drop the band or give it pixels that are not fill"
            )
        row = {}
        for column in columns:
            row[column] = float(summary[column])
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=pd.Index(columns))


class GeoRasterAccessor[DataT: xr.Dataset | xr.DataArray | xr.DataTree]:
    """Spatial, attrs, and pixel members every `gs` raster accessor shares.

    Concrete accessors bind their xarray object to `_data`; this class is
    never built directly. Each pixel operation delegates to `transform`,
    which treats a band, a raster, and a stack alike.
    """

    _data: DataT

    def __init__(self, data: DataT) -> None:
        """Bind the xarray object.

        Args:
            data: Band, raster, or stack to read through this accessor.

        Raises:
            ValueError: Its grid, or a group's, spans dimensions other than
                `y` and `x`.
        """
        require_yx(data)
        self._data = data

    @property
    def _grid_source(self) -> xr.Dataset | xr.DataArray:
        """Return the object odc-geo reads a geobox off directly.

        A DataTree carries no `.odc` accessor of its own, so its root
        Dataset stands in for it.

        Returns:
            `_data` itself, or its root Dataset for a DataTree.
        """
        return self._data.dataset if isinstance(self._data, xr.DataTree) else self._data

    @property
    def geobox(self) -> GeoBox | GCPGeoBox | None:
        """Read the pixel grid, if this object carries one.

        Same lookup as `.odc.geobox`, since `gs` is built on odc-geo: None is
        a normal answer, not a refused one. A caller that needs ground
        position checks for None and a regular `GeoBox` itself.

        Returns:
            Grid including CRS, transform, bounds, and shape; a `GCPGeoBox`
            if placed by ground control points rather than a transform; or
            None if this object carries no CRS or spatial dims.
        """
        return self._grid_source.odc.geobox

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

        A resampled axis carries a `TimeSpec`, so a label standing for a month
        covers that month. An axis carrying none covers what its labels spell:
        `2018-12-26` is a whole date, `2018-12-26T10:30:31` one second.

        Returns:
            First and last covered instant, or None for timeless data.
        """
        coords = getattr(self._data, "coords", None)
        if coords is None:
            return None
        if TIME_COORDINATE not in coords:
            return None
        labels = coords[TIME_COORDINATE].values
        spec = attrs.parse(coords[TIME_COORDINATE], attrs.TimeSpec)
        if spec is not None:
            return spec.timespan(labels)

        covered = [
            parse_daterange(str(spelled))
            for spelled in np.datetime_as_string(np.atleast_1d(labels), unit="auto")
        ]
        return min(start for start, _ in covered), max(end for _, end in covered)

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
            Anchor over this object's grid and time span, which names its
            centroid and place and formats both into a filename.

        Raises:
            ValueError: This object carries no locatable grid.

        Examples:
            >>> ds["ndvi"].gs.anchor.format("{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}")
            '45.00N_13.00E_20250601'
        """
        from odc.geo.geobox import GeoBox

        from .anchor import GeoAnchor

        geobox = self.geobox
        if not isinstance(geobox, GeoBox):
            raise ValueError(f"{type(self._data).__name__} carries no locatable grid")
        return GeoAnchor(geobox, timespan=self.timespan)

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
            fill: Value the blanked pixels take, also written where a variable
                carries no fill value yet. None reads what each one carries.

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
        from geosave.geodata.transform.crop import crop

        return crop(self._data, region, mask=mask)
