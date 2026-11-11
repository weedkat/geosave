"""The `gs` xarray DataArray accessor."""

from __future__ import annotations

import itertools

from typing import Any, TYPE_CHECKING, Literal, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import (
    BAND_DIMENSION,
)
from geosave.geodata.core.raster.utils.statistics import class_percent, statistics
from geosave.geodata.core.raster.utils.tensor import as_tensor


from .base import GeoRaster, RasterTime
from .transform.temporal import Reducer, Interpolation
from geosave.geodata.errors import RasterValidationError

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from geosave.geodata.utils.datetime import Freq
    from odc.geo import SomeResolution
    from .transform.warp import Resampling
    from collections.abc import Callable
    from .transform.extent import Padding, Region
    import pystac
    from odc.geo.geom import Geometry
    from geosave.geodata.core.region import GeoRegion
    from collections.abc import Mapping, Sequence

    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike

    from odc.geo import SomeCRS
    from odc.geo.gcp import GCPGeoBox
    from odc.geo.geobox import GeoBox

    from geosave.geodata import DataArray, Dataset, GeoDataFrame
    from dask.delayed import Delayed
    from upath import UPath
    from geosave.geodata.io.formats.geotiff import COGWriteOptions
    from geosave.geodata.io.formats.geotiff import GTiffWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from geosave.geodata.io.formats.netcdf import NetCDFEngine
    from geosave.geodata.io.formats.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.formats.zarr import ZarrWriteOptions
    from geosave.geodata.utils.color import Palette
    from geosave.geodata.viz.explore import Explorer


class GeoDataArray(RasterTime["DataArray"], GeoRaster["DataArray"]):
    """Read and transform one raster DataArray.

    A DataArray is the unit a panel draws, so this accessor carries what
    drawing needs and leaves whole-Dataset operations to `GeoDataset`.

    Args:
        data: DataArray holding one band.

    Examples:
        >>> ds["ndvi"].gs.plot(cmap="YlGn")
    """

    def composite(
        self,
        freq: Freq,
        method: Reducer,
        *,
        dim: str = "time",
        interpolation: Interpolation | None = None,
        properties: Literal["intersect", "stack"] = "intersect",
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | datetime = "start_day",
        offset: str | timedelta | None = None,
    ) -> DataArray:
        """Reduce decoded observations into native calendar bins.

        Args:
            freq: Pandas offset alias, such as "MS".
            method: Native reducer; class codes permit only first or last.
            dim: Datetime dimension to resample, default "time".
            interpolation: Fill interior pixel gaps with linear or nearest
                interpolation after reduction. None keeps gaps. Lazy output is
                rechunked along dim; properties remain observed.
            properties: Intersect scene properties or stack them in source order.
            closed: Inclusive bin edge, following native resample defaults.
            label: Bin label edge, following native resample defaults.
            origin: Native resample origin.
            offset: Shift applied to the origin.

        Returns:
            Lazy composite whose `start_datetime`/`end_datetime` state each
            resample bin, with CF cell methods. Input cells are replaced. Empty
            bins keep their cell and, without interpolation, NaN pixels;
            all-missing sums follow native behavior and return zero. Variance
            drops units for the caller to restate.

        Raises:
            ValueError: The selected datetime dimension is absent, active CF attrs require explicit
                `mask_and_scale()`, or class codes would be reduced numerically.
        """
        from .transform.temporal import CELL_METHODS, composite_da

        data = self._data
        cell_method = CELL_METHODS[method]
        if attrs.Packing.is_stated_in(data.attrs):
            raise ValueError(
                "decode active CF attrs with mask_and_scale() before composite"
            )
        if self.is_categorical and (
            cell_method != "point" or interpolation is not None
        ):
            raise ValueError(
                "class codes require 'first' or 'last' without interpolation"
            )

        result = composite_da(
            data,
            freq,
            method,
            dim=dim,
            interpolation=interpolation,
            properties=properties,
            closed=closed,
            label=label,
            origin=origin,
            offset=offset,
        )
        return cast("DataArray", result)

    @property
    def spatial_dims(self) -> tuple[str, str] | None:
        """Find spatial dimension names in vertical, horizontal order.

        CF ``axis``, ``standard_name`` and geographic units take precedence.
        Missing declarations may fall back to conventional names (``y``/``x``,
        ``latitude``/``longitude``, or ``lat``/``lon``). Ambiguous CF
        axes return None. Discovery does not require a CRS or compute pixels.

        Returns:
            Spatial pair, or None when this array's axes cannot be identified.

        Examples:
            >>> temperature.gs.spatial_dims
            ('latitude', 'longitude')
            >>> temperature.gs.validate()
            >>> from geosave.geodata.conventions import to_yx
            >>> to_yx(temperature).gs.spatial_dims
            ('y', 'x')
        """
        from .utils.grid import read_spatial_dims

        return read_spatial_dims(self._data)

    @property
    def geobox(self) -> GeoBox | GCPGeoBox | None:
        """Read this array's grid, or None when placement cannot be inferred.

        CF spatial axes are recognized without renaming the original array.
        A grid may exist without a CRS.
        """
        dims = self.spatial_dims
        if dims is None:
            return None
        data = self._data
        if data.odc.spatial_dims != dims:
            data = data.rename(dict(zip(dims, ("y", "x"), strict=True)))
        return data.odc.geobox

    def validate(self, *, georeferenced: bool = False, affine: bool = False) -> None:
        """Check spatial structure and ODC placement without reading pixels.

        Missing placement is valid. Existing spatial coordinates must support
        ODC's grid interpretation, and an affine transform must be finite and
        nonsingular. No metadata is repaired.

        Args:
            georeferenced: Additionally require ODC placement with a CRS.
            affine: Additionally require pixels on one affine `GeoBox`, not
                unplaced and not placed by ground control points.

        Raises:
            RasterValidationError: Structure or placement is invalid, or a
                requested geographic reference is absent.

        Examples:
            >>> image.gs.validate(georeferenced=True)
        """
        from odc.geo.geobox import GeoBox

        try:
            dims = self.spatial_dims
            self._validate_spatial(dims)
            self._validate_coordinates(cast("tuple[str, str]", dims))
            grid = self.geobox
            if isinstance(grid, GeoBox):
                self._validate_transform(grid)
            if georeferenced:
                self._validate_georeferenced(grid)
            if affine:
                self._validate_affine(grid)
        except RasterValidationError as error:
            if self._data.name is not None:
                error.variable = str(self._data.name)
            raise

    def _validate_georeferenced(self, grid: GeoBox | GCPGeoBox | None) -> None:
        """Require ODC placement and its coordinate reference system."""
        if grid is None:
            raise RasterValidationError(
                "Cannot determine pixel placement from the spatial coordinates. "
                "Supply its known grid with raster.gs.write_grid(geobox)."
            )
        if grid.crs is None:
            raise RasterValidationError(
                "The coordinate reference system (CRS) is missing. "
                "If you know the CRS of the existing coordinates, assign it "
                "with raster.gs.write_crs(...). This does not reproject pixels."
            )

    def _validate_affine(self, grid: GeoBox | GCPGeoBox | None) -> None:
        """Require pixels placed by one affine transform."""
        from odc.geo.geobox import GeoBox

        if grid is None:
            raise RasterValidationError(
                "Cannot determine pixel placement from the spatial coordinates. "
                "Supply its known grid with raster.gs.write_grid(geobox)."
            )
        if not isinstance(grid, GeoBox):
            raise RasterValidationError(
                "Pixels are placed by ground control points, not one affine grid. "
                "Reproject onto a GeoBox with raster.gs.reproject(...) first."
            )

    def _validate_spatial(self, dims: tuple[str, str] | None) -> None:
        """Check identified spatial axes without discovering or returning metadata."""
        if dims is None:
            raise RasterValidationError(
                "Cannot identify the spatial dimensions. Use recognized names such as "
                "y/x or latitude/longitude, or mark coordinate axes with CF axis='Y' "
                "and axis='X' attributes."
            )
        for dim in dims:
            if self._data.sizes[dim] == 0:
                raise RasterValidationError(
                    f"Spatial dimension {dim!r} contains no pixels. "
                    "Select a nonempty raster region before this operation."
                )

    def _validate_transform(self, grid: GeoBox) -> None:
        """Reject nonfinite or singular transforms returned by ODC."""
        if not np.isfinite(tuple(grid.transform)).all():
            raise RasterValidationError(
                "The pixel transform contains nonfinite values. "
                "Correct the placement with raster.gs.write_grid(geobox) using a known grid."
            )
        if grid.transform.is_degenerate:
            raise RasterValidationError(
                "The pixel transform collapses pixels onto a line or point. "
                "Supply a grid with nonzero pixel area using raster.gs.write_grid(geobox)."
            )

    def _validate_coordinates(self, dims: tuple[str, str]) -> None:
        """Check the regular-axis assumption used by ODC's affine inference."""
        for dim in dims:
            coordinate = self._data.coords.get(dim)
            if coordinate is None or coordinate.ndim != 1:
                continue
            values = coordinate.values
            if not np.issubdtype(values.dtype, np.number):
                raise RasterValidationError(
                    f"Spatial coordinate {dim!r} is not numeric. "
                    "Supply numeric positions in the raster's coordinate reference system."
                )
            if not np.isfinite(values).all():
                raise RasterValidationError(
                    f"Spatial coordinate {dim!r} contains nonfinite values. "
                    "Correct missing or infinite coordinate positions before this operation."
                )
            steps = np.diff(values.astype("float64"))
            if steps.size == 0:
                continue
            if steps[0] == 0:
                raise RasterValidationError(
                    f"Spatial coordinate {dim!r} repeats positions. "
                    "Correct the coordinates so adjacent pixels have distinct positions."
                )
            if not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-8):
                raise RasterValidationError(
                    f"Spatial coordinate {dim!r} is not regularly spaced. "
                    "Check the coordinate positions or resample the data onto a regular grid."
                )

    def expand_rasters(self, dim: str | Sequence[str]) -> tuple[DataArray, ...]:
        """Split this array into one lazy slice per position along `dim`.

        Args:
            dim: Dimension, or dimensions in order, to split along. An empty
                sequence keeps the array whole.

        Returns:
            One slice per position, in order. Each slice carries its labels as
            scalar coordinates; a dimension without coordinates labels each
            slice by its position. Pixels stay lazy.

        Raises:
            ValueError: A dimension is repeated, absent, or spatial.

        Examples:
            >>> [chip.time.dt.month.item() for chip in band.gs.expand_rasters("time")]
            [1, 2]
        """
        dims = (dim,) if isinstance(dim, str) else tuple(dim)
        if len(set(dims)) != len(dims):
            raise ValueError(f"split dimensions repeat: {dims}")
        for name in dims:
            if name not in self._data.dims:
                raise ValueError(f"split dimension {name!r} is absent")
            if name in (self.spatial_dims or ()):
                raise ValueError(f"cannot split spatial dimension {name!r}")

        ranges = []
        for name in dims:
            ranges.append(range(self._data.sizes[name]))
        slices = []
        for positions in itertools.product(*ranges):
            selection = dict(zip(dims, positions, strict=True))
            piece = self._data.isel(selection)
            # A dimension without labels names each slice by its position.
            unlabelled = {}
            for name, position in selection.items():
                if name not in piece.coords:
                    unlabelled[name] = position
            slices.append(cast("DataArray", piece.assign_coords(unlabelled)))
        return tuple(slices)

    @property
    def variables(self) -> tuple[str, ...]:
        """Name this array as a data variable, when it has a name."""
        return () if self._data.name is None else (str(self._data.name),)

    @property
    def axes(self) -> dict[str, np.ndarray | None]:
        """Read the axes this band spans besides the grid.

        Returns:
            {
                "<axis name>": its labels, or None where it carries none,
            }
            In the band's own order, ready to pass to `xr.DataArray` or `xr.Dataset` as `coords`.

        Examples:
            >>> ds["ndvi"].gs.axes
            {'time': array(['2025-06-01T00:00:00.000000000'], dtype='datetime64[ns]')}
        """
        spatial = self.spatial_dims or ()
        axes = {}
        for dim in self._data.dims:
            if dim in spatial:
                continue
            values = None
            if dim in self._data.coords:
                values = self._data.coords[dim].values
            axes[str(dim)] = values
        return axes

    @property
    def nodata(self) -> float | int | None:
        """Read the active scalar missing-value code, or None when none is declared."""
        found = attrs.Packing.from_attrs(self._data.attrs)
        if found is None:
            return None
        if found.fill_value is not None:
            return found.fill_value
        if isinstance(found.missing_value, (int, float)):
            return found.missing_value
        return None

    @property
    def legend(self) -> attrs.Legend | None:
        """Read what this band's values mean, or None where they are no classes."""
        return attrs.Legend.from_attrs(self._data.attrs)

    @property
    def is_categorical(self) -> bool:
        """Say whether values are class codes that blending or ranking would destroy.

        Dtype does not say: Sentinel-2 reflectance and a land cover map are
        both integers. CF flags (`flag_values` or `flag_masks` with
        `flag_meanings`, read by cf-xarray) mark the values as codes, and so
        does a GDAL colour table, which states colours but no CF flags.

        Returns:
            True for a CF flag variable or `colorinterp="palette"`; otherwise
            False. Pixel values and dtype are not inspected.

        Examples:
            >>> scene.landcover.gs.is_categorical, scene.red.gs.is_categorical
            (True, False)
        """
        if self._data.cf.is_flag_variable:
            return True
        gdal_band = attrs.GDALBand.from_attrs(self._data.attrs)
        return gdal_band is not None and gdal_band.colorinterp == "palette"

    @property
    def description(self) -> attrs.Description | None:
        """Read what this variable's values are (`long_name`, `standard_name`, `units`), or None."""
        return attrs.Description.from_attrs(self._data.attrs)

    @property
    def packing(self) -> attrs.Packing | None:
        """Read this variable's active CF scale, offset and missing-value declarations."""
        return attrs.Packing.from_attrs(self._data.attrs)

    @property
    def spectral(self) -> attrs.Spectral | None:
        """Read where in the spectrum this band was measured, or None."""
        return attrs.Spectral.from_attrs(self._data.attrs)

    def write_grid(self, geobox: GeoBox | GCPGeoBox) -> DataArray:
        """Assign grid coordinates and CRS, preserving spatial names and axis order.

        The usual way back from a model: wrap its output in `xr.DataArray`,
        then place it on the chip it came from.

        Args:
            geobox: Regular grid, or a grid placed by ground control points.

        Returns:
            New DataArray pointing at the CRS coordinate, its axes carrying CF
            attrs on a regular grid. Coordinates on other dimensions are kept.

        Raises:
            RasterValidationError: Spatial dimensions are missing, the pixel shape
                differs from the grid, or the supplied grid carries no CRS.

        Examples:
            >>> pred = xr.DataArray(logits.argmax(0), dims=("y", "x"), name="label")
            >>> pred.gs.write_grid(chip.gs.geobox).gs.write_nodata(255).gs.geobox == chip.gs.geobox
            True
        """
        from geosave.geodata.core.raster.utils import grid

        return cast(
            "DataArray", grid.clear_mapping_attr(grid.write_grid(self._data, geobox))
        )

    def reproject(
        self,
        how: GeoBox | SomeCRS,
        *,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | None = None,
    ) -> DataArray:
        """Warp pixels onto the grid a destination names.

        A destination grid is adopted whole — CRS, resolution, and extent — while a
        bare CRS only decides the projection, sizing the grid from this object.

        Args:
            how: Exact GeoBox or CRS to land on. Use reference.gs.geobox
                to match an existing raster.
            resampling: One GDAL kernel for every variable, or a mapping naming
                each variable's own, which `"*"` answers the rest of.
            resolution: Output pixel size, taken only for a CRS destination. None
                keeps the ground sampling as closely as the new CRS allows.

        Returns:
            New object on the destination grid, carrying its own `spatial_ref` and
            the CF semantics its axes earn.

        Raises:
            ValueError: This object or `how` sits on no locatable grid,
                `resolution` contradicts a destination grid, a variable does not
                span the grid, or `resampling` would blend class codes.

        Examples:
            >>> ds.gs.reproject("EPSG:3857").gs.crs.epsg
            3857
            >>> srtm.gs.reproject(scene.gs.geobox, resampling="bilinear")
        """

        from .transform.warp import reproject

        return cast(
            "DataArray",
            reproject(self._data, how, resolution=resolution, resampling=resampling),
        )

    def mask(
        self, mask: xr.DataArray | np.ndarray, *, fill: float | int | None = None
    ) -> DataArray:
        """Make nodata every pixel `mask` does not keep.

        Args:
            mask: Boolean array, True where a pixel is kept. A DataArray
                names its own axes and may span fewer, broadcasting over the
                rest; a bare numpy array is read as the grid alone.
            fill: Value the blanked pixels take, replacing nodata metadata
                on each affected variable. None uses each variable's packing fill_value.

        Returns:
            New object, its unkept pixels holding their variable's fill value.

        Raises:
            ValueError: `mask` is not boolean, has the wrong spatial shape, spans an
                axis this object does not, `fill` does not fit a dtype, or a
                variable carries no fill value and none is given.

        Examples:
            >>> clear = ds.gs.mask(ds.scl.isin([4, 5, 6, 7]))
        """

        from .transform.nodata import mask_with_da, mask_with_np

        if isinstance(mask, np.ndarray):
            return cast("DataArray", mask_with_np(self._data, mask, fill=fill))

        foreign = set(mask.dims) - set(self._data.dims)
        if foreign:
            raise ValueError(f"mask spans axes the raster does not carry: {foreign}")
        return cast("DataArray", mask_with_da(self._data, mask, fill=fill))

    def crop(self, region: Region, *, mask: bool = True) -> DataArray:
        """Crop to a CRS-bearing region, optionally masking outside its geometry.

        Args:
            region: ODC geometry/bounds or GeoPandas geometries with a CRS.
            mask: Fill pixels outside the geometry with declared nodata.

        Returns:
            Lazy raster covering the region, retaining its declared extent.

        Raises:
            TypeError: Region has no CRS.
            ValueError: Region is empty, does not overlap, or masking lacks nodata.
        """
        from .transform.extent import crop

        return cast("DataArray", crop(self._data, region, mask=mask))

    def crop_to_grid(
        self,
        geobox: GeoBox,
        *,
        mode: Padding = "constant",
        constant_value: int | float = 0,
    ) -> DataArray:
        """Take the pixels of an aligned grid, padding where it leaves this raster.

        Unlike `read_pixels`, nothing is resampled, converted or computed.

        Args:
            geobox: Grid sharing this raster's CRS and pixel size, offset by
                whole pixels, such as `chips.geobox(key)`. It may extend past
                the raster edges.
            mode: How pixels outside the raster are filled, as in `numpy.pad`.
            constant_value: Fill for `"constant"`, keeping the dtype.

        Returns:
            Lazy DataArray on exactly `geobox`. A declared extent is clipped to it;
            time and other coordinates, nodata and other attrs are kept.

        Raises:
            ValueError: This raster has no regular grid, or `geobox` is not
                pixel-aligned with it or does not overlap it.

        Examples:
            >>> chip = da.gs.crop_to_grid(da.gs.geobox[0:224, 0:224])
            >>> chip.gs.region.extent.centroid
            <Geometry ...>
        """

        from .transform.extent import crop_to_grid

        return cast(
            "DataArray",
            crop_to_grid(self._data, geobox, mode=mode, constant_value=constant_value),
        )

    @property
    def region(self) -> GeoRegion:
        """Read the declared extent, falling back to this array's grid, with its time span.

        Returns:
            Region over this object's coverage and time span, which names its
            centroid and place and formats both into a filename.

        Raises:
            ValueError: This object has neither an extent nor a locatable grid.

        Examples:
            >>> ds["ndvi"].gs.region.format("{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}")
            '45.00N_13.00E_20250601'
        """
        from geosave.geodata.core.region import GeoRegion

        extent = attrs.Extent.from_attrs(self._data.attrs)
        if extent is not None:
            return GeoRegion(extent.to_geometry(), timespan=self.timespan)
        self.validate(georeferenced=True)
        geobox = cast("GeoBox", self.geobox)
        return GeoRegion(geobox.extent, timespan=self.timespan)

    def write_extent(self, extent: Geometry | None) -> DataArray:
        """Record the declared extent without changing pixels or grid.

        Args:
            extent: CRS-aware geometry to preserve, or None to clear it.

        Returns:
            Shallow copy with updated extent attrs. Slicing and reprojection
            preserve this declaration until it is explicitly changed.

        Raises:
            ValueError: The extent is invalid, empty, lacks a CRS or is not 2D.
        """
        if extent is None:
            return attrs.clear_attrs(self._data, attrs.Extent)
        return attrs.stamp_attrs(self._data, attrs.Extent.from_geometry(extent))

    def clip_extent(self, geometry: Geometry) -> DataArray:
        """Narrow the declared extent to a geometry, as a crop narrows coverage.

        Args:
            geometry: CRS-aware geometry the pixels now cover at most.

        Returns:
            Shallow copy whose declared extent is the intersection, cleared
            when the two do not meet. Without a declared extent, unchanged.
        """
        extent = attrs.Extent.from_attrs(self._data.attrs)
        if extent is None:
            return cast("DataArray", self._data)
        clipped = extent.clip(geometry)
        if clipped is None:
            return attrs.clear_attrs(self._data, attrs.Extent)
        return attrs.stamp_attrs(self._data, clipped)

    def write_crs(self, crs: SomeCRS | None = None) -> DataArray:
        """Write onto the existing spatial coordinates what they measure.

        Args:
            crs: CRS the existing coordinates are in. None reads the one the
                band already carries.

        Returns:
            New DataArray whose spatial coordinates carry the CRS and CF attrs.

        Raises:
            RasterValidationError: Spatial placement is missing, has no CRS, or
                uses ground control points instead of an affine grid.
            ValueError: The supplied CRS cannot be parsed.

        Examples:
            >>> band.gs.write_crs("EPSG:32633").y.attrs["axis"]
            'Y'
        """
        from geosave.geodata.core.raster.utils import grid

        return cast(
            "DataArray", grid.clear_mapping_attr(grid.write_crs(self._data, crs))
        )

    def write_nodata(self, value: float | int | None) -> DataArray:
        """Write the stored value standing for this band's nodata pixels.

        It is written as CF's `_FillValue`, which odc and rioxarray read too.

        Args:
            value: Stored value marking nodata, of this band's own type. None
                clears it.

        Returns:
            New DataArray carrying `value` as its fill.

        Raises:
            ValueError: `value` is not of this band's type.

        Examples:
            >>> ds["red"].gs.write_nodata(0).attrs["_FillValue"]
            0
        """
        fields = dict(self._data.attrs)
        for key in ("_FillValue", "nodata", "missing_value"):
            fields.pop(key, None)
        if value is not None:
            fields["_FillValue"] = value
        packing = attrs.Packing.model_validate(fields)
        packing.check_dtype(self._data.dtype)
        written = self._data.copy(deep=False)
        written.attrs = fields
        return written

    def write_legend(
        self, class_map: Mapping[int, str] | None, color_map: Palette | None = None
    ) -> DataArray:
        """Write the class each of this band's stored values stands for.

        Args:
            class_map: Stored value mapped to class name. None removes the
                legend.
            color_map: Stored value mapped to hex or RGB colour. None writes
                names only.

        Returns:
            New DataArray carrying the legend.

        Raises:
            ValueError: `color_map` colours a value `class_map` does not name,
                or a value does not fit this band's dtype.

        Examples:
            >>> ds["label"].gs.write_legend({0: "water", 1: "trees"})
        """
        if class_map is None:
            return attrs.clear_attrs(self._data, attrs.Legend)
        legend = attrs.Legend.model_validate(
            {"classes": class_map, "color_map": color_map}
        )
        legend.check_dtype(self._data.dtype)
        return attrs.stamp_attrs(self._data, legend)

    def write_scale_offset(
        self, scale_factor: float | None, add_offset: float = 0.0
    ) -> DataArray:
        """Write how this band's stored values decode to physical values.

        Sentinel-2 L2A is the usual case: ESA stores reflectance as
        `DN * 1e-4 + offset`, with offset 0 for processing baseline below 04.00
        and -0.1 (`BOA_ADD_OFFSET` -1000) from 04.00, January 2022. Both are
        in each scene's `MTD_MSIL2A.xml`, which STAC items do not repeat; see
        https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/product-formatting.

        Args:
            scale_factor: Multiplier applied to each stored value. None
                removes the packing.
            add_offset: Value added after scaling.

        Returns:
            New DataArray carrying the packing; its pixels are unchanged.

        Examples:
            >>> ds["B04"].gs.write_scale_offset(1e-4, -0.1).gs.packing.scale_factor
            0.0001
        """
        fields = dict(self._data.attrs)
        for key in ("scale_factor", "add_offset"):
            fields.pop(key, None)
        if scale_factor is not None:
            fields.update(scale_factor=scale_factor, add_offset=add_offset)
        attrs.Packing.model_validate(fields)
        written = self._data.copy(deep=False)
        written.attrs = fields
        return written

    def write_spectral(
        self,
        common_name: str | None,
        *,
        center_wavelength: float | None = None,
        full_width_half_max: float | None = None,
    ) -> DataArray:
        """Write where in the spectrum this band was measured.

        Args:
            common_name: Band name shared across sensors, e.g. `"red"`. None
                with no wavelength removes the spectral description.
            center_wavelength: Centre of the band, in micrometres.
            full_width_half_max: Width at half peak response, in micrometres.

        Returns:
            New DataArray carrying the description.

        Examples:
            >>> ds["B04"].gs.write_spectral("red", center_wavelength=0.665)
        """
        fields = {
            "common_name": common_name,
            "center_wavelength": center_wavelength,
            "full_width_half_max": full_width_half_max,
        }
        if all(value is None for value in fields.values()):
            return attrs.clear_attrs(self._data, attrs.Spectral)
        return attrs.stamp_attrs(self._data, attrs.Spectral(**fields))

    def to_numpy(self, *, dtype: DTypeLike | None = None) -> np.ndarray:
        """Read this band as one model-input array.

        The grid pair trails and a `band` axis sits just ahead of it, matching
        what `GeoDataset.to_numpy` builds by stacking variables. Other axes keep
        this band's own order.

        Args:
            dtype: Cast applied to the pixels. None keeps their own dtype.

        Returns:
            Array shaped `(*axes, band, y, x)`, or `(*axes, y, x)` where this
            band spans no `band` axis.

        Examples:
            >>> ds["ndvi"].gs.to_numpy().shape
            (2, 256, 256)
        """
        self.validate()
        spatial = cast("tuple[str, str]", self.spatial_dims)
        # Put spatial dimensions last.
        ahead = [axis for axis in self.axes if axis != BAND_DIMENSION]
        if BAND_DIMENSION in self._data.dims:
            ahead.append(BAND_DIMENSION)

        ordered = self._data.transpose(*ahead, *spatial).values
        return ordered if dtype is None else ordered.astype(dtype)

    def to_tensor(self, *, dtype: str | torch.dtype | None = None) -> torch.Tensor:
        """Read this band as one model-input tensor, laid out as `to_numpy`.

        Args:
            dtype: Torch dtype or its string name. None keeps the pixel dtype.

        Returns:
            Tensor shaped `(*axes, band, y, x)`, or `(*axes, y, x)` where this
            band spans no `band` axis.

        Raises:
            ValueError: A dtype name is unknown.
            TypeError: `dtype` is neither a torch dtype nor a name, or a pixel
                dtype has no tensor equivalent.

        Examples:
            >>> ds["ndvi"].gs.to_tensor(dtype="float32").shape
            torch.Size([2, 256, 256])
        """
        return as_tensor(self.to_numpy, dtype)

    def _map_variables(
        self, change: Callable[[xr.DataArray], xr.DataArray]
    ) -> DataArray:
        """Apply one per-variable change to this array."""
        return cast("DataArray", change(self._data))

    def _read_rasters_by_path(self) -> dict[str, Dataset]:
        """Lay this array out as the one raster a publication splits.

        Band labels become variables, so splitting by variable writes one file
        per band, and writing the variables together keeps a multiband file.
        """
        from geosave.geodata.stac.publish import ROOT_PATH

        if BAND_DIMENSION in self._data.dims:
            raster = self.to_dataset(dim=BAND_DIMENSION)
        else:
            # A GeoTIFF band needs a name; an unnamed array is the file's only asset.
            named = self._data
            if named.name is None:
                named = named.rename("asset")
            raster = named.gs.to_dataset()
        return {ROOT_PATH: raster.assign_attrs(self._data.attrs)}

    def to_item(
        self,
        path: Location,
        *,
        split_item: Sequence[str] = ("time",),
        split_asset: Sequence[str] = ("variable",),
        id: str | None = None,
        format_id: str | None = None,
        format_paths: str | None = None,
        collection: pystac.Collection | None = None,
        **options: Any,
    ) -> pystac.Item:
        """Save COG assets and return one native PySTAC Item.

        Args:
            path: Output folder or filesystem URL.
            split_item: Observation dimensions; default time, ignored when absent.
            split_asset: Dimensions partitioning assets within an Item;
                "variable" gives each band its own asset; () keeps one
                multiband file.
            id: Explicit ID for one observation, overriding format_id.
            format_id: ID template using scalar observation coordinates.
            format_paths: Relative asset path template; defaults to {id}/{asset}.tif.
            collection: Native Collection overriding the stored stac_collection.
            **options: overwrite, storage_options, cog_options, and
                create_stac_item metadata options.

        Returns:
            One Item; more than one observation raises before writing.
        """
        from geosave.geodata.stac.publish import split_items
        from geosave.geodata.stac.publish import publish_items

        self.validate()
        items = split_items(self._read_rasters_by_path(), split_item, split_asset)
        if len(items) != 1:
            raise ValueError(
                "to_item requires one observation; use to_catalog for multiple Items"
            )
        published = publish_items(
            items,
            path,
            id=id,
            format_id=format_id,
            format_paths=format_paths,
            collection=collection,
            split_item=split_item,
            **options,
        )
        return published[0]

    def to_catalog(
        self,
        path: Location,
        *,
        split_item: Sequence[str] = ("time",),
        split_asset: Sequence[str] = ("variable",),
        id: str | None = None,
        format_id: str | None = None,
        format_paths: str | None = None,
        collection: pystac.Collection | None = None,
        **options: Any,
    ) -> GeoDataFrame:
        """Save COG assets and return a native Item GeoDataFrame.

        Args:
            path: Output folder or filesystem URL.
            split_item: Observation dimensions; default time, ignored when absent.
            split_asset: Dimensions partitioning assets within an Item;
                "variable" gives each band its own asset; () keeps one
                multiband file.
            id: Explicit ID for one observation, overriding format_id.
            format_id: ID template using scalar observation coordinates.
            format_paths: Relative asset path template; defaults to {id}/{asset}.tif.
            collection: Native Collection overriding the stored stac_collection.
            **options: overwrite, storage_options, cog_options, and
                create_stac_item metadata options.

        Returns:
            GeoDataFrame encoded through stac-geoparquet and Arrow.
        """
        from geosave.geodata.stac.publish import split_items
        from geosave.geodata.stac.publish import publish_items
        from geosave.geodata.stac.catalog import create_catalog

        self.validate()
        items = split_items(self._read_rasters_by_path(), split_item, split_asset)
        published = publish_items(
            items,
            path,
            id=id,
            format_id=format_id,
            format_paths=format_paths,
            collection=collection,
            split_item=split_item,
            **options,
        )
        return create_catalog(published)

    def to_cog(
        self,
        path: Location,
        *,
        split_item: Sequence[str] = ("time",),
        split_asset: Sequence[str] = ("variable",),
        format_paths: str | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write explicitly partitioned rasters; templates only name files.

        Args:
            path: Output folder or filesystem URL.
            split_item: Dimensions partitioning observations; default time.
            split_asset: Dimensions partitioning assets within an Item;
                "variable" gives each band its own asset; () keeps one
                multiband file.
            format_paths: Relative filename template using scalar coordinates,
                asset, var, and path. None uses time folders when dated.
            overwrite: Replace existing files.
            compute: False returns one delayed batch without reading pixels.
            storage_options: Filesystem options.
            **options: Raster creation options.

        Returns:
            Written paths, or a native delayed batch.
        """
        from geosave.geodata.stac.publish import split_items
        from geosave.geodata.io.formats import geotiff

        self.validate()
        items = split_items(self._read_rasters_by_path(), split_item, split_asset)
        return geotiff.write_items(
            items,
            path,
            format_paths=format_paths,
            writer=geotiff.write_cog,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )

    def to_zarr(
        self,
        path: Location,
        *,
        compute: bool = True,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> UPath | Delayed:
        """Write this band as one Zarr store, every date in it.

        Args:
            path: The store, ending in .zarr.
            compute: False returns one native `dask.Delayed`; computing it
                writes the store and returns its path.
            overwrite: Replace an existing store.
            storage_options: Filesystem options merged into the destination UPath.
            **write_options: Supported xarray Zarr write options.

        Returns:
            Path of the store written, or one delayed result.

        Raises:
            ValueError: The destination does not end in .zarr.
            KeyError: An encoding name is no variable of the band.
            FileExistsError: The store exists and overwrite is false.

        Examples:
            >>> ds["red"].gs.to_zarr("red.zarr")
        """
        from geosave.geodata.io.formats import zarr

        return zarr.write_zarr(
            self.to_dataset(
                dim=BAND_DIMENSION if BAND_DIMENSION in self._data.dims else None
            ),
            path,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )

    def to_netcdf(
        self,
        path: Location,
        *,
        compute: bool = True,
        engine: NetCDFEngine = "netcdf4",
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> UPath | Delayed:
        """Write this band as one netCDF file, every date in it.

        Args:
            path: The file, ending in .nc, .nc4 or .cdf.
            compute: False returns one native `dask.Delayed`; computing it
                writes the file and returns its path.
            engine: Concrete xarray netCDF backend.
            overwrite: Replace an existing file.
            storage_options: Filesystem options merged into the destination UPath.
            **write_options: Supported xarray netCDF write options.

        Returns:
            Path of the file written, or one delayed result.

        Raises:
            ValueError: The destination does not end in a netCDF suffix.
            KeyError: An encoding name is no variable of the band.
            FileExistsError: The file exists and overwrite is false.

        Examples:
            >>> ds["red"].gs.to_netcdf("red.nc")
        """
        from geosave.geodata.io.formats import netcdf

        return netcdf.write_netcdf(
            self.to_dataset(
                dim=BAND_DIMENSION if BAND_DIMENSION in self._data.dims else None
            ),
            path,
            compute=compute,
            engine=engine,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )

    def to_gtiff(
        self,
        path: Location,
        *,
        split_item: Sequence[str] = ("time",),
        split_asset: Sequence[str] = ("variable",),
        format_paths: str | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[GTiffWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write explicitly partitioned rasters; templates only name files.

        Args:
            path: Output folder or filesystem URL.
            split_item: Dimensions partitioning observations; default time.
            split_asset: Dimensions partitioning assets within an Item;
                "variable" gives each band its own asset; () keeps one
                multiband file.
            format_paths: Relative filename template using scalar coordinates,
                asset, var, and path. None uses time folders when dated.
            overwrite: Replace existing files.
            compute: False returns one delayed batch without reading pixels.
            storage_options: Filesystem options.
            **options: Raster creation options.

        Returns:
            Written paths, or a native delayed batch.
        """
        from geosave.geodata.stac.publish import split_items
        from geosave.geodata.io.formats import geotiff

        self.validate()
        items = split_items(self._read_rasters_by_path(), split_item, split_asset)
        return geotiff.write_items(
            items,
            path,
            format_paths=format_paths,
            writer=geotiff.write_geotiff,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )

    def vectorize(
        self,
        *,
        value_name: str = "value",
        mask: xr.DataArray | np.ndarray | None = None,
        connectivity: Literal[4, 8] = 4,
    ) -> GeoDataFrame:
        """Polygonize contiguous values of this flag plane.

        This computes lazy flags, because geometry depends on their values.

        Args:
            value_name: Property column receiving each region's value.
            mask: Optional exact-grid mask selecting additional valid pixels.
            connectivity: Four- or eight-neighbour region connectivity.

        Returns:
            One row per contiguous region, in this array's CRS.

        Raises:
            ValueError: The array, mask, name, connectivity or dtype is
                unsuitable for polygonization.

        Examples:
            >>> prediction.gs.vectorize(value_name="class")["class"].tolist()
            [1, 2]
        """

        from .transform.vector import vectorize

        return vectorize(
            self._data, value_name=value_name, mask=mask, connectivity=connectivity
        )

    def to_dataset(self, dim: str | None = None) -> Dataset:
        """Wrap this named array, or split a labelled dimension into variables.

        Args:
            dim: Dimension whose labels name separate variables. None wraps
                the whole array as one variable, preserving all dimensions.

        Returns:
            Lazy Dataset with this array's attrs on each variable. Splitting
            cannot recover per-variable attrs discarded during stacking.

        Raises:
            ValueError: The array has no name when wrapping, or the split
                dimension is absent or has no coordinate labels.

        Examples:
            >>> wrapped = image.gs.to_dataset()
            >>> bands = image.gs.to_dataset(dim="band")
        """
        if dim is not None and dim not in self._data.dims:
            raise ValueError(f"array has no {dim!r} dimension to split")
        if dim is not None and dim not in self._data.coords:
            raise ValueError(
                f"this array spans {self._data.sizes[dim]} entries on {dim!r} but "
                "labels none of them; assign a coordinate first"
            )
        dataset = cast("Dataset", self._data.to_dataset(dim=dim))
        if dim is not None:
            dataset.attrs = {}
            for band in dataset.data_vars.values():
                band.attrs = dict(self._data.attrs)

        return dataset

    def statistics(self) -> pd.DataFrame:
        """Summarise present pixels, with one table row per band.

        NaN and the fill value are excluded. This eagerly reads every band. A
        `band` axis is split through `to_dataset(dim="band")`, every band taking the
        array's own attrs, so a fill value counts only where all bands share
        it. Other axes are reduced together.

        Returns:
            DataFrame with minimum, maximum, mean, population stddev, and
            valid_percent columns. Rows follow band order; without a band
            axis, the single row uses the array's name, or None when unnamed.

        Raises:
            ValueError: A band holds no present pixels, or the band axis
                carries no labels to identify its bands.

        Examples:
            >>> ds["B04"].gs.statistics()
                 minimum  maximum  mean  stddev  valid_percent
            B04      1.0     63.0  32.0    18.2           98.4
        """
        if BAND_DIMENSION in self._data.dims:
            return self.to_dataset(dim=BAND_DIMENSION).gs.statistics()
        return statistics([{self._data.name: self._data}])

    def class_percent(self) -> pd.Series:
        """Give the share of present pixels in each class, reading the band.

        Returns:
            Percent per class name in legend order, summing to 100.

        Raises:
            ValueError: The band states no class names.

        Examples:
            >>> ds["label"].gs.class_percent()
            water    12.5
            trees    87.5
            Name: percent, dtype: float64
        """
        return class_percent([self._data])

    def colorize(self) -> DataArray:
        """Bake the class colours this band carries into display channels.

        Reads `Legend`, colouring each pixel by the class its code names. The
        result holds colour rather than codes, so it draws without a legend;
        prefer `plot` where a named colorbar is wanted.

        Returns:
            Georeferenced array shaped `(*axes, band, y, x)` valued in
            `[0, 1]`, whose `band` coordinate is `("red", "green", "blue")`.
            A pixel no class names is absent on every channel.

        Raises:
            ValueError: The band lists no classes, or names a class carrying
                no colour.

        Examples:
            >>> ds["landcover"].gs.colorize().sizes["band"]
            3
        """

        from geosave.geodata.conventions import SPATIAL_DIMENSIONS
        from geosave.geodata.utils.color import parse_color

        variable = self._data
        legend = variable.gs.legend
        if legend is None:
            raise ValueError(
                "the variable lists no classes, so its values name none to colour; "
                "write a Legend, or compose channels with GeoDataset.to_dataarray"
            )
        colors = legend.colors
        codes = np.array(sorted(colors))
        # One row per class, its red, green and blue in [0, 1].
        palette = np.array(
            [parse_color(colors[code]) for code in codes], dtype="float32"
        )
        palette /= 255.0

        def look_up(pixels: np.ndarray) -> np.ndarray:
            named = np.isin(pixels, codes)
            rows = np.where(named, np.searchsorted(codes, pixels), 0)
            return np.where(named[..., None], palette[rows], np.float32(np.nan))

        channels = xr.apply_ufunc(
            look_up,
            variable,
            output_core_dims=[[BAND_DIMENSION]],
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {BAND_DIMENSION: 3}},
            keep_attrs=False,
        )
        return (
            channels.assign_coords(variable.coords)
            .assign_coords({BAND_DIMENSION: ["red", "green", "blue"]})
            .transpose(*variable.gs.axes, BAND_DIMENSION, *SPATIAL_DIMENSIONS)
            .rename(None)
        )

    def explore(
        self,
        *,
        basemap: str | None = None,
        cmap: str | None = None,
        alpha: float = 0.8,
    ) -> Explorer:
        """Open this band on a web map with a sidebar to explore it.

        A band with several dates shows one at a time, chosen in the sidebar.

        Args:
            basemap: Basemap under the layers, as `viz.explore.page.map.BASEMAPS` names it.
                None uses the light basemap.
            cmap: Colormap, unless the band lists classes.
            alpha: Opacity of this band over the basemap.

        Returns:
            Explorer, shown by a notebook or served with `.servable()`.

        Examples:
            >>> ds["label"].gs.explore() * fields
        """
        from geosave.geodata.viz.explore import Explorer, RasterLayer

        layer = RasterLayer(self._data, cmap=cmap, alpha=alpha)
        return Explorer(layers=[layer], basemap=basemap)

    def plot(
        self,
        *,
        cmap: str | Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        cols: int = 4,
        title: str | None = None,
        xlabel: str | None = None,
    ) -> hv.Element | hv.Layout:
        """Draw this band through a colormap, or its classes in their colours.

        Draws the values as they stand, so decode a packed band first. A band
        listing classes in `Legend` draws each class in its colour, named in
        a shared legend to the figure's right. Repeated names and colours
        appear once.

        Args:
            cmap: Colormap. Refused for a band listing classes, whose palette
                is `Legend.color_map`.
            clim: Colour limits. Refused for a band listing classes.
            cols: Columns a `time` axis lays panels into. Ignored otherwise.
            title: Panel title above the axes. None suppresses automatic
                titles.
            xlabel: Location caption below the axes. None uses the band's
                reverse-geocoded location; an empty string suppresses it.

        Returns:
            Image, or RGB with a legend for classes, over this band's grid; or
            a Layout of one panel per `time` value, `cols` wide.

        Raises:
            ValueError: The band carries no locatable grid, `cmap` or `clim`
                is given for a band listing classes, or its attrs contradict
                each other.

        Examples:
            >>> ds["ndvi"].gs.plot(cmap="YlGn")
            >>> ds["ndvi"].gs.plot(cols=6)  # one panel per bucket
        """
        from geosave.geodata.viz.plot import draw_raster

        name = str(self._data.name) if self._data.name is not None else "value"
        return draw_raster(
            self._data.to_dataset(name=name),
            name,
            cmap=cmap,
            clim=clim,
            cols=cols,
            title=title,
            xlabel=xlabel,
        )
