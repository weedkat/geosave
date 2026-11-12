"""The `gs` xarray DataArray accessor."""

from __future__ import annotations


from typing import Any, TYPE_CHECKING, Literal, Unpack, cast

import cf_xarray  # noqa: F401  — registers the .cf accessor
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

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from geosave.geodata.utils.datetime import Freq
    from odc.geo import SomeResolution
    from .transform.warp import Resampling
    from .transform.extent import Padding, Region
    import pystac
    from odc.geo.geom import Geometry
    from geosave.geodata.core.region import GeoRegion
    from collections.abc import Mapping, Sequence

    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike

    from odc.geo import SomeCRS
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
        if attrs.CFPacking.is_stated_in(data.attrs):
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
        """Find spatial dimension names in vertical, horizontal order, as ODC does.

        Returns:
            ``("y", "x")``, ``("latitude", "longitude")`` or another pair ODC
            recognizes, or None when it finds none.

        Examples:
            >>> scene["red"].gs.spatial_dims
            ('y', 'x')
        """
        return self._data.odc.spatial_dims

    @property
    def geobox(self) -> GeoBox | None:
        """Read this array's grid as ODC infers it, or None when it carries none.

        A grid may exist without a CRS.

        Raises:
            NotImplementedError: Pixels are placed by ground control points.
        """

        from odc.geo.gcp import GCPGeoBox

        grid = self._data.odc.geobox
        if isinstance(grid, GCPGeoBox):
            raise NotImplementedError(
                "Rasters placed by ground control points are not supported yet. "
                "Reproject onto a GeoBox with raster.odc.reproject(...) first."
            )
        return grid

    @property
    def nodata(self) -> float | int | None:
        """Read the active scalar missing-value code, or None when none is declared."""
        found = attrs.CFPacking.from_attrs(self._data.attrs)
        if found is None:
            return None
        if found.fill_value is not None:
            return found.fill_value
        if isinstance(found.missing_value, (int, float)):
            return found.missing_value
        return None

    @property
    def flags(self) -> attrs.CFFlags | None:
        """Read what this band's values mean as CF flags, or None where they are no classes."""
        return attrs.CFFlags.from_attrs(self._data.attrs)

    @property
    def display(self) -> attrs.Display | None:
        """Read how this band draws: its class colours, or None."""
        return attrs.Display.from_attrs(self._data.attrs)

    @property
    def class_colors(self) -> dict[int, str] | None:
        """Read the colour of every class, in class order, or None without classes.

        Examples:
            >>> ds["label"].gs.class_colors
            {0: '#1F77B4', 1: '#0000FF'}
        """
        flags = self.flags
        if flags is None or flags.classes is None:
            return None
        return (self.display or attrs.Display()).colors(flags.classes)

    @property
    def is_categorical(self) -> bool:
        """Say whether values are class codes that blending or ranking would destroy.

        Dtype does not say: Sentinel-2 reflectance and a land cover map are
        both integers. CF flags (`flag_values` or `flag_masks` with
        `flag_meanings`) mark the values as codes, and so does a colour map.

        Returns:
            True for a CF flag variable or one carrying a colour map; otherwise
            False. Pixel values and dtype are not inspected.

        Examples:
            >>> scene.landcover.gs.is_categorical, scene.red.gs.is_categorical
            (True, False)
        """
        if self.flags is not None:
            return True
        display = self.display
        return display is not None and display.color_map is not None

    @property
    def description(self) -> attrs.CFDescription | None:
        """Read what this variable's values are (`long_name`, `standard_name`, `units`), or None."""
        return attrs.CFDescription.from_attrs(self._data.attrs)

    @property
    def packing(self) -> attrs.CFPacking | None:
        """Read this variable's active CF scale, offset and missing-value declarations."""
        return attrs.CFPacking.from_attrs(self._data.attrs)

    @property
    def spectral(self) -> attrs.EOBand | None:
        """Read where in the spectrum this band was measured, or None."""
        return attrs.EOBand.from_attrs(self._data.attrs)

    def reproject(
        self,
        how: GeoBox | SomeCRS,
        *,
        resampling: Resampling = "nearest",
        resolution: SomeResolution | None = None,
    ) -> DataArray:
        """Warp pixels onto the grid a destination names.

        A destination grid is adopted whole — CRS, resolution, and extent — while a
        bare CRS only decides the projection, sizing the grid from this object.
        Output spatial axes are named by the destination grid, as ODC names
        them (``y/x`` projected, ``latitude/longitude`` geographic), and follow
        the other dimensions.
        Declared nodata is preserved. Otherwise ODC supplies its default fill;
        use ``gs.write_nodata`` beforehand to choose a value explicitly.

        Args:
            how: Exact GeoBox or CRS to land on. Use reference.gs.geobox
                to match an existing raster.
            resampling: One ODC resampling method for the whole object.
            resolution: Output pixel size for a CRS destination, ignored for a
                GeoBox as in ODC. None keeps the ground sampling as closely as
                the new CRS allows. ODC sizes a CRS destination from the
                transformed footprint, so a new pixel size in the same CRS may
                gain a padded border; pass ``geobox.zoom_to(resolution=...)``
                for an exact one.

        Returns:
            New object on the destination grid, carrying its own `spatial_ref`.

        Raises:
            ValueError: This object sits on no locatable grid, a variable does
                not span the grid, or `resampling` would blend class codes.

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

        Unlike `reproject`, nothing is resampled or computed.

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

        extent = attrs.ACDDExtent.from_attrs(self._data.attrs)
        if extent is not None:
            return GeoRegion(extent.to_geometry(), timespan=self.timespan)
        self.validate()
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
            return attrs.clear_attrs(self._data, attrs.ACDDExtent)
        return attrs.stamp_attrs(self._data, attrs.ACDDExtent.from_geometry(extent))

    def clip_extent(self, geometry: Geometry) -> DataArray:
        """Narrow the declared extent to a geometry, as a crop narrows coverage.

        Args:
            geometry: CRS-aware geometry the pixels now cover at most.

        Returns:
            Shallow copy whose declared extent is the intersection, cleared
            when the two do not meet. Without a declared extent, unchanged.
        """
        extent = attrs.ACDDExtent.from_attrs(self._data.attrs)
        if extent is None:
            return cast("DataArray", self._data)
        clipped = extent.clip(geometry)
        if clipped is None:
            return attrs.clear_attrs(self._data, attrs.ACDDExtent)
        return attrs.stamp_attrs(self._data, clipped)

    def write_nodata(self, value: float | int | None) -> DataArray:
        """Write the stored value standing for this band's nodata pixels.

        A shorthand for the fill of `write_packing`, keeping scale and offset.
        It is written as CF's `_FillValue`, which ODC and rioxarray read too.

        Args:
            value: Stored value marking nodata, of this band's own type. None
                clears it.

        Returns:
            New DataArray carrying `value` as its fill.

        Raises:
            ValueError: This band's dtype cannot represent `value`, such as NaN
                on integers.
            OverflowError: `value` lies outside this band's dtype.

        Examples:
            >>> ds["red"].gs.write_nodata(0).attrs["_FillValue"]
            0
        """
        fill = None if value is None else self._data.dtype.type(value).item()
        packing = self.packing or attrs.CFPacking()
        updated = packing.model_copy(update={"fill_value": fill, "missing_value": None})
        return self.write_packing(updated)

    def write_flags(
        self,
        classes: Mapping[int, str] | None,
        colors: Palette | None = None,
    ) -> DataArray:
        """Write the class each of this band's stored values stands for.

        Classes are written as CF flags, colours as the band's `Display`.

        Args:
            classes: Stored value mapped to class name. None removes the
                flags and colours.
            colors: Stored value mapped to hex or RGB colour. None writes
                names only.

        Returns:
            New DataArray carrying the flags.

        Raises:
            ValueError: `colors` colours a value `classes` does not name, or a
                value does not fit this band's dtype.

        Examples:
            >>> ds["label"].gs.write_flags({0: "water", 1: "trees"})
        """
        if classes is None:
            cleared = attrs.clear_attrs(self._data, attrs.CFFlags)
            return attrs.clear_attrs(cleared, attrs.Display)
        flags = attrs.CFFlags.model_validate({"classes": classes})
        flags.check_dtype(self._data.dtype)
        written = attrs.stamp_attrs(self._data, flags)
        if colors is None:
            return attrs.clear_attrs(written, attrs.Display)
        display = attrs.Display.model_validate({"color_map": colors})
        uncolored = sorted(set(display.color_map or {}) - set(classes))
        if uncolored:
            raise ValueError(
                f"The colours name pixel values {uncolored}, but no class has "
                f"those values (the classes are {sorted(classes)}). Add the "
                "missing classes or remove their colours."
            )
        return attrs.stamp_attrs(written, display)

    def write_packing(self, packing: attrs.CFPacking | None) -> DataArray:
        """Write how this band's stored values decode to physical values.

        Sentinel-2 L2A is the usual case: ESA stores reflectance as
        `DN * 1e-4 + offset`, with offset 0 for processing baseline below 04.00
        and -0.1 (`BOA_ADD_OFFSET` -1000) from 04.00, January 2022; see
        https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/product-formatting.

        Args:
            packing: Scale, offset and fill, as CF packs them. None removes
                every packing and missing-value attr.

        Returns:
            New DataArray carrying the packing; its pixels are unchanged.

        Examples:
            >>> packing = attrs.CFPacking(scale_factor=1e-4, add_offset=-0.1, fill_value=0)
            >>> ds["B04"].gs.write_packing(packing).gs.packing.scale_factor
            0.0001
        """
        if packing is None:
            return attrs.clear_attrs(self._data, attrs.CFPacking)
        return attrs.stamp_attrs(self._data, packing)

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
            return attrs.clear_attrs(self._data, attrs.EOBand)
        return attrs.stamp_attrs(self._data, attrs.EOBand(**fields))

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
        spatial = self._require_spatial_dims()
        # Put spatial dimensions last.
        ahead = []
        for dim in self._data.dims:
            if dim not in (*spatial, BAND_DIMENSION):
                ahead.append(dim)
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

    def mask_and_scale(self) -> DataArray:
        """Decode CF packing with xarray, keeping coordinates and lazy pixels.

        Packing declarations must use native CF names and numeric values.
        Consumed attrs move to encoding for the writer; coordinates are not decoded.
        """
        from xarray.conventions import decode_cf_variable

        array = self._data
        decoded = decode_cf_variable(
            array.name,
            array.variable.copy(deep=False),
            concat_characters=False,
            decode_times=False,
        )
        result = xr.DataArray(decoded, coords=array.coords, name=array.name)
        result.encoding = decoded.encoding
        return cast("DataArray", result)

    def to_item(self, path: Location, **options: Any) -> pystac.Item:
        """STAC Item publication is being rebuilt on `expand_stac` and `tree.gs.to_cog`.

        Raises:
            NotImplementedError: Always, until publication is rebuilt.
        """
        raise NotImplementedError(
            "STAC publication is being rebuilt on expand_stac and tree.gs.to_cog"
        )

    def to_catalog(self, path: Location, **options: Any) -> GeoDataFrame:
        """STAC catalog publication is being rebuilt on `expand_stac` and `tree.gs.to_cog`.

        Raises:
            NotImplementedError: Always, until publication is rebuilt.
        """
        raise NotImplementedError(
            "STAC publication is being rebuilt on expand_stac and tree.gs.to_cog"
        )

    def to_cog(
        self,
        path: Location,
        *,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> UPath | Delayed:
        """Write this array to one explicit GeoTIFF path.

        Args:
            path: Destination file ending in .tif or .tiff.
            overwrite: Replace an existing file.
            compute: False returns one delayed write without reading pixels.
            storage_options: Filesystem options.
            **options: Raster creation options.

        Returns:
            Written path, or a delayed path.

        Raises:
            ValueError: The array cannot be represented by one GeoTIFF.
        """
        from geosave.geodata.io.formats import geotiff

        return geotiff.write_cog(
            self._data,
            path,
            overwrite=overwrite,
            compute=compute,
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

        if BAND_DIMENSION in self._data.dims:
            dataset = self.to_dataset()
        else:
            dataset = self._data.to_dataset()
        return zarr.write_zarr(
            dataset,
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

        if BAND_DIMENSION in self._data.dims:
            dataset = self.to_dataset()
        else:
            dataset = self._data.to_dataset()
        return netcdf.write_netcdf(
            dataset,
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
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[GTiffWriteOptions],
    ) -> UPath | Delayed:
        """Write this array to one explicit GeoTIFF path.

        Args:
            path: Destination file ending in .tif or .tiff.
            overwrite: Replace an existing file.
            compute: False returns one delayed write without reading pixels.
            storage_options: Filesystem options.
            **options: Raster creation options.

        Returns:
            Written path, or a delayed path.

        Raises:
            ValueError: The array cannot be represented by one GeoTIFF.
        """
        from geosave.geodata.io.formats import geotiff

        return geotiff.write_geotiff(
            self._data,
            path,
            overwrite=overwrite,
            compute=compute,
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

    def to_dataset(self, dim: str = BAND_DIMENSION) -> Dataset:
        """Split labelled raster bands, retaining shared CF attrs on each variable.

        Args:
            dim: Dimension whose unique labels name the output variables.
                Defaults to "band". Use native `array.to_dataset(name=...)`
                to wrap an array without splitting.

        Returns:
            Lazy Dataset with the array's shared attrs on each band. Original
            per-band metadata and storage recipes lost in stacking are not recovered.

        Examples:
            >>> bands = image.gs.to_dataset()
        """
        dataset = cast("Dataset", self._data.to_dataset(dim=dim))
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

        Reads `CFFlags` and `Display`, colouring each pixel by the class its code names. The
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

        from geosave.geodata.utils.color import parse_color

        variable = self._data
        colors = variable.gs.class_colors
        if colors is None:
            raise ValueError(
                "the variable lists no classes, so its values name none to colour; "
                "write its flags with gs.write_flags, or compose channels with "
                "GeoDataset.to_array"
            )
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
            .transpose(..., BAND_DIMENSION, *variable.gs._require_spatial_dims())
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
        listing classes in `CFFlags` draws each class in its colour, named in
        a shared legend to the figure's right. Repeated names and colours
        appear once.

        Args:
            cmap: Colormap. Refused for a band listing classes, whose palette
                is `Display.color_map`.
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
