"""The `gs` xarray DataArray accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import (
    BAND_DIMENSION,
    SPATIAL_DIMENSIONS,
)
from geosave.geodata.utils.summary import class_percent, statistics
from geosave.geodata.utils.tensor import as_tensor

from geosave.geodata.conventions import ITEM_ID_FORMAT

from .base import GeoRaster

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike

    import pystac

    from odc.geo import SomeCRS
    from odc.geo.gcp import GCPGeoBox
    from odc.geo.geobox import GeoBox

    from geosave.geodata import DataArray, Dataset, DataTree, GeoDataFrame
    from geosave.geodata.stac.item import ItemTime
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


class GeoDataArray(GeoRaster["DataArray"]):
    """Read and transform one raster DataArray.

    A DataArray is the unit a panel draws, so this accessor carries what
    drawing needs and leaves whole-Dataset operations to `GeoDataset`.

    Args:
        data: DataArray holding one band.

    Examples:
        >>> ds["ndvi"].gs.plot(cmap="YlGn")
    """

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
        return {
            str(dim): self._data.coords[dim].values
            if dim in self._data.coords
            else None
            for dim in self._data.dims
            if dim not in SPATIAL_DIMENSIONS
        }

    @property
    def nodata(self) -> float | int | None:
        """Read the stored value standing for nodata, or None where none is set."""
        found = attrs.Nodata.from_attrs(self._data.attrs)
        return None if found is None else found.fill_value

    @property
    def legend(self) -> attrs.Legend | None:
        """Read what this band's values mean, or None where they are no classes."""
        return attrs.Legend.from_attrs(self._data.attrs)

    @property
    def is_categorical(self) -> bool:
        """Say whether values are class codes that blending or ranking would destroy.

        Dtype does not say: Sentinel-2 reflectance and a land cover map are
        both integers. A legend key or a GDAL palette marks the values as
        codes. A half-spelled legend still answers; a `colorinterp` GDAL
        does not know raises.

        Returns:
            True when the attrs state a legend or `colorinterp="palette"`;
            otherwise False. Pixel values and dtype are not inspected.

        Examples:
            >>> scene.landcover.gs.is_categorical, scene.red.gs.is_categorical
            (True, False)
        """
        attributes = self._data.attrs
        if attrs.Legend.is_stated_in(attributes):
            return True
        gdal_band = attrs.GDALBand.from_attrs(attributes)
        return gdal_band is not None and gdal_band.colorinterp == "palette"

    @property
    def packing(self) -> attrs.Packing | None:
        """Read how this band's stored values decode, or None where they are physical."""
        return attrs.Packing.from_attrs(self._data.attrs)

    @property
    def spectral(self) -> attrs.Spectral | None:
        """Read where in the spectrum this band was measured, or None."""
        return attrs.Spectral.from_attrs(self._data.attrs)

    def write_grid(self, geobox: GeoBox | GCPGeoBox) -> DataArray:
        """Place this band on a grid: `y` and `x` coordinates and the CRS.

        The usual way back from a model: wrap its output in `xr.DataArray`,
        then place it on the chip it came from.

        Args:
            geobox: Regular grid, or a grid placed by ground control points.

        Returns:
            New DataArray pointing at the CRS coordinate, its axes carrying CF
            attrs on a regular grid. Coordinates on other dimensions are kept.

        Raises:
            ValueError: The dimensions do not end in `y, x`, their sizes
                differ from the grid's, or the grid carries no CRS.

        Examples:
            >>> pred = xr.DataArray(logits.argmax(0), dims=("y", "x"), name="label")
            >>> pred.gs.write_grid(chip.gs.geobox).gs.write_nodata(255).gs.geobox == chip.gs.geobox
            True
        """
        from .grid import place_on_grid

        return cast("DataArray", place_on_grid(self._data, geobox))

    def write_crs(self, crs: SomeCRS | None = None) -> DataArray:
        """Write onto the existing spatial coordinates what they measure.

        Args:
            crs: CRS the existing coordinates are in. None reads the one the
                band already carries.

        Returns:
            New DataArray whose spatial coordinates carry the CRS and CF attrs.

        Raises:
            ValueError: The band has no `y`/`x` coordinates, carries no CRS and
                `crs` names none, or is placed by ground control points.

        Examples:
            >>> band.gs.write_crs("EPSG:32633").y.attrs["axis"]
            'Y'
        """
        from .grid import write_reference

        return cast("DataArray", write_reference(self._data, crs))

    def to_datatree(self, collection: str) -> DataTree:
        """Make this band one collection's grid node, as `Dataset.gs.to_datatree` does.

        Args:
            collection: Name of the collection node.

        Returns:
            DataTree of one collection node over one grid node.

        Raises:
            ValueError: The band carries no name or sits on no georeferenced grid.

        Examples:
            >>> list(ndvi.gs.to_datatree("ndvi").gs.rasters)
            ['ndvi/r10m']
        """
        return self.to_raster().gs.to_datatree(collection)

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
        if value is None:
            return attrs.clear_attrs(self._data, attrs.Nodata)
        nodata = attrs.Nodata(fill_value=value)
        nodata.check_dtype(self._data.dtype)
        return attrs.stamp_attrs(self._data, nodata)

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

    def write_packing(
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
            >>> ds["B04"].gs.write_packing(1e-4, -0.1).gs.packing.scale_factor
            0.0001
        """
        if scale_factor is None:
            return attrs.clear_attrs(self._data, attrs.Packing)
        packing = attrs.Packing(scale_factor=scale_factor, add_offset=add_offset)
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
        # Put spatial dims (y, x) last
        ahead = [axis for axis in self.axes if axis != BAND_DIMENSION]
        if BAND_DIMENSION in self._data.dims:
            ahead.append(BAND_DIMENSION)

        ordered = self._data.transpose(*ahead, *SPATIAL_DIMENSIONS).values
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

    def to_catalog(
        self,
        path: Location,
        *,
        path_format: str | None = None,
        collection: pystac.Collection | str | None = None,
        id_format: str = ITEM_ID_FORMAT,
        datetime: ItemTime | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Any,
    ) -> GeoDataFrame:
        """Write this band's files and return their STAC Items as an item table.

        Written as `Dataset.gs.to_catalog` writes the raster this band is. A
        band keeps its coordinates, so `stac_properties`, but not the root
        attrs holding the Collection: name it with `collection`.

        Args:
            path: Folder the files go under.
            path_format: Relative path, as `Dataset.gs.to_catalog` takes it.
            collection: Collection, or its id, the Items belong to.
            id_format: Item id template, as `Dataset.gs.to_catalog` takes it.
            datetime: Date or interval for a band stating no time.
            overwrite: Replace files that exist.
            storage_options: Credentials and connection options for the files.
            **options: GeoTIFF creation options.

        Returns:
            Item table, one row per date.

        Examples:
            >>> ndvi.gs.to_catalog("out", path_format="ndvi_{time:%Y%m%d}.tif", collection="ndvi")
        """
        return self.to_raster().gs.to_catalog(
            path,
            path_format=path_format,
            collection=collection,
            id_format=id_format,
            datetime=datetime,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )

    def to_cog(
        self,
        path: Location,
        *,
        path_format: str | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write this band as Cloud Optimized GeoTIFFs under the folder `path`.

        A GeoTIFF holds one grid at one date, so `path_format` lays the raster
        out: its fields split one axis each, `{time}` into dates, `{band}` into
        variables. An axis without a field stays whole in one file, which
        several dates cannot share. Without `path_format` every band of every
        date is its own file, `{time:%Y%m%dT%H%M%S}/{band}.tif`.

        Args:
            path: Folder or URL prefix the files go under.
            path_format: Relative path filled with `str.format`, ending in
                .tif. `{time}` takes strftime directives.
            compute: False returns one native `dask.Delayed`; computing it
                writes all outputs and returns their paths.
            overwrite: Replace existing destinations.
            storage_options: Filesystem options merged into the destination UPath.
            **options: GeoTIFF creation options.

        Returns:
            Written paths in time then variable order, or one delayed result.

        Raises:
            ValueError: `path` names a file rather than a folder, a path format
                is not relative, destinations collide, or a raster cannot be
                one GeoTIFF.
            KeyError: A path format field has no value.
            FileExistsError: A destination exists and overwrite is false.

        Examples:
            >>> paths = ds["red"].gs.to_cog("out", path_format="{time:%Y%m%d}.tif")
        """
        return self.to_raster().gs.to_cog(
            path,
            path_format=path_format,
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

        return zarr.write_zarr(
            self.to_raster(),
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
            self.to_raster(),
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
        path_format: str | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[GTiffWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write this band as plain GeoTIFFs under the folder `path`.

        Laid out as `to_cog` lays it out; the files are not cloud-optimized.

        Args:
            path: Folder or URL prefix the files go under.
            path_format: Relative path filled with `str.format`, ending in
                .tif or .tiff. None writes one file per date.
            compute: False returns one native `dask.Delayed`; computing it
                writes all outputs and returns their paths.
            overwrite: Replace existing destinations.
            storage_options: Filesystem options merged into the destination UPath.
            **options: GeoTIFF creation options.

        Returns:
            Written paths in time order, or one delayed result.

        Raises:
            ValueError: `path` names a file rather than a folder, a path format
                is not relative, destinations collide, or a raster cannot be
                one GeoTIFF.
            FileExistsError: A destination exists and overwrite is false.

        Examples:
            >>> paths = ds["red"].gs.to_gtiff("out", path_format="{time:%Y%m%d}.tif")
        """
        from geosave.geodata.io.formats import geotiff
        from geosave.geodata.io.paths import expand_paths
        from geosave.geodata.io.storage import location

        files = expand_paths(
            self.to_raster(), location(path, storage_options), path_format
        )
        return geotiff.write_files(
            files, geotiff.write_geotiff, compute=compute, overwrite=overwrite, **options
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
        from geosave.geodata.transform.vector import vectorize

        return vectorize(
            self._data, value_name=value_name, mask=mask, connectivity=connectivity
        )

    def to_raster(self) -> Dataset:
        """Split this array back into the raster whose variables it stacks.

        Each band becomes one variable carrying a copy of this array's attrs.
        It undoes `GeoDataset.to_array` only for the attrs every band shared:
        a nodata, packing, spectral, or legend that differed between bands
        was not kept by `to_array` and does not come back. Decode with
        `mask_and_scale` before stacking where per-band nodata matters. An
        array spanning no band axis becomes a one-variable raster named after
        it.

        Returns:
            Dataset holding one variable per band, each carrying this array's
            attrs.

        Raises:
            ValueError: This array spans a `band` axis carrying no labels to
                name its variables, or spans none and carries no `.name`.

        Examples:
            >>> ds.gs.to_array().gs.to_raster()
            >>> ds["ndvi"].gs.to_raster()
        """
        if BAND_DIMENSION not in self._data.dims:
            if self._data.name is None:
                raise ValueError("array carries no name to write its band under")
            return cast("Dataset", self._data.to_dataset())

        if BAND_DIMENSION not in self._data.coords:
            raise ValueError(
                f"this array spans {self._data.sizes[BAND_DIMENSION]} bands but "
                f"labels none of them, so they name no variables; assign a "
                f"{BAND_DIMENSION!r} coordinate first"
            )

        raster = cast("Dataset", self._data.to_dataset(dim=BAND_DIMENSION))
        raster.attrs = {}
        for band in raster.data_vars.values():
            band.attrs = dict(self._data.attrs)
        return raster

    def statistics(self) -> pd.DataFrame:
        """Summarise present pixels, with one table row per band.

        NaN and the fill value are excluded. This eagerly reads every band. A
        `band` axis is split through `to_raster`, every band taking the
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
            return self.to_raster().gs.statistics()
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
        from geosave.geodata.transform import color

        return cast("DataArray", color.colorize(self._data))

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
        return Explorer(layer, basemap=basemap)

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
