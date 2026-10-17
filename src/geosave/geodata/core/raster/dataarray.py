"""The `gs` xarray DataArray accessor."""

from __future__ import annotations

from functools import partial

from typing import TYPE_CHECKING, Literal, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr

import geosave.geodata.attrs as attrs

from geosave.geodata.utils.summary import class_percent, statistics
from geosave.geodata.utils.tensor import as_tensor

from .base import GeoRaster
from geosave.geodata.conventions import (
    BAND_DIMENSION,
    SPATIAL_DIMENSIONS,
)

if TYPE_CHECKING:
    from geosave.geodata.io.raster.geotiff import COGWriteOptions
    from geosave.geodata.utils.color import Palette
    from geosave.geodata.viz.explorer import Explorer, Overlay
    import torch
    from collections.abc import Mapping, Sequence
    from os import PathLike
    from geosave.geodata.io.raster.file import Scene

    import holoviews as hv
    from numpy.typing import DTypeLike

    from geosave.geodata.io.storage import StorageOptions
    from geosave.geodata.io.raster.geotiff import (
        GTiffWriteOptions,
    )

    from geosave.geodata import DataArray, Dataset, GeoDataFrame


@xr.register_dataarray_accessor("gs")
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
            In the band's own order, ready to pass to `array` or `raster`.

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
        return None if found is None else found.value

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
        nodata = attrs.Nodata(value=value)
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

    def to_cog(
        self,
        path: str | PathLike[str],
        *,
        map_scale: float | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[Scene]:
        """Write this band as Cloud Optimized GeoTIFFs, as its path template names.

        Written as `Dataset.gs.to_cog` writes the one-band raster this band
        is; `{band}` is this band's name.

        Args:
            path: Local path or fsspec URL ending in `.tif`, a template as
                `Dataset.gs.to_cog` takes it.
            map_scale: Map denominator used to write pixels per centimetre.
            overwrite: Replace an existing file when true.
            compute: False returns every file pending, written by one
                `dask.compute` of the result.
            storage_options: Options for the filesystem a URL names.
            **options: COG creation options.

        Returns:
            One scene per date, in time order, as `Dataset.gs.to_cog` returns
            them; each holds this band's file under its name.

        Raises:
            ValueError: This array names none of the bands it spans or carries
                no `.name`, or the path cannot be filled as
                `Dataset.gs.to_cog` explains.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> ds["ndvi"].gs.to_cog("ndvi_{time:%Y%m%d}.tif")[0]
            Scene(None, {'ndvi': ndvi_20250601.tif})
        """
        return self.to_raster().gs.to_cog(
            path,
            map_scale=map_scale,
            overwrite=overwrite,
            compute=compute,
            storage_options=storage_options,
            **options,
        )

    def to_gtiff(
        self,
        path: str | PathLike[str],
        *,
        map_scale: float | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[GTiffWriteOptions],
    ) -> Scene:
        """Write this band as a plain GeoTIFF.

        Reach for `to_cog` unless a consumer needs a striped or otherwise
        non-COG file.

        Args:
            path: Output path ending in `.tif` or `.tiff`.
            map_scale: Map denominator used to write pixels per centimetre.
            overwrite: Replace an existing file when true.
            compute: False returns the scene pending, written on `.compute()`.
            storage_options: Options for the filesystem a URL names.
            **options: GTiff creation options.

        Returns:
            One scene holding the file under this band's name, or pending.

        Raises:
            ValueError: This array names none of the bands it spans, carries
                no `.name` to write a sole band under, or carries a `time`
                dimension rather than a scalar coordinate.
            FileExistsError: `path` exists and `overwrite` is false.
        """
        from geosave.geodata.io.raster.geotiff import write_gtiff

        write = partial(
            write_gtiff, map_scale=map_scale, overwrite=overwrite, **options
        )
        return self.to_raster().gs._save_one(path, write, storage_options, compute)

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
        overlay: Overlay | None = None,
        basemap: str | None = None,
        cmap: str | None = None,
        alpha: float = 0.8,
    ) -> Explorer:
        """Open this band on a web map with a sidebar to explore it.

        A band with several dates shows one at a time, chosen in the sidebar.

        Args:
            overlay: Rasters, GeoDataFrames or item tables drawn on top, such as
                field boundaries or a prediction; a mapping names each layer.
            basemap: Basemap under the layers, as `viz.map.BASEMAPS` names it.
                None uses the light basemap.
            cmap: Colormap, unless the band lists classes.
            alpha: Opacity of this band over the basemap.

        Returns:
            Explorer, shown by a notebook or served with `.servable()`.

        Examples:
            >>> ds["label"].gs.explore(overlay=fields)
        """
        from geosave.geodata.viz.explorer import Explorer, RasterLayer, create_layers

        layer = RasterLayer(self._data, cmap=cmap, alpha=alpha)
        return Explorer(layer, *create_layers(overlay), basemap=basemap)

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
        place = self.anchor.locate() if xlabel is None else None
        caption = place.to_address() if place is not None else xlabel
        return draw_raster(
            self._data.to_dataset(name=name),
            name,
            cmap=cmap,
            clim=clim,
            cols=cols,
            title=title,
            xlabel=caption,
        )
