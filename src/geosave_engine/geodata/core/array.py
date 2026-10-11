"""The `gs` xarray DataArray accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr

import geosave_engine.geodata.attrs as attrs

from .base import GeoRasterAccessor, as_tensor
from geosave_engine.geodata.utils.statistics import statistics
from geosave_engine.geodata.conventions import (
    BAND_DIMENSION,
    SPATIAL_DIMENSIONS,
)

if TYPE_CHECKING:
    from geosave_engine.geodata.io.raster.cogs import CogFile
    from geosave_engine.geodata.utils.color import Palette
    import panel as pn
    import torch
    from collections.abc import Mapping, Sequence
    from os import PathLike
    from upath import UPath

    import holoviews as hv
    from numpy.typing import DTypeLike

    from geosave_engine.geodata.io.storage import StorageOptions
    from geosave_engine.geodata.io.raster.geotiff import (
        GTiffWriteOptions,
    )

    from geosave_engine.geodata import DataArray, Dataset, GeoDataFrame


@xr.register_dataarray_accessor("gs")
class GeoArray(GeoRasterAccessor["DataArray"]):
    """Read and transform one raster DataArray.

    A DataArray is the unit a panel draws, so this accessor carries what
    drawing needs and leaves whole-Dataset operations to `GeoRaster`.

    Args:
        data: DataArray holding one band.

    Examples:
        >>> ds["ndvi"].gs.plot(cmap="RdYlGn")
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
        found = attrs.parse(self._data, attrs.Nodata)
        return None if found is None else found.value

    @property
    def legend(self) -> attrs.Legend | None:
        """Read what this band's values mean, or None where they are no classes."""
        return attrs.parse(self._data, attrs.Legend)

    @property
    def packing(self) -> attrs.Packing | None:
        """Read how this band's stored values decode, or None where they are physical."""
        return attrs.parse(self._data, attrs.Packing)

    @property
    def spectral(self) -> attrs.Spectral | None:
        """Read where in the spectrum this band was measured, or None."""
        return attrs.parse(self._data, attrs.Spectral)

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
        stated = None if value is None else attrs.Nodata(value=value)
        return attrs.stamp(self._data, attrs.Nodata, stated)

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
        stated = (
            None
            if class_map is None
            else attrs.Legend(classes=class_map, color_map=color_map)
        )
        return attrs.stamp(self._data, attrs.Legend, stated)

    def write_packing(
        self, scale_factor: float | None, add_offset: float = 0.0
    ) -> DataArray:
        """Write how this band's stored values decode to physical values.

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
        stated = (
            None
            if scale_factor is None
            else attrs.Packing(scale_factor=scale_factor, add_offset=add_offset)
        )
        return attrs.stamp(self._data, attrs.Packing, stated)

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
        stated = (
            attrs.Spectral(**fields)
            if any(value is not None for value in fields.values())
            else None
        )
        return attrs.stamp(self._data, attrs.Spectral, stated)

    def to_numpy(self, *, dtype: DTypeLike | None = None) -> np.ndarray:
        """Read this band as one model-input array.

        The grid pair trails and a `band` axis sits just ahead of it, matching
        what `GeoRaster.to_numpy` builds by stacking variables. Other axes keep
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
        storage_options: StorageOptions | None = None,
        **keywords: Any,
    ) -> tuple[CogFile, ...]:
        """Write this band as Cloud Optimized GeoTIFFs, as its path template names.

        Written as `Dataset.gs.to_cog` writes the one-band raster this band
        is; `{band}` is this band's name.

        Args:
            path: Local path or fsspec URL ending in `.tif`, a template as
                `Dataset.gs.to_cog` takes it.
            map_scale: Map denominator used to write pixels per centimetre.
            overwrite: Replace an existing file when true.
            storage_options: Options for the filesystem a URL names.
            **keywords: COG creation options and template fields.

        Returns:
            One `CogFile` per file written, in time order.

        Raises:
            ValueError: This array names none of the bands it spans or carries
                no `.name`, or the path cannot be filled as
                `Dataset.gs.to_cog` explains.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> ds["ndvi"].gs.to_cog("ndvi_{time:%Y%m%d}.tif")[0].bands
            ('ndvi',)
        """
        return self.to_raster().gs.to_cog(
            path,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **keywords,
        )

    def to_gtiff(
        self,
        path: str | PathLike[str],
        *,
        map_scale: float | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[GTiffWriteOptions],
    ) -> UPath:
        """Write this band as a plain GeoTIFF.

        Reach for `to_cog` unless a consumer needs a striped or otherwise
        non-COG file.

        Args:
            path: Output path ending in `.tif` or `.tiff`.
            map_scale: Map denominator used to write pixels per centimetre.
            overwrite: Replace an existing file when true.
            storage_options: Options for the filesystem a URL names.
            **options: GTiff creation options.

        Returns:
            The written path.

        Raises:
            ValueError: This array names none of the bands it spans, carries
                no `.name` to write a sole band under, or carries a `time`
                dimension rather than a scalar coordinate.
            FileExistsError: `path` exists and `overwrite` is false.
        """
        from geosave_engine.geodata.io.raster.geotiff import write_gtiff

        return write_gtiff(
            self.to_raster(),
            path,
            map_scale=map_scale,
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
        from geosave_engine.geodata.transform.vector import vectorize

        return vectorize(
            self._data, value_name=value_name, mask=mask, connectivity=connectivity
        )

    def to_raster(self) -> Dataset:
        """Split this array back into the raster whose variables it stacks.

        The inverse of `GeoRaster.to_array`: each band becomes one variable,
        carrying the attrs that ride on the `band` coordinate. An array
        spanning no band axis becomes a one-variable raster named after it.

        Returns:
            Dataset holding one variable per band, each carrying this array's
            attrs, which every band of it shares.

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
        return statistics({self._data.name: self._data})

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
        from geosave_engine.geodata.transform import color

        return cast("DataArray", color.colorize(self._data))

    def explore(self, *, alpha: float = 0.7) -> pn.pane.HoloViews:
        """Draw this band over a web map, the way `plot` draws it.

        Needs the `viz` extra. The map shows one date; select it with
        `.isel(time=...)` first where there are several.

        Args:
            alpha: Opacity of the band over the map.

        Returns:
            Panel pane drawing the band over OpenStreetMap with bokeh.

        Raises:
            ValueError: This band holds more than one date.

        Examples:
            >>> ds["label"].gs.explore()
        """
        from geosave_engine.geodata.viz import explore

        return explore(self._data, lambda band: band.gs.plot(xlabel=""), alpha=alpha)

    def plot(
        self,
        *,
        cmap: str | Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        cols: int = 4,
        title: str | None = None,
        xlabel: str | None = None,
    ) -> hv.Element | hv.NdLayout:
        """Draw this band as an Image. Needs the `viz` extra.

        Draws the values as they stand, so decode a packed band first. A band
        listing classes in `Legend` draws through that palette with them
        named on the colorbar, while `viz.continuous` draws the bare codes.

        Args:
            cmap: Colormap. Refused for a band carrying a class map, whose
                palette is `Legend.color_map`.
            clim: Colour limits. Refused for a band carrying a class map,
                whose limits follow the class count.
            cols: Columns a `time` axis lays panels into. Ignored otherwise.
            title: Panel title above the axes. None suppresses automatic
                titles.
            xlabel: Location caption below the axes. None uses the band's
                reverse-geocoded location; an empty string suppresses it.

        Returns:
            Image over this band's grid, or a Layout of one panel per `time`
            value, `cols` wide.

        Raises:
            ValueError: The band carries no locatable grid, `cmap` or `clim`
                is given for a band carrying a class map, or its attrs
                contradict each other.

        Examples:
            >>> ds["ndvi"].gs.plot(cmap="RdYlGn")
            >>> ds["ndvi"].gs.plot(cols=6)  # one panel per bucket
        """
        from geosave_engine.geodata.viz import plot

        legend = self.legend
        place = self.anchor.locate() if xlabel is None else None
        caption = place.to_address() if place is not None else xlabel
        return plot(
            self._data,
            cmap=cmap,
            clim=clim,
            class_map=legend.classes if legend else None,
            color_map=legend.color_map if legend else None,
            cols=cols,
            title=title,
            xlabel=caption,
        )
