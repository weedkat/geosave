"""Build raster Datasets and read them through the `gs` Dataset accessor.

Georeferencing is odc-geo's. GeoSave adds only the CF axis semantics odc
omits, which `write_crs` writes.

Examples:
    A raster carrying no CRS holds pixels without claiming ground position, so
    `GeoRaster.geobox` is None, and members needing ground position refuse.
    Variable access still works:

    >>> png
    <xarray.Dataset> Size: 262kB
    Dimensions:  (y: 512, x: 512)
    Dimensions without coordinates: y, x
    Data variables:
        band     (y, x) uint8 ...

    A georeferenced one carries `spatial_ref`, which odc reads a GeoBox off:

    >>> opened
    <xarray.Dataset> Size: 2MB
    Dimensions:      (time: 2, y: 512, x: 512)
    Coordinates:
      * time         (time) datetime64[ns] 2025-06-01 2025-06-11
      * y            (y) float64 5.0051e+06 5.0051e+06 ... 5.0000e+06
      * x            (x) float64 3.0000e+05 3.0000e+05 ... 3.0511e+05
        spatial_ref  int32 32633
    Data variables:
        red          (time, y, x) uint16 ...
    >>> opened.y.attrs
    {'units': 'metre', 'resolution': -10.0, 'crs': 'EPSG:32633'}

    `write_crs` adds what those axes measure, which odc leaves out:

    >>> opened.gs.write_crs().y.attrs
    {'units': 'metre', 'resolution': -10.0, 'crs': 'EPSG:32633',
     'standard_name': 'projection_y_coordinate', 'axis': 'Y'}

    Geographic grids span `y` and `x` too. CF metadata identifies them as
    latitude and longitude:

    >>> wgs84.red.dims
    ('y', 'x')
    >>> wgs84.y.attrs["standard_name"], wgs84.y.attrs["units"]
    ('latitude', 'degrees_north')
"""

from __future__ import annotations

from upath import UPath
from typing import TYPE_CHECKING, Any, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.attrs.groups.gdal import RGB

from .base import GeoRasterAccessor, as_tensor
from .constructors import write_cf_axes
from geosave_engine.geodata.core.statistics import statistics
from geosave_engine.geodata.core.variables import variable_names
from geosave_engine.geodata.conventions import (
    BAND_DIMENSION,
    SPATIAL_DIMENSIONS,
)


if TYPE_CHECKING:
    from geosave_engine.geodata.utils.color import Palette
    import panel as pn
    import torch
    from collections.abc import Mapping, Sequence
    from dask.delayed import Delayed
    from os import PathLike
    from numpy.typing import DTypeLike
    from odc.geo import SomeCRS

    from geosave_engine.geodata.io.raster.cogs import CogFile
    from geosave_engine.geodata.io.raster.netcdf import (
        NetCDFEngine,
        NetCDFWriteOptions,
    )
    from geosave_engine.geodata.io.raster.zarr import ZarrWriteOptions
    from geosave_engine.geodata.io.storage import StorageOptions

    import holoviews as hv

    from geosave_engine.geodata import DataArray, Dataset


@xr.register_dataset_accessor("gs")
class GeoRaster(GeoRasterAccessor["Dataset"]):
    """Read and transform one raster Dataset.

    Each member reads only what it needs, so a Dataset mid-computation still
    answers for its grid.

    Args:
        data: Dataset holding one raster.

    Examples:
        >>> ds = source.load(anchor)
        >>> ds.gs.anchor.geobox.shape
        (512, 512)
    """

    @property
    def variables(self) -> tuple[str, ...]:
        """Name the data variables.

        Returns:
            Variable names, in Dataset order.
        """
        return tuple(str(name) for name in self._data.data_vars)

    @property
    def nodata(self) -> dict[str, float | int | None]:
        """Read the stored value standing for nodata in each variable.

        Returns:
            {
                variable name: its fill value, or None where it carries none,
            }

        Examples:
            >>> ds.gs.nodata
            {'red': 0, 'nir': 0}
        """
        return {name: self._data[name].gs.nodata for name in self.variables}

    @property
    def legend(self) -> dict[str, attrs.Legend | None]:
        """Read what each variable's values mean, None where they are no classes."""
        return {name: self._data[name].gs.legend for name in self.variables}

    @property
    def packing(self) -> dict[str, attrs.Packing | None]:
        """Read how each variable's stored values decode, None where they are physical."""
        return {name: self._data[name].gs.packing for name in self.variables}

    @property
    def spectral(self) -> dict[str, attrs.Spectral | None]:
        """Read where in the spectrum each variable was measured, None where unknown."""
        return {name: self._data[name].gs.spectral for name in self.variables}

    def write_crs(self, crs: SomeCRS | None = None) -> Dataset:
        """Write onto the spatial coordinates what they measure.

        The grid decides the answer: a projected raster names
        `projection_y_coordinate` and `projection_x_coordinate`, a geographic
        one `latitude` and `longitude`. No pixel moves; `reproject` does that.

        Args:
            crs: CRS the existing coordinates are in, for a raster that
                carries none or carries the wrong one. None reads the CRS
                the raster already carries.

        Returns:
            New Dataset whose spatial coordinates carry what the grid says of
            them: odc's units, resolution, and CRS, and CF's standard name and
            axis. Their other attrs are replaced.

        Raises:
            ValueError: The raster carries no locatable grid and `crs` names
                none, `crs` names no known reference system, or the raster is
                placed by ground control points rather than a regular grid.

        Examples:
            >>> ds.gs.write_crs("EPSG:32633").y.attrs["standard_name"]
            'projection_y_coordinate'
        """
        result = self._data if crs is None else self._data.odc.assign_crs(crs)
        geobox = result.gs.geobox
        if geobox is None:
            raise ValueError(
                "raster carries no locatable grid and no crs was given; pass "
                "one or assign a CRS first"
            )
        if not isinstance(geobox, GeoBox):
            raise ValueError(
                f"raster is placed by {type(geobox).__name__}, not a regular "
                f"grid, so its axes measure no CF coordinate"
            )
        return write_cf_axes(result, geobox)

    def write_nodata(
        self,
        value: float | int | None,
        *,
        target: str | Sequence[str] | None = None,
    ) -> Dataset:
        """Write the stored value standing for nodata pixels.

        It is written as CF's `_FillValue`, which odc and rioxarray read too.

        Args:
            value: Stored value marking nodata. None clears it.
            target: Variable names to write, defaulting to every data variable.

        Returns:
            New Dataset whose named variables carry `value` as their fill.

        Raises:
            KeyError: `target` names something that is not a data variable.
            ValueError: A named variable's dtype cannot hold `value`.

        Examples:
            >>> ds.gs.write_nodata(0).red.attrs["_FillValue"]
            0
            >>> ds.gs.write_nodata(-9999, target="elevation")
        """
        written = self._data.copy()
        for name in variable_names(self._data, target):
            written[name] = self._data[name].gs.write_nodata(value)
        return written

    def write_legend(
        self,
        class_map: Mapping[int, str] | None,
        color_map: Palette | None = None,
        *,
        target: str | Sequence[str] | None = None,
    ) -> Dataset:
        """Write the class each stored value stands for, and its colour.

        Plots then name and colour the classes, and a GeoTIFF or Zarr written
        afterwards carries them.

        Args:
            class_map: Stored value mapped to class name. None removes the
                legend.
            color_map: Stored value mapped to hex or RGB colour. None writes
                names only.
            target: Variable names to write, defaulting to every data variable.

        Returns:
            New Dataset whose named variables carry the legend.

        Raises:
            KeyError: `target` names something that is not a data variable.
            ValueError: `color_map` colours a value `class_map` does not name.

        Examples:
            >>> labels.gs.write_legend({0: "water", 1: "trees"}, {0: "#419bdf", 1: "#397d49"})
        """
        written = self._data.copy()
        for name in variable_names(self._data, target):
            written[name] = self._data[name].gs.write_legend(class_map, color_map)
        return written

    def write_packing(
        self,
        scale_factor: float | None,
        add_offset: float = 0.0,
        *,
        target: str | Sequence[str] | None = None,
    ) -> Dataset:
        """Write how stored values decode to physical values; pixels are unchanged.

        Args:
            scale_factor: Multiplier applied to each stored value. None
                removes the packing.
            add_offset: Value added after scaling.
            target: Variable names to write, defaulting to every data variable.

        Returns:
            New Dataset whose named variables carry the packing.

        Raises:
            KeyError: `target` names something that is not a data variable.

        Examples:
            >>> ds.gs.write_packing(1e-4, -0.1, target=["B04", "B08"])
        """
        written = self._data.copy()
        for name in variable_names(self._data, target):
            written[name] = self._data[name].gs.write_packing(scale_factor, add_offset)
        return written

    def write_spectral(
        self,
        common_name: str | None,
        *,
        center_wavelength: float | None = None,
        full_width_half_max: float | None = None,
        target: str | Sequence[str] | None = None,
    ) -> Dataset:
        """Write where in the spectrum the named variables were measured.

        Args:
            common_name: Band name shared across sensors, e.g. `"red"`.
            center_wavelength: Centre of the band, in micrometres.
            full_width_half_max: Width at half peak response, in micrometres.
            target: Variable names to write, defaulting to every data variable.

        Returns:
            New Dataset whose named variables carry the description.

        Raises:
            KeyError: `target` names something that is not a data variable.

        Examples:
            >>> ds.gs.write_spectral("red", center_wavelength=0.665, target="B04")
        """
        written = self._data.copy()
        for name in variable_names(self._data, target):
            written[name] = self._data[name].gs.write_spectral(
                common_name,
                center_wavelength=center_wavelength,
                full_width_half_max=full_width_half_max,
            )
        return written

    def write_rgb(self, red: str, green: str, blue: str) -> Dataset:
        """Name the three variables a true-colour composite draws.

        What a band measures does not say which channel draws it, a false
        colour composite drawing near-infrared as red. Recording the choice is
        also the only way a `GeoStack` panel can name its own bands.

        Args:
            red: Variable drawn as the red channel.
            green: Variable drawn as the green channel.
            blue: Variable drawn as the blue channel.

        Returns:
            New Dataset whose three named variables carry their channel, every
            other variable left uninterpreted.

        Raises:
            KeyError: A name is not a data variable.
            ValueError: One variable is named for two channels.

        Examples:
            >>> ds.gs.write_rgb("B04", "B03", "B02").gs.plot()
        """
        channels = dict(zip(("red", "green", "blue"), (red, green, blue), strict=True))

        variable_names(self._data, list(channels.values()))
        if len(set(channels.values())) < len(channels):
            raise ValueError(
                f"one variable cannot draw two channels, but these name "
                f"{channels}; give each channel its own variable"
            )

        drawn = {name: colour for colour, name in channels.items()}
        # A band left interpreted would compose a second, stale colour.
        return self._data.assign(
            {
                name: _interpret(self._data[name], drawn.get(name))
                for name in self.variables
            }
        )

    def statistics(self) -> pd.DataFrame:
        """Summarise every variable's present pixels, reading each of them.

        Returns:
            One row per variable, in Dataset order, with minimum, maximum,
            mean, population stddev, and valid_percent columns. All axes of
            each variable are reduced together.

        Raises:
            ValueError: A variable holds no present pixel.

        Examples:
            >>> ds.gs.statistics()
                 minimum  maximum  mean  stddev  valid_percent
            red      1.0     96.0  45.6    29.9          100.0
            nir      8.0     98.0  56.7    23.2          100.0
        """
        return statistics(self._data.data_vars)

    def to_array(self, *, dtype: DTypeLike | None = None) -> DataArray:
        """Stack every variable this raster carries into one array.

        No packing or display scaling is applied. A multi-band array keeps
        names on `band`, and `GeoArray.to_raster` stacks it back apart.

        Args:
            dtype: Cast applied to the stacked array. None keeps the source
                dtype, which every stacked variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`. Its own attrs are the ones
            every band carries with one value, so it reads as one variable;
            the rest are not kept.

        Raises:
            ValueError: This raster carries no variables or already spans the
                stacked axis, the variables carry different non-spatial
                dimensions, or they differ in dtype while `dtype` is None.

        Examples:
            >>> ds[["ndvi"]].gs.to_array()
            >>> ds[["B04", "B03", "B02"]].gs.to_array()
        """
        array = self._stacked(dtype)
        array.attrs = attrs.shared(
            [band.attrs for band in self._data.data_vars.values()]
        )
        return cast("DataArray", array)

    def explore(
        self, variable: str | tuple[str, str, str] | None = None, *, alpha: float = 0.7
    ) -> pn.pane.HoloViews:
        """Draw this raster over a web map, the way `plot` draws it.

        The map shows one date; select it with
        `.isel(time=...)` first where there are several.

        Args:
            variable: What to draw, as `plot` takes it.
            alpha: Opacity of the raster over the map.

        Returns:
            Panel pane drawing the raster over OpenStreetMap with bokeh.

        Raises:
            ValueError: This raster holds more than one date.

        Examples:
            >>> ds.gs.explore()
            >>> ds.gs.explore("ndvi", alpha=0.5)
        """
        from geosave_engine.geodata.viz import explore

        return explore(
            self._data,
            lambda raster: raster.gs.plot(variable, xlabel=""),
            alpha=alpha,
        )

    def plot(
        self,
        variable: str | tuple[str, str, str] | None = None,
        *,
        cmap: str | Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        cols: int = 4,
        title: str | None = None,
        xlabel: str | None = None,
    ) -> hv.Element | hv.NdLayout:
        """Draw this raster the way `variable` names it.

        Three names give a composite, one gives an Image, and None falls back
        to a sole variable, then to the ones measuring red, green, and blue.
        A `Legend` the variable carries supplies the class and colour maps.

        Args:
            variable: One data variable, three ordered red, green, blue, or
                None to read what this raster carries.
            cmap: Colormap. Refused for a composite, which draws no colormap.
            clim: Bounds mapped onto the full display range: colour limits for
                an Image, the composite's stretch onto `[0, 1]`.
            cols: Columns a `time` axis lays panels into. Ignored otherwise.
            title: Panel title above the axes. None suppresses automatic
                titles.
            xlabel: Location caption below the axes. None uses the raster's
                reverse-geocoded location; an empty string suppresses it.

        Returns:
            Image or RGB element over this raster's grid, or a Layout of one
            panel per `time` value, `cols` wide.

        Raises:
            KeyError: A named variable is absent.
            ValueError: `variable` is None while this raster carries more
                than one variable and no red, green, and blue among them, or
                the drawing refuses the raster. Where two variables draw one
                colour, the first draws it.

        Examples:
            >>> hv.save(ds.gs.plot("ndvi"), "ndvi.png")
            >>> ds = ds.gs.write_rgb("B04", "B03", "B02")
            >>> ds.gs.plot("ndvi", cmap="RdYlGn") + ds.gs.plot(clim=(0.0, 0.3))
        """
        # viz/__init__ re-exports a function named plot, shadowing the submodule.
        from geosave_engine.geodata.viz.plot import plot as draw

        if isinstance(variable, str):
            selected: tuple[str, ...] = (variable,)
        elif variable is not None:
            selected = tuple(variable)
        elif len(self.variables) == 1:
            selected = self.variables
        else:
            names = self.variables
            selected = tuple(names[band] for band in _rgb_indices(self._data))

        variable_names(self._data, selected)
        array = self._data[list(selected)].gs.to_array()

        # A composite carries colour rather than class codes, so it draws no legend.
        legend = None
        if len(selected) == 1:
            legend = self._data[selected[0]].gs.legend

        place = self.anchor.locate() if xlabel is None else None
        caption = place.to_address() if place is not None else xlabel
        return draw(
            array,
            cmap=cmap,
            clim=clim,
            class_map=legend.classes if legend else None,
            color_map=legend.color_map if legend else None,
            cols=cols,
            title=title,
            xlabel=caption,
        )

    def to_numpy(self, *, dtype: DTypeLike | None = None) -> np.ndarray:
        """Stack every variable this raster carries into one model-input array.

        Every axis besides the grid leads, then the variables, then the grid
        pair — torchgeo's `[T, C, H, W]`. No batch axis is added, so one raster
        is one sample. Order the variables with xarray before stacking them.

        Args:
            dtype: Cast applied to the stacked array. None keeps the source
                dtype, which every variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`.

        Raises:
            ValueError: This raster carries no variables, they carry different
                non-spatial dimensions, or they differ in dtype while `dtype`
                is None.

        Examples:
            >>> ds[["B04", "B08"]].gs.to_numpy().shape
            (2, 2, 256, 256)
        """
        return self._stacked(dtype).values

    def to_tensor(self, *, dtype: str | torch.dtype | None = None) -> torch.Tensor:
        """Stack every variable into one model-input tensor, laid out as `to_numpy`.

        Args:
            dtype: Torch dtype or its string name. None keeps the source dtype,
                which every variable must then share.

        Returns:
            Tensor shaped `(*axes, band, y, x)`.

        Raises:
            ValueError: A dtype name is unknown, or `to_numpy` refuses the
                raster.
            TypeError: `dtype` is neither a torch dtype nor a name, or a pixel
                dtype has no tensor equivalent.

        Examples:
            >>> ds[["B04", "B08"]].gs.to_tensor(dtype="float32").shape
            torch.Size([2, 2, 256, 256])
        """
        return as_tensor(self.to_numpy, dtype)

    def _stacked(self, dtype: DTypeLike | None) -> xr.DataArray:
        """Stack every variable onto the band axis, in the order held.

        Args:
            dtype: Cast applied to the stack. None keeps the source dtype,
                which every variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`, carrying no attrs.

        Raises:
            ValueError: This raster carries no variables or already spans the
                band axis, a variable does not span the grid, the variables
                carry different axes, or they differ in dtype while `dtype` is
                None.
        """
        var_names = self.variables
        if not var_names:
            raise ValueError(
                "this raster carries no variables to stack; it holds only coordinates"
            )

        if BAND_DIMENSION in self._data.dims:
            raise ValueError(
                f"this raster already carries a {BAND_DIMENSION!r} dimension, "
                f"which the stacked variables would name; rename it first"
            )

        ungridded = sorted(
            name
            for name in var_names
            if not set(SPATIAL_DIMENSIONS) <= set(self._data.variables[name].dims)
        )
        if ungridded:
            raise ValueError(
                f"{ungridded} do not span the grid {list(SPATIAL_DIMENSIONS)}, so stacking "
                f"them would repeat one value across every pixel; drop them or "
                f"place them on the grid first"
            )

        # Past the grid, the variables must span the same axes to stack.
        axes: dict[str, tuple[str, ...]] = {}
        for name in var_names:
            dims = self._data.variables[name].dims
            axes[name] = tuple(
                str(dim) for dim in dims if dim not in SPATIAL_DIMENSIONS
            )

        drifted = sorted(
            name for name in var_names[1:] if axes[name] != axes[var_names[0]]
        )
        if drifted:
            raise ValueError(
                f"{drifted} carry different axes from {var_names[0]!r}'s "
                f"{axes[var_names[0]]}; stack variables sharing one shape"
            )

        # Stacking promotes silently, so disagreeing dtypes are refused ahead of it.
        dtypes = {str(self._data.variables[name].dtype) for name in var_names}
        if dtype is None and len(dtypes) > 1:
            raise ValueError(
                f"{list(var_names)} carry different dtypes ({sorted(dtypes)}); stacking "
                f"them would promote them, so pass dtype= to choose one"
            )

        # A model reads the grid last, bands right before it: [T, C, H, W].
        stacked = (
            self._data[list(var_names)]
            .to_dataarray(dim=BAND_DIMENSION)
            .transpose(*axes[var_names[0]], BAND_DIMENSION, *SPATIAL_DIMENSIONS)
        )
        return stacked if dtype is None else stacked.astype(dtype)

    def to_cog(
        self,
        path: str | PathLike[str],
        *,
        map_scale: float | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **keywords: Any,
    ) -> tuple[CogFile, ...]:
        """Write this raster as Cloud Optimized GeoTIFFs, as its path template names.

        The path is a `str.format` template, filled in once per date and band;
        its fields decide the files:

        - `"out/scene.tif"`: one file holding every band.
        - `"out/s2_{band}.tif"`: one file per band.
        - `"out/{time:%Y%m%d}.tif"`: one file per date.
        - `"out/{time:%Y%m%d}/{band}.tif"`: one file per date and band.

        Two fields are built in: `time`, each file's instant, taking
        `strftime` directives, and `band`, each file's variable. Any keyword
        that is not a COG creation option is a field of the caller's, filled
        in as given or, when callable, called with each date's raster, as
        `lat=lambda raster: raster.gs.anchor.lonlat[1]`.

        A GeoTIFF holds one date, so several dates rendering to one path is
        an error; bands rendering to one path share a file.

        Args:
            path: Local path or fsspec URL ending in `.tif`.
            map_scale: Map denominator for TIFF resolution tags.
            overwrite: Replace existing files.
            storage_options: Options for the filesystem a URL names.
            **keywords: COG creation options, as `compress="deflate"`, and
                template fields, as `prefix="s2"`.

        Returns:
            One `CogFile` per file written, in time then variable order: its
            href, the bands it holds, and its instant.

        Raises:
            ValueError: The path does not end in `.tif`, several dates would
                share a file, `{time}` is named on a raster without dates, or
                the raster cannot be arranged as COGs.
            TypeError: A keyword repeats a built-in field.
            KeyError: The path names a field nothing fills.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> ds.gs.to_cog("out/{time:%Y%m%d}.tif")
            (CogFile(/abs/out/20250601.tif, bands=['red', 'nir'], 2025-06-01, -),
             CogFile(/abs/out/20250611.tif, bands=['red', 'nir'], 2025-06-11, -))
            >>> files = ds.gs.to_cog("out/s2_{time:%Y%m%d}_{band}.tif", compress="deflate")
            >>> [f.bands for f in files]
            [('red',), ('nir',), ('red',), ('nir',)]
            >>> ds.isel(time=0).gs.to_cog(
            ...     "out/{lat:.3f}.tif", lat=lambda raster: raster.gs.anchor.lonlat[1]
            ... )
            (CogFile(/abs/out/45.130.tif, bands=['red', 'nir'], undated, -),)
        """
        from geosave_engine.geodata.io import cogs

        options, fields = cogs.split_keywords(keywords)
        return cogs.write(
            self._data,
            path,
            fields=fields,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )

    def to_zarr(
        self,
        destination: str | PathLike[str],
        *,
        compute: bool = True,
        overwrite: bool = False,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> UPath | Delayed:
        """Write this raster to Zarr.

        Args:
            destination: Output path ending in `.zarr`.
            compute: False defers writing pixels.
            overwrite: Replace an existing destination when true.
            **write_options: Supported xarray Zarr write options.

        Returns:
            Destination path, or a delayed task returning it when `compute=False`.

        Raises:
            FileExistsError: The destination exists and overwrite is false.
            TypeError: An option is unsupported.
            ValueError: The destination is invalid.
        """
        from geosave_engine.geodata.io import zarr

        return zarr.write(
            self._data,
            destination,
            compute=compute,
            overwrite=overwrite,
            **write_options,
        )

    def to_netcdf(
        self,
        destination: str | PathLike[str],
        *,
        compute: bool = True,
        engine: NetCDFEngine = "netcdf4",
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> UPath | Delayed:
        """Write this raster to netCDF.

        Args:
            destination: Output path ending in `.nc`, `.nc4`, or `.cdf`.
            compute: False returns a delayed task that writes and returns its path.
            engine: Concrete xarray netCDF writing engine.
            overwrite: Replace an existing destination when true.
            storage_options: Options for the filesystem a URL names.
            **write_options: Supported xarray netCDF write options.

        Returns:
            Destination path, or a delayed task returning it when `compute=False`.

        Raises:
            FileExistsError: The destination exists and overwrite is false.
            TypeError: An option is unsupported.
            ValueError: The destination is invalid.
        """
        from geosave_engine.geodata.io import netcdf

        return netcdf.write(
            self._data,
            destination,
            compute=compute,
            overwrite=overwrite,
            engine=engine,
            storage_options=storage_options,
            **write_options,
        )


def _interpret(band: xr.DataArray, colour: str | None) -> xr.DataArray:
    """Return the band drawn as `colour`, its variable name kept; None draws it as nothing."""
    known = attrs.parse(band, attrs.GDALBand) or attrs.GDALBand()
    stated = known.model_copy(update={"colorinterp": colour})
    empty = stated.variable_name is None and stated.colorinterp is None
    return attrs.stamp(band, attrs.GDALBand, None if empty else stated)


def _rgb_indices(ds: xr.Dataset) -> tuple[int, int, int]:
    """Find the bands a true-colour composite draws, red first.

    Raises:
        ValueError: A colour is measured by no band. Two bands measuring one
            colour compose from the first.
    """
    bands = (attrs.parse(band, attrs.GDALBand) for band in ds.data_vars.values())
    measures = [None if band is None else band.colorinterp for band in bands]
    absent = [colour for colour in RGB if colour not in measures]
    if absent:
        raise ValueError(
            f"no band of {list(ds.data_vars)} draws {absent}, so this raster "
            f"composes no colour of its own; name its channels once with "
            f"ds.gs.write_rgb('B04', 'B03', 'B02'), or name three bands for "
            f"one drawing with ds.gs.plot(('B08', 'B04', 'B03'))"
        )
    red, green, blue = (measures.index(colour) for colour in RGB)
    return red, green, blue
