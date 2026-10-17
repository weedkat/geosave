"""Build raster Datasets and read them through the `gs` Dataset accessor.

Georeferencing is odc-geo's. GeoSave adds only the CF axis semantics odc
omits, which `write_crs` writes.

Examples:
    A raster carrying no CRS holds pixels without claiming ground position, so
    `GeoDataset.geobox` is None, and members needing ground position refuse.
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

from functools import partial
from typing import TYPE_CHECKING, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata.attrs.groups.gdal import RGB

from geosave.geodata.utils.summary import class_percent, statistics
from geosave.geodata.utils.tensor import as_tensor

from .base import GeoRaster
from .create import write_cf_axes
from geosave.geodata.conventions import (
    BAND_DIMENSION,
    SPATIAL_DIMENSIONS,
)


if TYPE_CHECKING:
    from geosave.geodata.utils.color import Palette
    import torch
    from collections.abc import Mapping, Sequence
    from os import PathLike
    from numpy.typing import DTypeLike
    from odc.geo import SomeCRS

    from geosave.geodata.io.raster.file import Scene, Writer
    from geosave.geodata.viz.explorer import Explorer, Overlay
    from geosave.geodata.io.raster.netcdf import (
        NetCDFEngine,
        NetCDFWriteOptions,
    )
    from geosave.geodata.io.raster.geotiff import COGWriteOptions
    from geosave.geodata.io.raster.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions

    import holoviews as hv

    from geosave.geodata import DataArray, Dataset


@xr.register_dataset_accessor("gs")
class GeoDataset(GeoRaster["Dataset"]):
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

    def _targets(self, target: str | Sequence[str] | None) -> list[str]:
        """Return the data variables `target` names, every one when None.

        Raises:
            KeyError: A name is not a data variable.
        """
        if target is None:
            return list(self.variables)
        names = [target] if isinstance(target, str) else list(target)
        missing = [name for name in names if name not in self.variables]
        if missing:
            raise KeyError(
                f"{missing} are not data variables of this raster; it carries "
                f"{list(self.variables)}"
            )
        return names

    @property
    def rgb(self) -> tuple[str, str, str] | None:
        """Name the variables drawing red, green and blue, or None where none are named.

        Examples:
            >>> ds.gs.write_rgb("B04", "B03", "B02").gs.rgb
            ('B04', 'B03', 'B02')
        """
        # Two variables drawing one channel compose from the first.
        drawing: dict[str, str] = {}
        for name in reversed(self.variables):
            stated = attrs.GDALBand.from_attrs(self._data[name].attrs)
            if stated is not None and stated.colorinterp in RGB:
                drawing[stated.colorinterp] = name
        if drawing.keys() != set(RGB):
            return None
        red, green, blue = (drawing[colour] for colour in RGB)
        return red, green, blue

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

    @property
    def acdd(self) -> attrs.ACDD | None:
        """Read what the whole dataset is, None where its root states no ACDD key."""
        return attrs.ACDD.from_attrs(self._data.attrs)

    def write_acdd(self, acdd: attrs.ACDD | None) -> Dataset:
        """Write what the whole dataset is onto its root attrs.

        Every ACDD key is replaced first, so no stale one survives.

        Args:
            acdd: Description to state. None removes every ACDD key.

        Returns:
            Shallow copy of the Dataset; pixels are not copied.

        Examples:
            >>> ds.gs.write_acdd(attrs.ACDD(title="Forest")).gs.acdd.title
            'Forest'
        """
        if acdd is None:
            return attrs.clear_attrs(self._data, attrs.ACDD)
        return attrs.stamp_attrs(self._data, acdd)

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
        for name in self._targets(target):
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
        for name in self._targets(target):
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

        Sentinel-2 L2A is the usual case: ESA stores reflectance as
        `DN * 1e-4 + offset`, with offset 0 for processing baseline below 04.00
        and -0.1 (`BOA_ADD_OFFSET` -1000) from 04.00, January 2022. Both are
        in each scene's `MTD_MSIL2A.xml`, which STAC items do not repeat; see
        https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/product-formatting.

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
        for name in self._targets(target):
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
        for name in self._targets(target):
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
        also the only way a `GeoDataTree` panel can name its own bands.

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

        self._targets(list(channels.values()))
        if len(set(channels.values())) < len(channels):
            raise ValueError(
                f"one variable cannot draw two channels, but these name "
                f"{channels}; give each channel its own variable"
            )

        drawn = {name: colour for colour, name in channels.items()}
        # A band left interpreted would compose a second, stale colour.
        return self._data.assign(
            {
                name: attrs.stamp_attrs(
                    self._data[name],
                    attrs.GDALBand.replace(
                        self._data[name].attrs, colorinterp=drawn.get(name)
                    ),
                )
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
        return statistics([self._data])

    def class_percent(self) -> pd.DataFrame:
        """Give the share of present pixels in each class of every label variable.

        Returns:
            One column per categorical variable that states class names, one
            row per class name in legend order.

        Examples:
            >>> ds.gs.class_percent()
                   label
            water   12.5
            trees   87.5
        """
        columns = {}
        for name, band in self._data.data_vars.items():
            if band.gs.legend is not None and band.gs.legend.classes is not None:
                columns[name] = class_percent([band])
        return pd.DataFrame(columns)

    def to_array(self, *, dtype: DTypeLike | None = None) -> DataArray:
        """Stack every variable this raster carries into one array.

        No packing or display scaling is applied. A multi-band array keeps
        names on `band`, and `GeoDataArray.to_raster` stacks it back apart.

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
        array.attrs = attrs.intersect(
            [band.attrs for band in self._data.data_vars.values()]
        )
        return cast("DataArray", array)

    def explore(
        self,
        band: str | None = None,
        *,
        rgb: tuple[str, str, str] | None = None,
        overlay: Overlay | None = None,
        basemap: str | None = None,
        cmap: str | None = None,
        alpha: float = 0.8,
    ) -> Explorer:
        """Open this raster on a web map with a sidebar to explore it.

        Choose a band or the declared RGB composite from Display in the sidebar. COGs use
        viewport windows and overviews. Computed arrays are indexed before
        execution, but an overview can still touch every source chunk.

        Args:
            band: Band to show first. None shows the composite `rgb` records,
                else the first band.
            rgb: Variables composed red, green and blue, offered and shown
                first.
            overlay: Rasters, GeoDataFrames or item tables drawn on top, such as
                field boundaries or a prediction; a mapping names each layer.
            basemap: Basemap under the layers, as `viz.map.BASEMAPS` names it.
                None uses the light basemap.
            cmap: Colormap of bands without classes.
            alpha: Opacity of this raster over the basemap.

        Returns:
            Explorer, shown by a notebook or served with `.servable()`.

        Raises:
            KeyError: A named variable is absent.
            ValueError: Both `band` and `rgb` are named.

        Examples:
            >>> ds.gs.explore()
            >>> ds.gs.explore("ndvi", overlay=fields, basemap="Imagery")
            >>> ds.gs.explore(rgb=("B08", "B04", "B03"))
        """
        from geosave.geodata.viz.explorer import Explorer, RasterLayer, create_layers

        layer = RasterLayer(self._data, band, rgb=rgb, cmap=cmap, alpha=alpha)
        return Explorer(layer, *create_layers(overlay), basemap=basemap)

    def plot(
        self,
        band: str | None = None,
        *,
        rgb: tuple[str, str, str] | None = None,
        cmap: str | Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        cols: int = 4,
        title: str | None = None,
        xlabel: str | None = None,
    ) -> hv.Element | hv.Layout:
        """Draw one band of this raster, or a composite of three.

        A band draws through `cmap`; one whose `Legend` lists classes draws
        each class in its colour, named in a shared legend to the figure's
        right. Repeated names and colours appear once. A composite draws its
        three variables red, green and blue. Naming neither draws the sole
        variable, else the composite `rgb` records.

        Args:
            band: Variable to draw.
            rgb: Variables drawn red, green and blue, such as a false-colour
                `("B08", "B04", "B03")`.
            cmap: Colormap of a band. Refused for classes and composites.
            clim: Colour limits of a band, or a composite's stretch onto
                `[0, 1]`. Refused for classes.
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
            ValueError: Both `band` and `rgb` are named, neither is while this
                raster carries several variables and records no composite, or
                the drawing refuses the raster.

        Examples:
            >>> hv.save(ds.gs.plot("ndvi"), "ndvi.png")
            >>> ds.gs.plot(rgb=("B08", "B04", "B03"), clim=(0.0, 0.3))
            >>> ds.gs.plot("ndvi", cmap="YlGn") + ds.gs.write_rgb("B04", "B03", "B02").gs.plot()
        """
        from geosave.geodata.viz.plot import draw_raster

        band, rgb = self._drawn(band, rgb)
        place = self.anchor.locate() if xlabel is None else None
        caption = place.to_address() if place is not None else xlabel
        return draw_raster(
            self._data,
            band,
            rgb=rgb,
            cmap=cmap,
            clim=clim,
            cols=cols,
            title=title,
            xlabel=caption,
        )

    def _drawn(
        self, band: str | None, rgb: tuple[str, str, str] | None
    ) -> tuple[str | None, tuple[str, str, str] | None]:
        """Return the band or composite a drawing names, else the sole variable, else `rgb`."""
        if band is not None and rgb is not None:
            raise ValueError("name a band or a composite, not both")
        if band is None and rgb is None:
            if len(self.variables) == 1:
                return self.variables[0], None
            if self.rgb is None:
                raise ValueError(
                    f"this raster holds {list(self.variables)} and records no "
                    f"composite; name one band, ds.gs.plot('ndvi'), or a "
                    f"composite, ds.gs.plot(rgb=('B04', 'B03', 'B02')), or record "
                    f"one with ds.gs.write_rgb('B04', 'B03', 'B02')"
                )
            return None, self.rgb
        self._targets((band,) if band is not None else cast("tuple[str, ...]", rgb))
        return band, rgb

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
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[Scene]:
        """Write this raster as Cloud Optimized GeoTIFFs, as its path template names.

        The path is a `str.format` template, filled in once per date and band;
        its fields decide the files:

        - `"out/scene.tif"`: one file holding every band.
        - `"out/s2_{band}.tif"`: one file per band.
        - `"out/{time:%Y%m%d}.tif"`: one file per date.
        - `"out/{time:%Y%m%d}/{band}.tif"`: one file per date and band.

        Two fields are built in: `time`, each file's instant, taking
        `strftime` directives, and `band`, each file's variable. Anything else
        goes into the path before it is given, as `f"out/{name}_{{band}}.tif"`.

        A GeoTIFF holds one date, so several dates rendering to one path is
        an error; bands rendering to one path share a file. The files of one
        date are one `Scene`, each file an asset keyed by the band it holds,
        or `"image"` where it holds several.

        Args:
            path: Local path or fsspec URL ending in `.tif`.
            map_scale: Map denominator for TIFF resolution tags.
            overwrite: Replace existing files.
            compute: False returns every scene pending, written by one
                `dask.compute` of the result.
            storage_options: Options for the filesystem a URL names.
            **options: COG creation options, as `compress="deflate"`.

        Returns:
            One scene per date, in time order. `open_raster` and
            `stac.create_items` take them as they are.

        Raises:
            ValueError: The path does not end in `.tif`, several dates would
                share a file, `{time}` is named on a raster without dates, or
                the raster cannot be arranged as COGs.
            KeyError: The path names a field nothing fills.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> ds.gs.to_cog("out/{time:%Y%m%d}.tif")
            [Scene(None, {'image': 20250601.tif}), Scene(None, {'image': 20250611.tif})]
            >>> ds.gs.to_cog("out/s2_{time:%Y%m%d}_{band}.tif", compress="deflate")[0]
            Scene(None, {'red': s2_20250601_red.tif, 'nir': s2_20250601_nir.tif})
            >>> (scenes,) = dask.compute(ds.gs.to_cog("out/{time:%Y%m%d}.tif", compute=False))
        """
        from geosave.geodata.io.raster.geotiff import write_cog
        from geosave.geodata.io.raster.layout import layout, save
        from geosave.geodata.io.storage import location

        plan = layout(self._data, location(path, storage_options))
        write = partial(write_cog, map_scale=map_scale, overwrite=overwrite, **options)
        return save(plan, write, compute=compute)

    def to_zarr(
        self,
        destination: Location,
        *,
        compute: bool = True,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> Scene:
        """Write this raster to Zarr.

        Args:
            destination: Output path or fsspec URL ending in `.zarr`.
            compute: False returns the scene pending, written on `.compute()`.
            overwrite: Replace an existing destination when true.
            storage_options: Options for the filesystem a URL names.
            **write_options: Supported xarray Zarr write options.

        Returns:
            One scene holding the store, keyed as `to_cog` keys a file, or pending.

        Raises:
            FileExistsError: The destination exists and overwrite is false.
            TypeError: An option is unsupported.
            ValueError: The destination is invalid.

        Examples:
            >>> ds.gs.to_zarr("scene.zarr")
            Scene(None, {'image': scene.zarr})
        """
        from geosave.geodata.io import zarr

        write = partial(zarr.write, overwrite=overwrite, **write_options)
        return self._save_one(destination, write, storage_options, compute)

    def to_netcdf(
        self,
        destination: Location,
        *,
        compute: bool = True,
        engine: NetCDFEngine = "netcdf4",
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> Scene:
        """Write this raster to netCDF.

        Args:
            destination: Output path or fsspec URL ending in `.nc`, `.nc4`, or
                `.cdf`.
            compute: False returns the scene pending, written on `.compute()`.
            engine: Concrete xarray netCDF writing engine.
            overwrite: Replace an existing destination when true.
            storage_options: Options for the filesystem a URL names.
            **write_options: Supported xarray netCDF write options.

        Returns:
            One scene holding the file, keyed as `to_cog` keys a file, or pending.

        Raises:
            FileExistsError: The destination exists and overwrite is false.
            TypeError: An option is unsupported.
            ValueError: The destination is invalid.
        """
        from geosave.geodata.io import netcdf

        write = partial(
            netcdf.write, engine=engine, overwrite=overwrite, **write_options
        )
        return self._save_one(destination, write, storage_options, compute)

    def _save_one(
        self,
        destination: Location,
        write: Writer,
        storage_options: StorageOptions | None,
        compute: bool,
    ) -> Scene:
        """Write this raster whole to one file or store, as one scene."""
        from geosave.geodata.io.raster.layout import layout, save
        from geosave.geodata.io.storage import location

        plan = layout(self._data, location(destination, storage_options))
        (scene,) = save(plan, write, compute=compute)
        return scene
