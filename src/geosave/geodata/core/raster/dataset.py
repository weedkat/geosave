"""Read raster Datasets through the `gs` Dataset accessor.

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

from typing import TYPE_CHECKING, Any, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata.attrs.groups.gdal import RGB
from geosave.geodata.conventions import (
    BAND_DIMENSION,
    CRS_COORDINATE,
    SPATIAL_DIMENSIONS,
    TIME_COORDINATE,
)
from geosave.geodata.core.raster.grid import name_grid, place_on_grid, write_reference
from geosave.geodata.conventions import ITEM_ID_FORMAT
from geosave.geodata.utils.summary import class_percent, statistics
from geosave.geodata.utils.tensor import as_tensor

from .base import GeoRaster

if TYPE_CHECKING:
    import pystac
    from collections.abc import Mapping, Sequence

    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike
    from odc.geo import SomeCRS

    from odc.geo.gcp import GCPGeoBox

    from geosave.geodata import DataArray, Dataset, DataTree, GeoDataFrame
    from geosave.geodata.stac.item import ItemTime
    from dask.delayed import Delayed
    from upath import UPath
    from geosave.geodata.io.formats.geotiff import COGWriteOptions
    from geosave.geodata.io.formats.netcdf import NetCDFEngine
    from geosave.geodata.io.formats.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.formats.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from geosave.geodata.utils.color import Palette
    from geosave.geodata.viz.explore import Explorer


class GeoDataset(GeoRaster["Dataset"]):
    """Read and transform one raster Dataset.

    Each member reads only what it needs, so a Dataset mid-computation still
    answers for its grid.

    Args:
        data: Dataset holding one raster.

    Examples:
        >>> ds = io.load_dataset(items, geobox=grid)
        >>> ds.gs.geobox.shape
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
    def stac_collection(self) -> pystac.Collection | None:
        """Read the STAC Collection this raster was loaded from, None where it states none.

        Examples:
            >>> io.load_dataset(items, geobox=grid).gs.stac_collection.id
            'sentinel-2-l2a'
        """
        import pystac

        stated = attrs.StacMetadata.from_attrs(self._data.attrs)
        if stated is None or stated.collection is None:
            return None
        return pystac.Collection.from_dict(stated.collection)

    @property
    def stac_common_properties(self) -> dict[str, Any]:
        """Read the Item properties every date of this raster shares, empty where none are stated.

        Examples:
            >>> io.load_dataset(items, geobox=grid).gs.stac_common_properties["platform"]
            'sentinel-2a'
        """
        stated = attrs.StacMetadata.from_attrs(self._data.attrs)
        return {} if stated is None else dict(stated.common_properties or {})

    @property
    def stac_properties(self) -> pd.DataFrame:
        """Read each date's Item properties from the `stac_properties` coordinate.

        Returns:
            One row per `time` label, one column per property key; a key a
            date lacks is NaN. Empty where the coordinate is absent.

        Examples:
            >>> clear = ds.sel(time=ds.gs.stac_properties["eo:cloud_cover"] < 30)
        """
        from geosave.geodata.stac.item import read_properties_coordinate

        mappings = read_properties_coordinate(self._data)
        if not mappings:
            return pd.DataFrame()
        labels = np.atleast_1d(self._data[TIME_COORDINATE].values)
        return pd.DataFrame(mappings, index=pd.DatetimeIndex(labels, name="time"))

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

    def write_grid(self, geobox: GeoBox | GCPGeoBox) -> Dataset:
        """Place this raster on a grid: `y` and `x` coordinates and the CRS.

        Rasters are built with xarray's own constructors; this writes what they
        cannot say themselves, where each pixel sits and what its numbers
        measure, as `write_nodata` writes what marks an absent pixel.

        Args:
            geobox: Regular grid, or a grid placed by ground control points.

        Returns:
            New Dataset whose variables point at the CRS coordinate, its axes
            carrying CF attrs on a regular grid. Coordinates on other
            dimensions, such as `time`, are kept.

        Raises:
            ValueError: A variable's dimensions do not end in `y, x`, their
                sizes differ from the grid's, or the grid carries no CRS.

        Examples:
            >>> scene = xr.Dataset({"red": (("time", "y", "x"), red)}, coords={"time": dates})
            >>> scene.gs.write_grid(grid).gs.geobox == grid
            True
        """
        return cast("Dataset", place_on_grid(self._data, geobox))

    def write_crs(self, crs: SomeCRS | None = None) -> Dataset:
        """Write onto the existing spatial coordinates what they measure.

        The grid decides the answer: a projected raster names
        `projection_y_coordinate` and `projection_x_coordinate`, a geographic
        one `latitude` and `longitude`. No pixel moves; `reproject` does that,
        and `write_grid` places a raster that has no coordinates yet.

        Args:
            crs: CRS the existing coordinates are in, for a raster that
                carries none or carries the wrong one. None reads the CRS
                the raster already carries.

        Returns:
            New Dataset whose spatial coordinates carry what the grid says of
            them: odc's units, resolution, and CRS, and CF's standard name and
            axis. Their other attrs are replaced.

        Raises:
            ValueError: The raster has no `y`/`x` coordinates, carries no CRS
                and `crs` names none, `crs` names no known reference system,
                or the raster is placed by ground control points.

        Examples:
            >>> ds.gs.write_crs("EPSG:32633").y.attrs["standard_name"]
            'projection_y_coordinate'
        """
        return cast("Dataset", write_reference(self._data, crs))

    def to_datatree(self, collection: str) -> DataTree:
        """Make this raster one collection's grid node, as `/{collection}/{res}`.

        The node is named after its grid as EOPF names grid groups (`r10m`).
        This raster's attrs go on the collection node, so the tree's
        `to_dataset` gives them back. Combine collections, or one
        collection's grids, with `xr.merge`.

        Args:
            collection: Name of the collection node.

        Returns:
            DataTree of one collection node over one grid node.

        Raises:
            ValueError: The raster sits on no georeferenced grid.

        Examples:
            >>> tree = xr.merge([scene.gs.to_datatree("s2"), dem.gs.to_datatree("dem")])
            >>> list(tree.gs.rasters)
            ['s2/r10m', 'dem/r30m']
        """
        geobox = self.geobox
        if not isinstance(geobox, GeoBox) or geobox.crs is None:
            raise ValueError(
                f"a grid node is named after its grid, and this raster sits on none; "
                f"place it with gs.write_grid before making it {collection!r}"
            )
        node = self._data.copy()
        node.attrs = {}
        collection_node = xr.Dataset(attrs=dict(self._data.attrs))
        return cast(
            "DataTree",
            xr.DataTree.from_dict(
                {collection: collection_node, f"{collection}/{name_grid(geobox)}": node}
            ),
        )

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
        bands: dict[str, xr.DataArray] = {}
        for name in self.variables:
            band = self._data[name]
            stated = attrs.GDALBand.from_attrs(band.attrs) or attrs.GDALBand()
            interpreted = stated.model_copy(update={"colorinterp": drawn.get(name)})
            bands[name] = attrs.stamp_attrs(band, interpreted)
        return self._data.assign(bands)

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
            basemap: Basemap under the layers, as `viz.explore.page.map.BASEMAPS` names it.
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
            >>> ds.gs.explore("ndvi", basemap="Imagery") * fields
            >>> ds.gs.explore(rgb=("B08", "B04", "B03"))
        """
        from geosave.geodata.viz.explore import Explorer, RasterLayer

        layer = RasterLayer(self._data, band, rgb=rgb, cmap=cmap, alpha=alpha)
        return Explorer(layer, basemap=basemap)

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

        return draw_raster(
            self._data,
            band,
            rgb=rgb,
            cmap=cmap,
            clim=clim,
            cols=cols,
            title=title,
            xlabel=xlabel,
        )

    def read_pixels(self, geobox: GeoBox) -> Dataset:
        """Read the pixels inside `geobox`, keeping every n-th before computing.

        Only the window `geobox` overlaps is computed, thinned to about its
        pixel size; a Dataset holds no overviews, so a coarse `geobox` still
        computes the chunks under the window.

        Args:
            geobox: Grid to read onto, such as a map's view.

        Returns:
            Dataset on `geobox` with the same variables and attrs, float32
            with nodata and pixels outside the raster as NaN.

        Examples:
            >>> ds.gs.read_pixels(view).red.shape
            (600, 900)
        """
        from odc.geo.overlap import compute_reproject_roi
        from odc.geo.roi import roi_is_empty
        from odc.geo.xr import xr_coords, xr_reproject

        grid = self.geobox
        if not isinstance(grid, GeoBox):
            raise ValueError("reading pixels needs a regular georeferenced grid")
        roi = compute_reproject_roi(grid, geobox)

        # 1. A raster the grid misses is NaN, computed from nothing
        if roi_is_empty(roi.roi_src):
            missing = {}
            for name, variable in self._data.data_vars.items():
                shape = (*variable.shape[:-2], *geobox.shape)
                missing[name] = xr.DataArray(
                    np.full(shape, np.nan, "float32"),
                    dims=variable.dims,
                    attrs=variable.attrs,
                )
            return cast("Dataset", xr.Dataset(missing, coords=xr_coords(geobox)))

        # 2. The window, every `read_shrink`-th pixel, computed
        y_window, x_window = roi.roi_src
        steps = {
            "y": slice(y_window.start, y_window.stop, roi.read_shrink),
            "x": slice(x_window.start, x_window.stop, roi.read_shrink),
        }
        window = self._data.isel(steps).gs.to_nan().astype("float32").compute()

        # 3. Onto the grid; the source's CRS attrs survive the warp
        pixels = xr_reproject(window, geobox, resampling="nearest", dst_nodata=np.nan)
        pixels = pixels.odc.assign_crs(geobox.crs).assign_coords(
            {CRS_COORDINATE: xr_coords(geobox)[CRS_COORDINATE]}
        )
        return cast("Dataset", pixels)

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
        path: Location,
        *,
        path_format: str | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write this raster as Cloud Optimized GeoTIFFs under the folder `path`.

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
            >>> paths = ds.gs.to_cog("out", path_format="{time:%Y%m%d}_{band}.tif")
        """
        from geosave.geodata.io.formats import geotiff
        from geosave.geodata.io.paths import expand_paths
        from geosave.geodata.io.storage import location

        files = expand_paths(self._data, location(path, storage_options), path_format)
        return geotiff.write_files(
            files, geotiff.write_cog, compute=compute, overwrite=overwrite, **options
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
        """Write this raster as one Zarr store, every date and variable in it.

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
            KeyError: An encoding name is no variable of the raster.
            FileExistsError: The store exists and overwrite is false.

        Examples:
            >>> ds.gs.to_zarr("scene.zarr")
        """
        from geosave.geodata.io.formats import zarr

        return zarr.write_zarr(
            self._data,
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
        """Write this raster as one netCDF file, every date and variable in it.

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
            KeyError: An encoding name is no variable of the raster.
            FileExistsError: The file exists and overwrite is false.

        Examples:
            >>> ds.gs.to_netcdf("scene.nc")
        """
        from geosave.geodata.io.formats import netcdf

        return netcdf.write_netcdf(
            self._data,
            path,
            compute=compute,
            engine=engine,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )

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
        **options: Unpack[COGWriteOptions],
    ) -> GeoDataFrame:
        """Write this raster's files and return their STAC Items as an item table.

        Each date becomes one Item, its files the assets. The Items are built
        from this raster: its grid, dates and band attrs, and the STAC
        metadata `io.load_dataset` records, `stac_common_properties` overlaid
        with each date's `stac_properties`.

        Args:
            path: Folder the files go under.
            path_format: Relative path ending in .tif, as `to_cog` lays it out,
                and filling `{collection}` too. None writes one file per band
                and date. Catalog assets are COGs; write a store with
                `to_zarr` or `to_netcdf`.
            collection: Collection, or its id, the Items belong to. None
                takes the `stac_collection` this raster carries, if any.
            id_format: Item id template filling `lat`, `lon`, `start`, `end`,
                and for one-grid Items `res`, `width`, `height`, `epsg`.
            datetime: Date or interval for a raster stating no time.
            overwrite: Replace files that exist.
            storage_options: Credentials and connection options for the files.
            **options: GeoTIFF creation options.

        Returns:
            Item table, one row per date. The Collection, its extent covering
            the Items, is in the table's metadata.

        Raises:
            ValueError: A file does not end in .tif, several dates share a
                file, or the raster has no locatable grid.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> scene = io.load_dataset(items, geobox=grid, with_properties=["eo:cloud_cover"])
            >>> catalog = scene.gs.to_catalog("out", path_format="{time:%Y%m%d}/{band}.tif")
            >>> catalog["eo:cloud_cover"].tolist()
            [4.2, 11.0]
        """
        import pystac

        from geosave.geodata.io.stac import write_assets

        carried = self.stac_collection
        if isinstance(collection, pystac.Collection):
            name, recorded_collection = collection.id, collection
        elif collection is not None:
            name = collection
            recorded_collection = (
                carried if carried is not None and carried.id == name else None
            )
        else:
            name = None if carried is None else carried.id
            recorded_collection = carried
        return write_assets(
            self._data,
            path,
            path_format=path_format,
            collection=name,
            collections=[] if recorded_collection is None else [recorded_collection],
            id_format=id_format,
            datetime=datetime,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )
