"""Read raster Datasets through the `gs` Dataset accessor.

Georeferencing is odc-geo's: the structure ODC produces is the structure
the accessor reads. CF axis attrs are written only into netCDF and Zarr files.

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

    Geographic grids are named as ODC names them:

    >>> wgs84.red.dims
    ('latitude', 'longitude')
"""

from __future__ import annotations


from typing import TYPE_CHECKING, Any, Literal, Unpack, cast

import cf_xarray  # noqa: F401  — registers the .cf accessor
import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import (
    BAND_DIMENSION,
    TIME_COORDINATE,
)
from geosave.geodata.core.raster.utils.statistics import class_percent, statistics
from geosave.geodata.core.raster.utils.tensor import as_tensor

from .base import GeoRaster, RasterTime
from .transform.temporal import Reducer, Interpolation
from geosave.geodata.errors import RasterValidationError

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from geosave.geodata.utils.datetime import Freq
    from xarray.core.types import CombineAttrsOptions
    from odc.geo import SomeResolution
    from .transform.warp import Resampling
    from .transform.extent import Padding, Region
    from odc.geo.geom import Geometry
    from geosave.geodata.core.region import GeoRegion
    import pystac
    from geosave.geodata import GeoDataFrame
    from collections.abc import Hashable, Mapping, Sequence

    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike
    from odc.geo import SomeCRS

    from geosave.geodata import DataArray, Dataset, DataTree
    from dask.delayed import Delayed
    from upath import UPath
    from geosave.geodata.io.formats.netcdf import NetCDFEngine
    from geosave.geodata.io.formats.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.formats.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from geosave.geodata.utils.color import Palette
    from geosave.geodata.viz.explore import Explorer


class GeoDataset(RasterTime["Dataset"], GeoRaster["Dataset"]):
    """Read and transform one raster Dataset.

    Each member reads only what it needs, so a Dataset mid-computation still
    answers for its grid.

    Args:
        data: Dataset holding one raster.

    Examples:
        >>> ds = stac.load_dataset(items, geobox=grid)
        >>> ds.gs.geobox.shape
        (512, 512)
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
    ) -> Dataset:
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
        from .transform.temporal import CELL_METHODS, composite_ds

        data = self._data
        cell_method = CELL_METHODS[method]
        for name in self.variables:
            variable = data[name]
            if attrs.CFPacking.is_stated_in(variable.attrs):
                raise ValueError(
                    f"decode active CF attrs on {name!r} with mask_and_scale() before composite"
                )
            if dim not in variable.dims:
                continue
            if variable.gs.is_categorical and (
                cell_method != "point" or interpolation is not None
            ):
                raise ValueError(
                    f"class codes in {name!r} require 'first' or 'last' without interpolation"
                )

        result = composite_ds(
            self._data,
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
        return cast("Dataset", result)

    @property
    def spatial_dims(self) -> tuple[str, str] | None:
        """Find spatial dimension names in vertical, horizontal order, as ODC does."""
        return self._data.odc.spatial_dims

    @property
    def geobox(self) -> GeoBox | None:
        """Read the grid as ODC infers it, or None when it carries none.

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

    def _validate_nonempty(self, variables: Mapping[Hashable, xr.DataArray]) -> None:
        """Require at least one data-bearing variable."""
        if not variables:
            raise RasterValidationError(
                "There are no raster variables to validate. Select or load raster data first."
            )

    @property
    def variables(self) -> tuple[Hashable, ...]:
        """Name the data variables, in Dataset order."""
        return tuple(self._data.data_vars)

    def _targets(self, target: list[str] | None) -> list[Hashable]:
        """Return the data variables `target` names, every one when None.

        Raises:
            KeyError: A name is not a data variable.
        """
        if target is None:
            return list(self.variables)
        missing = [name for name in target if name not in self._data.data_vars]
        if missing:
            raise KeyError(
                f"{missing} are not data variables of this raster; it carries "
                f"{list(self.variables)}"
            )
        return list(target)

    @property
    def rgb(self) -> tuple[str, str, str] | None:
        """Name the variables drawing red, green and blue, or None where none are named.

        A stated `Display.rgb` wins; otherwise the variables whose `EOBand`
        common name is red, green and blue draw them.

        Examples:
            >>> ds.gs.write_rgb("B04", "B03", "B02").gs.rgb
            ('B04', 'B03', 'B02')
        """
        display = attrs.Display.from_attrs(self._data.attrs)
        if display is not None and display.rgb is not None:
            if all(name in self._data.data_vars for name in display.rgb):
                red, green, blue = display.rgb
                return red, green, blue
            return None
        drawing: dict[str, str] = {}
        for name in reversed(self.variables):
            band = self._data[name].gs.spectral
            if band is not None and band.common_name in ("red", "green", "blue"):
                drawing[band.common_name] = str(name)
        if len(drawing) != 3:
            return None
        return drawing["red"], drawing["green"], drawing["blue"]

    @property
    def nodata(self) -> dict[Hashable, float | int | None]:
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
    def flags(self) -> dict[Hashable, attrs.CFFlags | None]:
        """Read what each variable's values mean, None where they are no classes."""
        return {name: self._data[name].gs.flags for name in self.variables}

    @property
    def packing(self) -> dict[Hashable, attrs.CFPacking | None]:
        """Read how each variable's stored values decode, None where they are physical."""
        return {name: self._data[name].gs.packing for name in self.variables}

    @property
    def spectral(self) -> dict[Hashable, attrs.EOBand | None]:
        """Read where in the spectrum each variable was measured, None where unknown."""
        return {name: self._data[name].gs.spectral for name in self.variables}

    @property
    def stac_collection(self) -> pystac.Collection | None:
        """Read the STAC Collection this raster was loaded from, None where it states none.

        Examples:
            >>> stac.load_dataset(items, geobox=grid).gs.stac_collection.id
            'sentinel-2-l2a'
        """
        import pystac

        stated = attrs.StacCollection.from_attrs(self._data.attrs)
        if stated is None or stated.collection is None:
            return None
        return pystac.Collection.from_dict(stated.collection)

    @property
    def stac_common_properties(self) -> dict[str, Any]:
        """Derive properties present and equal across the selected observations.

        Examples:
            >>> stac.load_dataset(items, geobox=grid).gs.stac_common_properties["platform"]
            'sentinel-2a'
        """
        from geosave.geodata.stac.raster import read_properties_coordinate
        from geosave.geodata.utils.mapping import intersect_mappings

        properties = read_properties_coordinate(self._data)
        return intersect_mappings(properties) if properties else {}

    @property
    def stac_properties(self) -> pd.DataFrame:
        """Read each date's Item properties from the `stac_properties` coordinate.

        Returns:
            One row per `time` label, one column per property key; a key a
            date lacks is NaN. Empty where the coordinate is absent.

        Examples:
            >>> clear = ds.sel(time=ds.gs.stac_properties["eo:cloud_cover"] < 30)
        """
        from geosave.geodata.stac.raster import read_properties_coordinate

        mappings = read_properties_coordinate(self._data)
        if not mappings:
            return pd.DataFrame()
        coordinate = self.time_coordinate
        if coordinate is None:
            return pd.DataFrame(mappings)
        labels = np.atleast_1d(coordinate.values)
        return pd.DataFrame(mappings, index=pd.Index(labels, name=coordinate.name))

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

    def reproject(
        self,
        how: GeoBox | SomeCRS,
        *,
        resampling: Resampling = "nearest",
        resolution: SomeResolution | None = None,
    ) -> Dataset:
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

        raster = self._data

        raster.gs.validate()
        if resampling not in {"nearest", "mode"} and any(
            raster[name].gs.is_categorical for name in raster.gs.variables
        ):
            raise ValueError(
                "class codes require a value-preserving kernel: nearest or mode"
            )
        source = cast("GeoBox", raster.gs.geobox)
        if isinstance(how, GeoBox):
            geobox = how
        else:
            size = "auto" if resolution is None else resolution
            geobox = raster.gs.output_geobox(how, resolution=size)

        dims = cast("tuple[str, str]", raster.gs.spatial_dims)
        raster = raster.transpose(..., *dims)
        if geobox == source:
            return cast("Dataset", raster)

        crs_coord = raster.odc.crs_coord
        crs_names = [] if crs_coord is None else [crs_coord.name]
        coordinates = (
            raster.coords.to_dataset()
            .drop_dims(dims, errors="ignore")
            .drop_vars(crs_names, errors="ignore")
        )
        warped = raster[list(raster.gs.variables)].odc.reproject(
            geobox, resampling=resampling
        )
        # ODC Dataset.map keeps the source CRS and drops coordinate-only dimensions.
        crs_name = str(warped.odc.crs_coord.name)
        placed_crs = xr_coords(geobox, crs_coord_name=crs_name)[crs_name]
        return cast(
            "Dataset",
            warped.assign_coords({crs_name: placed_crs}).assign_coords(
                coordinates.coords
            ),
        )

    def mask(
        self, mask: xr.DataArray | np.ndarray, *, fill: float | int | None = None
    ) -> Dataset:
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

        self._require_spatial_dims()

        if isinstance(mask, xr.DataArray):
            foreign = set(mask.dims) - set(self._data.dims)
            if foreign:
                raise ValueError(
                    f"mask spans axes the raster does not carry: {foreign}"
                )

        variables = {}
        if isinstance(mask, np.ndarray):
            for name in self.variables:
                variable = self._data[name]
                variables[name] = mask_with_np(variable, mask, fill=fill).variable
        else:
            for name in self.variables:
                variable = self._data[name]
                variables[name] = mask_with_da(variable, mask, fill=fill).variable

        masked = self._data.assign(variables)
        masked.encoding.pop("source", None)
        return cast("Dataset", masked)

    def crop(self, region: Region, *, mask: bool = True) -> Dataset:
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

        return cast("Dataset", crop(self._data, region, mask=mask))

    def crop_to_grid(
        self,
        geobox: GeoBox,
        *,
        mode: Padding = "constant",
        constant_value: int | float = 0,
    ) -> Dataset:
        """Take the pixels of an aligned grid, padding where it leaves this raster.

        Unlike `reproject`, nothing is resampled or computed.

        Args:
            geobox: Grid sharing this raster's CRS and pixel size, offset by
                whole pixels, such as `chips.geobox(key)`. It may extend past
                the raster edges.
            mode: How pixels outside the raster are filled, as in `numpy.pad`.
            constant_value: Fill for `"constant"`, keeping the dtype.

        Returns:
            Lazy Dataset on exactly `geobox`. A declared extent is clipped to it;
            time and other coordinates, nodata and other attrs are kept.

        Raises:
            ValueError: This raster has no regular grid, or `geobox` is not
                pixel-aligned with it or does not overlap it.

        Examples:
            >>> chip = ds.gs.crop_to_grid(ds.gs.geobox[0:224, 0:224])
            >>> chip.gs.region.extent.centroid
            <Geometry ...>
        """

        from .transform.extent import crop_to_grid

        return cast(
            "Dataset",
            crop_to_grid(self._data, geobox, mode=mode, constant_value=constant_value),
        )

    @property
    def region(self) -> GeoRegion:
        """Read the declared extent or union of spatial variable extents, with the time span.

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
        geometry = None
        for name in self.variables:
            variable = self._data[name]
            if variable.gs.spatial_dims is None:
                continue
            member = variable.gs.region.extent
            if geometry is None:
                geometry = member
            elif geometry != member:
                geometry = geometry.union(member.to_crs(geometry.crs))
        if geometry is not None:
            return GeoRegion(geometry, timespan=self.timespan)
        self.validate()
        geobox = cast("GeoBox", self.geobox)
        return GeoRegion(geobox.extent, timespan=self.timespan)

    def write_extent(self, extent: Geometry | None) -> Dataset:
        """Record the declared extent on the Dataset and its spatial variables.

        Args:
            extent: CRS-aware geometry to preserve, or None to clear it.

        Returns:
            Shallow copy retaining pixels and grid. Selected spatial variables
            retain the declaration; nonspatial variables remain unchanged.

        Raises:
            ValueError: The extent is invalid, empty, lacks a CRS or is not 2D.

        Examples:
            >>> located = ds.gs.write_extent(city)
            >>> located.red.gs.region.extent == city
            True
        """
        if extent is None:
            written = attrs.clear_attrs(self._data, attrs.ACDDExtent)
        else:
            written = attrs.stamp_attrs(
                self._data, attrs.ACDDExtent.from_geometry(extent)
            )
        for name, variable in written.data_vars.items():
            if variable.gs.spatial_dims is not None:
                written[name] = variable.gs.write_extent(extent)
        return written

    def clip_extent(self, geometry: Geometry) -> Dataset:
        """Narrow the declared extents to a geometry, as a crop narrows coverage.

        Args:
            geometry: CRS-aware geometry the pixels now cover at most.

        Returns:
            Shallow copy whose root and variable extents are each intersected
            with `geometry`, cleared where they do not meet.
        """
        raster = self._data
        extent = attrs.ACDDExtent.from_attrs(raster.attrs)
        if extent is not None:
            narrowed = extent.clip(geometry)
            if narrowed is None:
                raster = attrs.clear_attrs(raster, attrs.ACDDExtent)
            else:
                raster = attrs.stamp_attrs(raster, narrowed)
        else:
            raster = raster.copy(deep=False)
        for name in self.variables:
            raster[name] = raster[name].gs.clip_extent(geometry)
        return cast("Dataset", raster)

    def write_nodata(
        self,
        value: float | int | None,
        *,
        target: list[str] | None = None,
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
            >>> ds.gs.write_nodata(-9999, target=["elevation"])
        """
        written = self._data.copy()
        for name in self._targets(target):
            written[name] = self._data[name].gs.write_nodata(value)
        return written

    def write_flags(
        self,
        classes: Mapping[int, str] | None,
        colors: Palette | None = None,
        *,
        target: list[str] | None = None,
    ) -> Dataset:
        """Write the class each stored value stands for, and its colour.

        Plots then name and colour the classes, and a GeoTIFF or Zarr written
        afterwards carries them.

        Args:
            classes: Stored value mapped to class name. None removes the
                flags and colours.
            colors: Stored value mapped to hex or RGB colour. None writes
                names only.
            target: Variable names to write, defaulting to every data variable.

        Returns:
            New Dataset whose named variables carry the flags.

        Raises:
            KeyError: `target` names something that is not a data variable.
            ValueError: `colors` colours a value `classes` does not name.

        Examples:
            >>> labels.gs.write_flags({0: "water", 1: "trees"}, {0: "#419bdf", 1: "#397d49"})
        """
        written = self._data.copy()
        for name in self._targets(target):
            written[name] = self._data[name].gs.write_flags(classes, colors)
        return written

    def write_packing(
        self,
        packing: attrs.CFPacking | None,
        *,
        target: list[str] | None = None,
    ) -> Dataset:
        """Write how stored values decode to physical values; pixels are unchanged.

        Args:
            packing: Scale, offset and fill, as CF packs them. None removes
                every packing and missing-value attr.
            target: Variable names to write, defaulting to every data variable.

        Returns:
            New Dataset whose named variables carry the packing.

        Raises:
            KeyError: `target` names something that is not a data variable.

        Examples:
            >>> packing = attrs.CFPacking(scale_factor=1e-4, add_offset=-0.1)
            >>> ds.gs.write_packing(packing, target=["B04", "B08"])
        """
        written = self._data.copy()
        for name in self._targets(target):
            written[name] = self._data[name].gs.write_packing(packing)
        return written

    def write_spectral(
        self,
        common_name: str | None,
        *,
        center_wavelength: float | None = None,
        full_width_half_max: float | None = None,
        target: list[str] | None = None,
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
            >>> ds.gs.write_spectral("red", center_wavelength=0.665, target=["B04"])
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
        colour composite drawing near-infrared as red. The GeoTIFF writer
        maps the choice to GDAL colour interpretations.

        Args:
            red: Variable drawn as the red channel.
            green: Variable drawn as the green channel.
            blue: Variable drawn as the blue channel.

        Returns:
            New Dataset whose attrs state the composite.

        Raises:
            KeyError: A name is not a data variable.
            ValueError: One variable is named for two channels.

        Examples:
            >>> ds.gs.write_rgb("B04", "B03", "B02").gs.plot()
        """
        channels = [red, green, blue]
        self._targets(channels)
        if len(set(channels)) < len(channels):
            raise ValueError(
                f"one variable cannot draw two channels, but these name "
                f"{channels}; give each channel its own variable"
            )
        display = attrs.Display.from_attrs(self._data.attrs) or attrs.Display()
        return attrs.stamp_attrs(
            self._data, display.model_copy(update={"rgb": channels})
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
        return statistics([self._data[list(self.variables)]])

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
        for name in self.variables:
            band = self._data[name]
            flags = band.gs.flags
            if flags is not None and flags.classes is not None:
                columns[name] = class_percent([band])
        return pd.DataFrame(columns)

    def expand_stac(
        self,
        *,
        format_id: str = "{time:%Y%m%dT%H%M%S}",
        asset: str | None = None,
    ) -> DataTree:
        """Expand a collection's Dataset into a STAC tree: an Item per time, assets as leaves.

        Each `time` step becomes an Item node holding the coordinates along
        `time`; every other coordinate and the Dataset attrs sit once on the
        top node, the collection. A Dataset without `time` is one Item.

        Args:
            format_id: Item ID template over each observation's scalar
                coordinates, such as ``"S2_{tile}_{time:%Y%m%d}"``.
            asset: None makes each variable its own asset; a name makes every
                variable a band of that one asset, as one multi-band file.

        Returns:
            Lazy tree of ``{item}/{asset}`` leaves, rooted at the collection.

        Raises:
            ValueError: A variable lacks `time`, so it belongs to no Item; two
                observations share an ID; or `format_id` names a field the
                observation lacks.

        Examples:
            >>> [leaf.path for leaf in scene.gs.expand_stac().leaves][:2]
            ['/20250601T103000/red', '/20250601T103000/nir']
            >>> scene.gs.expand_stac(asset="image").gs.to_cog("out")
            [UPath('/abs/out/20250601T103000/image.tif'), ...]
        """
        from geosave.geodata.io.paths import read_scalar_coords

        raster = self._data
        observations = [raster]
        if TIME_COORDINATE in raster.dims:
            for name, variable in raster.data_vars.items():
                if TIME_COORDINATE not in variable.dims:
                    raise ValueError(
                        f"{name!r} has no {TIME_COORDINATE!r}, so it belongs to no "
                        "Item; expand it as its own collection"
                    )
            observations = []
            for index in range(raster.sizes[TIME_COORDINATE]):
                observations.append(raster.isel({TIME_COORDINATE: index}))

        # Coordinates along time describe each Item; the rest is the collection's.
        own = []
        for name, coord in raster.coords.items():
            if name == TIME_COORDINATE or TIME_COORDINATE in coord.dims:
                own.append(name)
        shared = [name for name in raster.coords if name not in own]

        top = xr.Dataset(coords=raster.coords.to_dataset()[shared].coords)
        nodes = {"/": top.assign_attrs(raster.attrs)}
        for observation in observations:
            labels = read_scalar_coords(observation)
            try:
                item = format_id.format_map(labels)
            except KeyError as missing:
                raise ValueError(
                    f"format_id needs {missing}, which the observation lacks; "
                    "name its Item with a format_id over its own coordinates"
                ) from None
            if item in nodes:
                raise ValueError(
                    f"two observations format to one id {item!r}; give format_id a "
                    "field that tells them apart"
                )
            facts = {}
            for name in own:
                facts[name] = observation[name].variable
            nodes[item] = xr.Dataset(coords=facts)

            # The top node holds the Dataset attrs; a variable keeps its own.
            leaf = observation.drop_vars([*own, *shared]).drop_attrs(deep=False)
            if asset is not None:
                nodes[f"{item}/{asset}"] = leaf
                continue
            for name in leaf.data_vars:
                nodes[f"{item}/{name}"] = leaf[[name]]
        return cast("DataTree", xr.DataTree.from_dict(nodes))

    def to_array(
        self,
        *,
        dtype: DTypeLike | None = None,
        combine_attrs: CombineAttrsOptions = "drop_conflicts",
    ) -> DataArray:
        """Stack variables lazily along `band`, in their existing order.

        Stacked pixels share one meaning, so every variable must state the same
        `CFPacking`, `CFFlags` and units; those stay on the array whatever
        `combine_attrs` decides for the descriptive rest.

        Args:
            dtype: Cast each variable before stacking. None requires all
                variables to have the same dtype.
            combine_attrs: Native policy for the remaining variable attrs.

        Returns:
            Array shaped `(*axes, band, y, x)`. No packing is applied.

        Raises:
            ValueError: Variables are absent, span different axes, state
                different pixel meaning, or need an explicit common dtype.
            RasterValidationError: ODC finds no spatial dimensions.

        Examples:
            >>> image = ds[["B04", "B08"]].gs.to_array(dtype="float32")
            >>> image.dims
            ('time', 'band', 'y', 'x')
        """
        names = list(self.variables)
        if not names:
            raise ValueError(
                "this raster carries no variables to stack; it holds only coordinates"
            )
        if BAND_DIMENSION in self._data.dims:
            raise ValueError(
                f"this raster already spans {BAND_DIMENSION!r}, which stacking "
                "would name; rename it first"
            )
        spatial = self._require_spatial_dims()
        first = self._data[names[0]]
        pixel_keys = (*attrs.CFPacking.KEYS, *attrs.CFFlags.KEYS, "units")
        for name in names[1:]:
            band = self._data[name]
            if band.dims != first.dims:
                raise ValueError(
                    f"{name!r} spans {band.dims} but {names[0]!r} spans "
                    f"{first.dims}; stack variables sharing one shape"
                )
            if dtype is None and band.dtype != first.dtype:
                raise ValueError(
                    f"{names[0]!r} and {name!r} carry different dtypes; "
                    "pass dtype= to choose one"
                )
            if band.gs.packing != first.gs.packing or band.gs.flags != first.gs.flags:
                raise ValueError(
                    f"{names[0]!r} and {name!r} state different CF packing or "
                    "flags; decode them with mask_and_scale() before stacking"
                )
            if band.attrs.get("units") != first.attrs.get("units"):
                raise ValueError(
                    f"{names[0]!r} and {name!r} state different units, so "
                    "stacked pixels would mean different things"
                )

        descriptive = []
        for name in names:
            kept = dict(self._data[name].attrs)
            for key in pixel_keys:
                kept.pop(key, None)
            descriptive.append(kept)
        stacked = (
            self._data[names] if dtype is None else self._data[names].astype(dtype)
        )
        array = stacked.to_dataarray(dim=BAND_DIMENSION).transpose(
            ..., BAND_DIMENSION, *spatial
        )
        shared = {key: first.attrs[key] for key in pixel_keys if key in first.attrs}
        array.attrs = {
            **attrs.combine_attrs(descriptive, policy=combine_attrs),
            **shared,
        }
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
        return Explorer(layers=[layer], basemap=basemap)

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

        A band draws through `cmap`; one whose CF flags list classes draws
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

    def to_numpy(self, *, dtype: DTypeLike | None = None) -> np.ndarray:
        """Stack every variable this raster carries into one model-input array.

        Every axis besides the grid leads, then the variables, then the grid
        pair — torchgeo's `[T, C, H, W]`. No batch axis is added, so one raster
        is one sample. Order the variables with xarray before stacking them.

        Args:
            dtype: Cast applied to each variable before stacking. None keeps the source
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
        return self.to_array(dtype=dtype).values

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

    def mask_and_scale(self) -> Dataset:
        """Decode the Dataset's CF values, coordinates and relationships lazily.

        Consumed declarations move from attrs to encoding for native writers.
        """
        decoded = xr.decode_cf(self._data.copy(deep=False), decode_coords="all")
        return cast("Dataset", decoded)

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

    def to_cog(self, path: Location, **options: Any) -> list[UPath]:
        """A Dataset is not COG-shaped: a COG holds one instant of one grid.

        Write ``ds.gs.expand_stac(...).gs.to_cog(path)`` for one file per asset,
        or ``array.gs.to_cog(path)`` for one array.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "a Dataset is not COG-shaped; use ds.gs.expand_stac(...).gs.to_cog(path) "
            "or DataArray.gs.to_cog(path)"
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
