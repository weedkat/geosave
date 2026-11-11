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

import itertools

from typing import TYPE_CHECKING, Any, Literal, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata.attrs.groups.gdal import RGB
from geosave.geodata.conventions import (
    BAND_DIMENSION,
)
from geosave.geodata.core.raster.utils import grid
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
    from collections.abc import Callable
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

    from odc.geo.gcp import GCPGeoBox

    from geosave.geodata import DataArray, Dataset
    from dask.delayed import Delayed
    from upath import UPath
    from geosave.geodata.io.formats.geotiff import COGWriteOptions
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
        for name in self.raster_variables:
            variable = data[name]
            if attrs.Packing.is_stated_in(variable.attrs):
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
            self._select_pixels(),
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

    def _select_pixels(self) -> xr.Dataset:
        """Hold the CRS carriers as coordinates, so only pixel variables remain data."""
        return self._data.set_coords(list(self.support_variables))

    def _read_variable(self, name: Hashable) -> xr.DataArray:
        """Retain native coordinates and attach referenced scalar CRS carriers.

        CF selection also attaches ancillary pixel arrays. Those remain peer
        data variables during pixel operations, otherwise their unchanged
        coordinates conflict with independently transformed values.
        """
        variable = self._data[name]
        related = self._data.cf[name]
        carriers = {}
        for coordinate in self.support_variables:
            if coordinate in related.coords and not related[coordinate].dims:
                carriers[coordinate] = related[coordinate]
        return variable.assign_coords(carriers)

    @property
    def spatial_dims(self) -> tuple[str, str] | None:
        """Return the common spatial axes, or None for missing or differing values."""
        shared = None
        for name in self.raster_variables:
            variable = self._read_variable(name)
            current = variable.gs.spatial_dims
            if current is None or (shared is not None and current != shared):
                return None
            shared = current
        return shared

    @property
    def geobox(self) -> GeoBox | GCPGeoBox | None:
        """Return the common grid, or None for missing or differing values."""
        shared = None
        for name in self.raster_variables:
            variable = self._read_variable(name)
            current = variable.gs.geobox
            if current is None or (shared is not None and current != shared):
                return None
            shared = current
        return shared

    def validate(
        self,
        *,
        georeferenced: bool = False,
        affine: bool = False,
        same_grid: bool = False,
        same_dims: bool = False,
    ) -> None:
        """Validate each raster and optionally compare spatial structure.

        Unreferenced pixels and different grids are valid by default.
        Spatial coordinates and existing placement are always checked without
        reading pixels. Empty inputs fail.

        Args:
            georeferenced: Require placement and a CRS for every raster.
            affine: Require one affine `GeoBox` on spatial axes shared by all
                variables, so `gs.geobox` is a GeoBox. Implies `same_grid`
                and `same_dims`.
            same_grid: Require known, equal ODC grids across all variables.
            same_dims: Require matching spatial axis names across all variables.

        Raises:
            RasterValidationError: A raster is invalid or a requested
                guarantee fails. The error identifies the variable.

        Examples:
            >>> scene.gs.validate(georeferenced=True, same_grid=True)
        """
        same_grid = same_grid or affine
        same_dims = same_dims or affine
        variables = self._data[list(self.raster_variables)].data_vars
        self._validate_nonempty(variables)
        reference = None
        for name in self.raster_variables:
            variable = self._read_variable(name)
            try:
                variable.gs.validate(georeferenced=georeferenced, affine=affine)
                if same_grid:
                    self._validate_grid(variable)
                if reference is None:
                    reference = variable
                    continue
                if same_dims:
                    self._validate_same_dims(variable, reference)
                if same_grid:
                    self._validate_same_grid(variable, reference)
            except RasterValidationError as error:
                error.variable = str(variable.name)
                raise

    def _validate_nonempty(self, variables: Mapping[Hashable, xr.DataArray]) -> None:
        """Require at least one data-bearing variable."""
        if not variables:
            raise RasterValidationError(
                "There are no raster variables to validate. Select or load raster data first."
            )

    def _validate_spatial(self) -> None:
        """Check structure before replacing existing placement coordinates."""
        self._validate_nonempty(self._data[list(self.raster_variables)].data_vars)
        for name in self.raster_variables:
            variable = self._read_variable(name)
            dims = variable.gs.spatial_dims
            if dims is None or any(variable.sizes[dim] == 0 for dim in dims):
                raise RasterValidationError(
                    "Writing a grid requires identifiable, nonempty spatial dimensions.",
                    variable=str(name),
                )

    def _validate_same_dims(
        self, variable: xr.DataArray, reference: xr.DataArray
    ) -> None:
        """Require spatial axis names to match the first variable."""
        if variable.gs.spatial_dims != reference.gs.spatial_dims:
            raise RasterValidationError(
                "Variables use different spatial dimensions. Rename their spatial axes "
                "consistently, for example with to_yx, before combining them.",
                variable=str(variable.name),
            )

    def _validate_grid(self, variable: xr.DataArray) -> None:
        """Require known placement before comparing grids."""
        if variable.gs.geobox is None:
            raise RasterValidationError(
                "Cannot compare absent grids. Supply known placement with "
                "raster.gs.write_grid(geobox) before requesting same_grid."
            )

    def _validate_same_grid(
        self, variable: xr.DataArray, reference: xr.DataArray
    ) -> None:
        """Require equal known ODC placement."""
        if variable.gs.geobox != reference.gs.geobox:
            raise RasterValidationError(
                "Variables do not share one grid. Reproject each variable "
                "onto the same target grid before combining them.",
                variable=str(variable.name),
            )

    @property
    def variables(self) -> tuple[str, ...]:
        """Name the data variables.

        Returns:
            Variable names, in Dataset order.
        """
        return tuple(str(name) for name in self._data.data_vars)

    @property
    def support_variables(self) -> tuple[Hashable, ...]:
        """Name the scalar CRS carriers, including coordinate variables."""
        from .utils.cf import read_support_variables

        return read_support_variables(self._data)

    @property
    def raster_variables(self) -> tuple[Hashable, ...]:
        """Name data variables excluding CRS carriers.

        Unknown nonspatial data variables remain included so raster validation
        reports them rather than silently discarding them.
        """
        support = set(self.support_variables)
        return tuple(name for name in self._data.data_vars if name not in support)

    def expand_rasters(self, dim: str | Sequence[str]) -> tuple[Dataset, ...]:
        """Split this Dataset into one lazy slice per position along `dim`.

        Args:
            dim: Dimension, or dimensions in order, to split along. An empty
                sequence keeps the Dataset whole.

        Returns:
            One slice per position, in order. Each slice carries its labels as
            scalar coordinates; a dimension without coordinates labels each
            slice by its position. Pixels stay lazy.

        Raises:
            ValueError: A dimension is repeated, absent, or spatial.

        Examples:
            >>> [chip.time.dt.month.item() for chip in scene.gs.expand_rasters("time")]
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
            slices.append(cast("Dataset", piece.assign_coords(unlabelled)))
        return tuple(slices)

    def _targets(self, target: str | Sequence[str] | None) -> list[Hashable]:
        """Return the data variables `target` names, every one when None.

        Raises:
            KeyError: A name is not a data variable.
        """
        if target is None:
            return list(self.raster_variables)
        names: list[Hashable] = [target] if isinstance(target, str) else list(target)
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
        for name in reversed(self.raster_variables):
            stated = attrs.GDALBand.from_attrs(self._data[name].attrs)
            if stated is not None and stated.colorinterp in RGB:
                drawing[stated.colorinterp] = str(name)
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
            >>> stac.load_dataset(items, geobox=grid).gs.stac_collection.id
            'sentinel-2-l2a'
        """
        import pystac

        stated = attrs.StacMetadata.from_attrs(self._data.attrs)
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

    def write_grid(self, geobox: GeoBox | GCPGeoBox) -> Dataset:
        """Assign grid coordinates and CRS, preserving spatial names and axis order.

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
            RasterValidationError: Spatial dimensions are missing or inconsistent,
                the pixel shape differs from the grid, or the grid carries no CRS.

        Examples:
            >>> scene = xr.Dataset({"red": (("time", "y", "x"), red)}, coords={"time": dates})
            >>> scene.gs.write_grid(grid).gs.geobox == grid
            True
        """
        self._validate_spatial()
        placed = grid.write_grid(self._select_pixels(), geobox)
        for name in placed.gs.raster_variables:
            placed[name] = grid.clear_mapping_attr(placed[name]).variable
        return cast("Dataset", placed)

    def reproject(
        self,
        how: GeoBox | SomeCRS,
        *,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | None = None,
    ) -> Dataset:
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

        from .transform.warp import read_kernel
        from .utils.grid import resolve_grid

        self.validate(georeferenced=True, affine=True)
        source = cast("GeoBox", self.geobox)
        geobox = resolve_grid(source, how, resolution)
        if geobox == source:
            return cast("Dataset", self._data)
        variables = {}
        for name in self.raster_variables:
            variable = self._read_variable(name)
            kernel = read_kernel(resampling, str(name))
            variables[name] = variable.gs.reproject(geobox, resampling=kernel)
        # ODC's Dataset path retains the old CRS carrier; assemble warped variables.
        warped = xr.Dataset(variables).gs.write_crs().assign_attrs(self._data.attrs)
        spatial = cast("tuple[str, str]", self.spatial_dims)
        for name, coordinate in self._data.coords.items():
            if name not in self.grid_mapping_coords and set(spatial).isdisjoint(
                coordinate.dims
            ):
                warped = warped.assign_coords({name: coordinate.variable})
        return cast("Dataset", warped)

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

        self.validate()

        if isinstance(mask, xr.DataArray):
            foreign = set(mask.dims) - set(self._data.dims)
            if foreign:
                raise ValueError(
                    f"mask spans axes the raster does not carry: {foreign}"
                )

        variables = {}
        if isinstance(mask, np.ndarray):
            for name in self.raster_variables:
                variable = self._read_variable(name)
                variables[name] = mask_with_np(variable, mask, fill=fill).variable
        else:
            for name in self.raster_variables:
                variable = self._read_variable(name)
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

        Unlike `read_pixels`, nothing is resampled, converted or computed.

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

        extent = attrs.Extent.from_attrs(self._data.attrs)
        if extent is not None:
            return GeoRegion(extent.to_geometry(), timespan=self.timespan)
        geometry = None
        for name in self.raster_variables:
            variable = self._read_variable(name)
            if variable.gs.spatial_dims is None:
                continue
            member = variable.gs.region.extent
            if geometry is None:
                geometry = member
            elif geometry != member:
                geometry = geometry.union(member.to_crs(geometry.crs))
        if geometry is not None:
            return GeoRegion(geometry, timespan=self.timespan)
        self.validate(georeferenced=True, same_grid=True)
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
            written = attrs.clear_attrs(self._data, attrs.Extent)
        else:
            written = attrs.stamp_attrs(self._data, attrs.Extent.from_geometry(extent))
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
        extent = attrs.Extent.from_attrs(raster.attrs)
        if extent is not None:
            narrowed = extent.clip(geometry)
            if narrowed is None:
                raster = attrs.clear_attrs(raster, attrs.Extent)
            else:
                raster = attrs.stamp_attrs(raster, narrowed)
        else:
            raster = raster.copy(deep=False)
        for name in self.raster_variables:
            raster[name] = raster[name].gs.clip_extent(geometry)
        return cast("Dataset", raster)

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
            RasterValidationError: Spatial placement is missing or inconsistent,
                has no CRS, or uses ground control points instead of an affine grid.
            ValueError: The supplied CRS cannot be parsed.

        Examples:
            >>> ds.gs.write_crs("EPSG:32633").y.attrs["standard_name"]
            'projection_y_coordinate'
        """
        written = grid.write_crs(self._data, crs)
        for name in written.gs.raster_variables:
            written[name] = grid.clear_mapping_attr(written[name]).variable
        return cast("Dataset", written)

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

    def write_scale_offset(
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
            >>> ds.gs.write_scale_offset(1e-4, -0.1, target=["B04", "B08"])
        """
        written = self._data.copy()
        for name in self._targets(target):
            written[name] = self._data[name].gs.write_scale_offset(
                scale_factor, add_offset
            )
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
        bands: dict[Hashable, xr.DataArray] = {}
        for name in self.raster_variables:
            band = self._data[name]
            stated = attrs.GDALBand.from_attrs(band.attrs) or attrs.GDALBand()
            interpreted = stated.model_copy(
                update={"colorinterp": drawn.get(str(name))}
            )
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
        return statistics([self._data[list(self.raster_variables)]])

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
        for name in self.raster_variables:
            band = self._data[name]
            legend = band.gs.legend
            if legend is not None and legend.classes is not None:
                columns[name] = class_percent([band])
        return pd.DataFrame(columns)

    def to_dataarray(
        self,
        *,
        dtype: DTypeLike | None = None,
        combine_attrs: CombineAttrsOptions = "drop_conflicts",
    ) -> DataArray:
        """Stack variables lazily in their existing order, without broadcasting.

        Args:
            dtype: Cast each variable before stacking. None requires all
                variables to have the same dtype.
            combine_attrs: Native xarray attr policy or callable. Conflicting
                keys are dropped by default; dropped keys produce a warning.

        Returns:
            Array shaped `(*axes, band, y, x)` with variable attrs combined
            by the selected policy. No packing or display scaling is applied.
            Extra-dimensional CF support is omitted, as for native DataArray
            extraction; bounds references to omitted variables are removed.

        Raises:
            ValueError: Variables are absent, already span `band`, have
                different non-spatial axes or CF packing, or need an explicit
                common dtype. Decode differing packing before stacking.
            RasterValidationError: Variables do not share spatial dimensions.

        Examples:
            >>> image = ds[["B04", "B08"]].gs.to_dataarray(dtype="float32")
            >>> image.dims
            ('time', 'band', 'y', 'x')
        """
        names = list(self.raster_variables)
        if not names:
            raise ValueError(
                "this raster carries no variables to stack; it holds only coordinates"
            )
        if BAND_DIMENSION in self._data.dims:
            raise ValueError(
                f"this raster already carries a {BAND_DIMENSION!r} dimension, "
                f"which the stacked variables would name; rename it first"
            )

        self.validate(same_dims=True)
        spatial = cast("tuple[str, str]", self.spatial_dims)
        first = self._data[names[0]]
        packing = first.gs.packing or attrs.Packing()
        axes = tuple(dim for dim in first.dims if dim not in spatial)
        variables = {}
        metadata = []
        for name in self.raster_variables:
            band = self._read_variable(name)
            dims = tuple(dim for dim in band.dims if dim not in spatial)
            if dims != axes:
                raise ValueError(
                    f"{name!r} carries different axes from {names[0]!r}'s "
                    f"{axes}; stack variables sharing one shape"
                )
            if dtype is None and band.dtype != first.dtype:
                raise ValueError(
                    f"{names[0]!r} and {name!r} carry different dtypes; "
                    "pass dtype= to choose one"
                )

            if (band.gs.packing or attrs.Packing()) != packing:
                raise ValueError(
                    "stacking requires matching CF packing; decode each variable "
                    "with mask_and_scale() before stacking"
                )

            metadata.append(band.attrs)
            if dtype is not None:
                band = band.astype(dtype)
            variables[name] = band

        dataset = self._data[names].assign(variables)
        array = dataset.to_dataarray(dim=BAND_DIMENSION).transpose(
            *axes, BAND_DIMENSION, *spatial
        )
        array.attrs = attrs.combine_attrs(metadata, policy=combine_attrs)
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
            Dataset on `geobox` with the same variables, non-spatial coordinates
            and attrs, float32 with nodata and outside pixels as NaN. Spatial
            axis names follow the target grid; dimension order is preserved.

        Examples:
            >>> ds.gs.read_pixels(view).red.shape
            (600, 900)
        """

        from .transform.packing import mask_missing
        from .utils.grid import clear_mapping_attr
        from geosave.geodata.conventions import GRID_MAPPING

        from odc.geo.overlap import compute_reproject_roi
        from odc.geo.roi import roi_is_empty
        from odc.geo.xr import xr_coords, xr_reproject

        spatial = self.spatial_dims
        raster = self._select_pixels()
        try:
            raster.gs.validate(georeferenced=True, affine=True)
        except RasterValidationError as error:
            error.operation = "Cannot read the requested window"
            raise
        grid = cast("GeoBox", raster.gs.geobox)
        roi = compute_reproject_roi(grid, geobox)

        spatial = cast("tuple[str, str]", raster.gs.spatial_dims)
        dimensions: dict[Hashable, str] = dict(
            zip(spatial, geobox.dimensions, strict=True)
        )
        carriers = raster.gs.grid_mapping_coords
        coords = {}
        for name, coord in raster.coords.items():
            if set(spatial).isdisjoint(coord.dims) and name not in carriers:
                coords[name] = coord
        coords.update(xr_coords(geobox))

        # A missed window needs no source pixels, including lazy time slices.
        if roi_is_empty(roi.roi_src):
            missing = {}
            sizes: dict[Hashable, int] = dict(zip(spatial, geobox.shape, strict=True))
            for name, variable in raster.data_vars.items():
                shape = tuple(
                    sizes.get(dim, variable.sizes[dim]) for dim in variable.dims
                )
                missing[name] = xr.DataArray(
                    np.full(shape, np.nan, "float32"),
                    dims=tuple(dimensions.get(dim, dim) for dim in variable.dims),
                    attrs=variable.attrs,
                )
            pixels = xr.Dataset(
                missing, coords=coords, attrs=raster.attrs
            ).gs._map_variables(mask_missing)
        else:
            y_window, x_window = roi.roi_src
            steps = {
                spatial[0]: slice(y_window.start, y_window.stop, roi.read_shrink),
                spatial[1]: slice(x_window.start, x_window.stop, roi.read_shrink),
            }
            window = (
                raster.isel(steps).gs._map_variables(mask_missing).astype("float32")
            )
            # Thinning can leave singleton axes, which cannot establish spacing.
            # Preserve the selected pixel centers in an explicit source grid.
            from affine import Affine

            shrink = roi.read_shrink
            selected_grid = grid[roi.roi_src]
            transform = (
                selected_grid.transform
                * Affine.translation((1 - shrink) / 2, (1 - shrink) / 2)
                * Affine.scale(shrink)
            )
            selected_grid = GeoBox(
                (window.sizes[spatial[0]], window.sizes[spatial[1]]),
                transform,
                grid.crs,
            )
            window = window.gs.write_grid(selected_grid)
            # ODC requires the spatial pair to be adjacent and ordered.
            for name, variable in window.data_vars.items():
                axes = [dim for dim in variable.dims if dim not in spatial]
                window[name] = variable.transpose(*axes, *spatial)
            window = window.rename(dict(zip(spatial, ("y", "x"), strict=True)))
            pixels = xr_reproject(
                window.compute(), geobox, resampling="nearest", dst_nodata=np.nan
            ).assign_coords(coords)
            for name, variable in raster.data_vars.items():
                order = tuple(dimensions.get(dim, dim) for dim in variable.dims)
                pixels[name] = pixels[name].transpose(*order)

        pixels = pixels.odc.assign_crs(geobox.crs).assign_coords(
            {GRID_MAPPING: xr_coords(geobox)[GRID_MAPPING]}
        )
        for name in pixels.gs.raster_variables:
            pixels[name] = clear_mapping_attr(pixels[name]).variable
        return cast("Dataset", pixels)

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
        return self.to_dataarray(dtype=dtype).values

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

    def _map_variables(self, change: Callable[[xr.DataArray], xr.DataArray]) -> Dataset:
        """Apply one per-variable change to every data variable."""
        variables = {}
        for name in self.raster_variables:
            variable = self._read_variable(name)
            variables[name] = change(variable).variable
        raster = self._data.assign(variables)
        # Its pixels are no longer the file's.
        raster.encoding.pop("source", None)
        return cast("Dataset", raster)

    def _read_rasters_by_path(self) -> dict[str, Dataset]:
        """Lay this Dataset out as the one raster a publication splits."""
        from geosave.geodata.stac.publish import ROOT_PATH

        return {ROOT_PATH: self._data}

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
                "variable" gives each data variable its own asset.
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
                "variable" gives each data variable its own asset.
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
                "variable" gives each data variable its own asset.
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
