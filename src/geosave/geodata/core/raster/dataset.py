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

import geosave.geodata.attrs as attrs

from geosave.geodata.errors import RasterValidationError
from geosave.geodata.core.conventions import (
    BAND_DIMENSION,
    DEFAULT_COLLECTION,
    ITEM_ID_FORMAT,
    TIME_COORDINATE,
)
from geosave.geodata.core.raster.utils.statistics import class_percent, statistics
from geosave.geodata.core.raster.utils.tensor import as_tensor

from .base import GeoRaster, RasterStacProperties, RasterTime
from .transform.temporal import Reducer, Interpolation

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from geosave.geodata.utils.datetime import Freq
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
    from geosave.geodata.io.formats.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.formats.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from geosave.geodata.utils.color import Palette
    from geosave.geodata.viz.explore import Explorer


# Rules a Dataset can be asked to meet; `ds.gs.validate(same_pixel_meaning)`
# runs one.
def same_pixel_meaning(raster: xr.Dataset) -> None:
    """Require every variable to state the same CF packing, flags and units.

    Raises:
        RasterValidationError: Two variables state different pixel meaning.
    """
    names = list(raster.data_vars)
    if not names:
        return

    first = raster[names[0]]
    for name in names[1:]:
        band = raster[name]
        if (
            band.gs.packing != first.gs.packing
            or band.gs.flags != first.gs.flags
            or band.attrs.get("units") != first.attrs.get("units")
        ):
            raise RasterValidationError(
                f"It states different CF packing, flags or units than "
                f"{names[0]!r}, so stacked pixels would mean different things. "
                "Decode them with mask_and_scale() first.",
                variable=name,
            )


class GeoDataset(
    RasterTime["Dataset"], RasterStacProperties["Dataset"], GeoRaster["Dataset"]
):
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
                rechunked along dim. A per-scene STAC property no
                longer describes a bin and drops out; shared ones stay.
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

        bands = {}
        for name in self.variables:
            band = self._data[name]
            # A band without `dim`, such as a static elevation, passes through.
            if dim not in band.dims:
                bands[name] = band
                continue
            bands[name] = band.gs.composite(
                freq,
                method,
                dim=dim,
                interpolation=interpolation,
                closed=closed,
                label=label,
                origin=origin,
                offset=offset,
            )
        return cast("Dataset", xr.Dataset(bands, attrs=self._data.attrs))

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

    @property
    def variables(self) -> tuple[Hashable, ...]:
        """Name the data variables, in Dataset order."""
        return tuple(self._data.data_vars)

    @property
    def rgb(self) -> tuple[str, str, str] | None:
        """Read the variables `Display.rgb` names for red, green and blue.

        STAC loading and GeoTIFF opening state it from band colours; `write_rgb`
        states it explicitly.

        Returns:
            The three variable names, or None where none are stated or a named
            variable is no longer in the Dataset.

        Examples:
            >>> ds.gs.write_rgb("B04", "B03", "B02").gs.rgb
            ('B04', 'B03', 'B02')
        """
        display = attrs.Display.from_attrs(self._data.attrs)
        if display is None or display.rgb is None:
            return None

        # Selecting variables keeps the Dataset attrs, so a stated name may be gone.
        red, green, blue = display.rgb
        if red not in self._data or green not in self._data or blue not in self._data:
            return None

        return red, green, blue

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

        bands = {}
        for name in self.variables:
            bands[name] = self._data[name].gs.reproject(
                how, resampling=resampling, resolution=resolution
            )
        return cast("Dataset", xr.Dataset(bands, attrs=self._data.attrs))

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

        bands = {}
        for name in self.variables:
            bands[name] = self._data[name].gs.mask(mask, fill=fill)
        return cast("Dataset", xr.Dataset(bands, attrs=self._data.attrs))

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

    def _target_variables(self, target: list[str] | None) -> list[str]:
        """Select variables for metadata writes without treating a name as characters."""
        if isinstance(target, str):
            raise TypeError(
                "target must be a list of variable names, "
                f"for example target=[{target!r}]"
            )
        return self.variables if target is None else target

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
            target: List of variable names; None writes every data variable,
                and an empty list writes none.

        Returns:
            New Dataset whose named variables carry `value` as their fill.

        Raises:
            TypeError: `target` is a string instead of a list of names.
            KeyError: `target` names something that is not a data variable.
            ValueError: A named variable's dtype cannot hold `value`.

        Examples:
            >>> ds.gs.write_nodata(0).red.attrs["_FillValue"]
            0
            >>> ds.gs.write_nodata(-9999, target=["elevation"])
        """
        written = self._data.copy()
        for name in self._target_variables(target):
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
            target: List of variable names; None writes every data variable,
                and an empty list writes none.

        Returns:
            New Dataset whose named variables carry the flags.

        Raises:
            TypeError: `target` is a string instead of a list of names.
            KeyError: `target` names something that is not a data variable.
            ValueError: `colors` colours a value `classes` does not name.

        Examples:
            >>> labels.gs.write_flags({0: "water", 1: "trees"}, {0: "#419bdf", 1: "#397d49"})
        """
        written = self._data.copy()
        for name in self._target_variables(target):
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
            target: List of variable names; None writes every data variable,
                and an empty list writes none.

        Returns:
            New Dataset whose named variables carry the packing.

        Raises:
            TypeError: `target` is a string instead of a list of names.
            KeyError: `target` names something that is not a data variable.

        Examples:
            >>> packing = attrs.CFPacking(scale_factor=1e-4, add_offset=-0.1)
            >>> ds.gs.write_packing(packing, target=["B04", "B08"])
        """
        written = self._data.copy()
        for name in self._target_variables(target):
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
            target: List of variable names; None writes every data variable,
                and an empty list writes none.

        Returns:
            New Dataset whose named variables carry the description.

        Raises:
            TypeError: `target` is a string instead of a list of names.
            KeyError: `target` names something that is not a data variable.

        Examples:
            >>> ds.gs.write_spectral("red", center_wavelength=0.665, target=["B04"])
        """
        written = self._data.copy()
        for name in self._target_variables(target):
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
        self._data[channels]  # a missing variable raises KeyError
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

    def expand(
        self, item_dim: str = TIME_COORDINATE, id_format: str = ITEM_ID_FORMAT
    ) -> DataTree:
        """Expand a collection's Dataset into a STAC tree: collection, an Item per step, an asset per variable.

            /                          catalog
            └── /sentinel-2-l2a        collection   StacCollection attrs
                ├── /20250601T..._33UUU_...  Item   the Dataset's other attrs; coords along item_dim
                │   ├── /red           asset        one variable with its attrs and its grid
                │   └── /nir           asset

        The collection is named by its `StacCollection` id, else ``"default"``.
        An Item is named by its id, formatted from its region: its period and
        place, as `GeoRegion.format` reads them. The id is the Item's name in
        every catalog the tree is published to. A variable without `item_dim`
        belongs to no Item and is left out.

        Args:
            item_dim: Dimension whose steps become Items.
            id_format: Item id template over `GeoRegion.format` fields, such as
                ``"s2_{start:%Y%m%d}_{mgrs}"``.

        Returns:
            Lazy tree of ``{collection}/{item id}/{variable}`` leaves.

        Raises:
            KeyError: The Dataset has no `item_dim`.
            ValueError: An Item lacks a field the format names, or two Items
                format to one id.

        Examples:
            >>> [leaf.path for leaf in scene.gs.expand().leaves][:2]
            ['/sentinel-2-l2a/20250601T103000_33UUU_52.5200_13.4050/red', ...]
        """
        raster = self._data
        count = raster.sizes[item_dim]

        # A variable without the Item dim belongs to no Item, so no asset holds it.
        raster = raster.drop_vars(list(raster.drop_dims(item_dim).data_vars))

        # Collection: the STAC Collection alone, named by its id; the grid
        # belongs to each asset.
        collection = xr.Dataset()
        collection_id = DEFAULT_COLLECTION
        stated_collection = attrs.StacCollection.from_attrs(raster.attrs)
        if stated_collection is not None:
            collection = attrs.stamp_attrs(collection, stated_collection)
            if stated_collection.collection is not None:
                collection_id = stated_collection.collection["id"]
        nodes: dict[str, xr.Dataset] = {"/": xr.Dataset(), collection_id: collection}

        # The other Dataset attrs describe every Item alike.
        item_attrs = attrs.clear_attrs(raster, attrs.StacCollection).attrs

        # Coordinates along the Item dim describe one Item each.
        item_coords = []
        for name, coord in raster.coords.items():
            if item_dim in coord.dims:
                item_coords.append(name)

        for index in range(count):
            observation = raster.isel({item_dim: index})
            item = observation.gs.region.format(id_format)
            # from_dict would overwrite a repeated node.
            if f"{collection_id}/{item}" in nodes:
                raise ValueError(
                    f"two Items format to the id {item!r}; give expand an "
                    "id_format that tells them apart"
                )

            # Item: its coordinates, now scalar, and the Dataset's other attrs.
            coords = {name: observation[name].variable for name in item_coords}
            nodes[f"{collection_id}/{item}"] = xr.Dataset(
                coords=coords, attrs=item_attrs
            )

            # Asset: one variable with its own attrs and its grid; it inherits
            # the Item's coordinates.
            for name in observation.data_vars:
                nodes[f"{collection_id}/{item}/{name}"] = (
                    observation[[name]].drop_vars(item_coords).drop_attrs(deep=False)
                )

        return cast("DataTree", xr.DataTree.from_dict(nodes))

    def to_array(self, dim: str = BAND_DIMENSION) -> DataArray:
        """Stack variables lazily along `band`, in their existing order.

        Native `Dataset.to_array` stacks, broadcasting dims and promoting
        dtypes as xarray does. Stacked pixels share one meaning, so every
        variable must state the same `CFPacking`, `CFFlags` and units. As
        natively, the stack keeps the Dataset attrs and drops variable attrs,
        except packing and flags, which the pixels need to keep their meaning.

        Args:
            dim: Name of the stacking dimension, as in native `to_array`.

        Returns:
            Array shaped `(*axes, dim, y, x)`. No packing is applied.

        Raises:
            ValueError: The raster already spans `dim`.
            RasterValidationError: A variable has no georeferenced grid, or two
                variables state different pixel meaning.

        Examples:
            >>> image = ds[["B04", "B08"]].gs.to_array().astype("float32")
            >>> image.dims
            ('time', 'band', 'y', 'x')
        """
        # Native to_array only warns, then builds an array with `dim` twice.
        if dim in self._data.dims:
            raise ValueError(
                f"this raster already spans {dim!r}, which stacking would name; "
                "pass another dim or rename it first"
            )

        self.validate(same_pixel_meaning)

        spatial = cast("tuple[str, str]", self.spatial_dims)
        array = self._data.to_array(dim=dim).transpose(..., dim, *spatial)

        # Native stacking drops variable attrs. Packing and flags are checked equal
        # and kept, since without them a stacked pixel would mean another number.
        first = self._data[self.variables[0]]
        if first.gs.packing is not None:
            array = attrs.stamp_attrs(array, first.gs.packing)
        if first.gs.flags is not None:
            array = attrs.stamp_attrs(array, first.gs.flags)
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
        array = self.to_array()
        if dtype is not None:
            array = array.astype(dtype)
        return array.values

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
        """STAC Item publication is being rebuilt as `ds.gs.expand().gs.to_catalog(path)`.

        Raises:
            NotImplementedError: Always, until publication is rebuilt.
        """
        raise NotImplementedError(
            "STAC publication is being rebuilt as ds.gs.expand().gs.to_catalog(path)"
        )

    def to_catalog(self, path: Location, **options: Any) -> GeoDataFrame:
        """STAC catalog publication is being rebuilt as `ds.gs.expand().gs.to_catalog(path)`.

        Raises:
            NotImplementedError: Always, until publication is rebuilt.
        """
        raise NotImplementedError(
            "STAC publication is being rebuilt as ds.gs.expand().gs.to_catalog(path)"
        )

    def to_cog(self, path: Location, **options: Any) -> list[UPath]:
        """A Dataset is not COG-shaped: a COG holds one instant of one grid.

        Publish ``ds.gs.expand().gs.to_catalog(path)`` for one file per asset,
        or ``array.gs.to_cog(path)`` for one array.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "a Dataset is not COG-shaped; publish ds.gs.expand().gs.to_catalog(path) "
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
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> UPath | Delayed:
        """Write this raster as one netCDF file, every date and variable in it.

        Args:
            path: The file, ending in .nc, .nc4 or .cdf.
            compute: False returns one native `dask.Delayed`; computing it
                writes the file and returns its path.
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
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )
