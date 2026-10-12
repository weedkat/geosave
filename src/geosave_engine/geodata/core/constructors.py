"""Build rasters and arrays, optionally on a reference's coordinates.

A reference says where the pixels sit. A GeoBox supplies the grid; an xarray
object also passes on its coordinates, as-is, on the dimensions the new pixels
use. Native xarray construction checks ranks and lengths, and an exact join
refuses pixels that do not line up with the reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from geosave_engine.geodata import DataArray, Dataset

# Native xarray variables declare all dimensions independently of coordinates.
type RasterVariable = xr.DataArray | tuple[Sequence[str], Any]
type Reference = GeoBox | GCPGeoBox | xr.DataArray | xr.Dataset


def raster(
    variables: Mapping[str, RasterVariable],
    reference: Reference | None = None,
    *,
    nodata: float | int | None = None,
    **coords: Any,
) -> Dataset:
    """Build named raster variables, optionally on a reference's coordinates.

    Args:
        variables: Names mapped to DataArrays or `(dims, pixels)` tuples.
            Dimensions must end with `y` and `x`.
        reference: Where the pixels sit. A GeoBox or GCPGeoBox supplies the
            grid. An xarray object also passes on, as-is, its coordinates on
            every dimension the variables use, such as `time`, and each of its
            bands' attrs, such as a legend, to the variable of the same name.
            A variable's own attrs win. None leaves placement to the
            variables.
        nodata: Fill value written on every variable. None leaves it unchanged.
        **coords: Coordinates for dimensions the reference lacks, such as
            class labels, or replacing one it has, such as `time`.

    Returns:
        Native Dataset with CF/GDAL grid metadata when a reference is given.
        Construction does not compute lazy pixels.

    Raises:
        TypeError: A variable lacks dimensions or the reference is unsupported.
        ValueError: Variables are empty, their trailing dimensions are not
            `y, x`, an xarray reference has no locatable grid, or the
            variables do not line up with the reference's coordinates.

    Examples:
        >>> scene = raster(
        ...     {"red": (("time", "y", "x"), cube),
        ...      "dem": (("y", "x"), elevation)},
        ...     geobox,
        ...     time=dates,
        ... )
        >>> predicted = raster({"prediction": (("time", "y", "x"), pixels)}, scene)
        >>> predicted.time.equals(scene.time)
        True
        >>> predicted = raster({"label": (("y", "x"), classes)}, labels)
        >>> predicted.label.gs.legend == labels.label.gs.legend
        True
    """
    if not variables:
        raise ValueError("a raster needs at least one named array")
    for name, variable in variables.items():
        if not isinstance(variable, (xr.DataArray, tuple)):
            raise TypeError(
                f"{name!r} needs explicit dimensions: use (dims, pixels) or a DataArray"
            )

    dataset = xr.Dataset(variables, coords=coords)
    for name, variable in dataset.data_vars.items():
        if variable.dims[-2:] != SPATIAL_DIMENSIONS:
            raise ValueError(
                f"{name!r}'s trailing dimensions must be {SPATIAL_DIMENSIONS}"
            )

    if reference is not None:
        dataset = _assign_reference_coords(dataset, reference, replaced=coords)
    if isinstance(reference, (xr.Dataset, xr.DataArray)):
        dataset = _pass_on_band_attrs(dataset, reference)
    if nodata is not None:
        dataset = dataset.gs.write_nodata(nodata)
    return cast("Dataset", dataset)


def array(
    pixels: Any,
    reference: Reference | None = None,
    *,
    dims: Sequence[str],
    nodata: float | int | None = None,
    **coords: Any,
) -> DataArray:
    """Build an array with explicit dimensions, optionally on a reference's coordinates.

    Args:
        pixels: NumPy or lazy array data.
        reference: Where the pixels sit, as `raster` takes it: a GeoBox, or an
            xarray object whose coordinates on `dims` are passed on as-is.
        dims: All dimensions in pixel-axis order, ending with `y` and `x`.
        nodata: Stored value marking absent pixels. None writes no fill value.
        **coords: Coordinates for dimensions the reference lacks, or replacing
            one it has.

    Returns:
        Unnamed native DataArray preserving pixel dtype and laziness.

    Raises:
        TypeError: The reference has an unsupported type.
        ValueError: Dimensions are invalid or the pixels do not line up with
            the reference's coordinates.

    Examples:
        >>> probabilities = array(
        ...     logits, scene, dims=("class", "y", "x"),
        ...     **{"class": ["water", "urban", "crop"]},
        ... )
        >>> probabilities.dims
        ('class', 'y', 'x')
    """
    dataset = raster({"pixels": (dims, pixels)}, reference, nodata=nodata, **coords)
    return cast("DataArray", dataset["pixels"].rename(None))


def write_cf_axes[T: xr.Dataset](raster: T, geobox: GeoBox) -> T:
    """Write onto the spatial coordinates odc's attrs and what CF says they measure.

    Raises:
        ValueError: The grid carries no CRS.
    """
    crs = geobox.crs
    if crs is None:
        raise ValueError(
            "geobox carries no CRS, so its axes measure pixels rather than "
            "ground position; assign one before describing them"
        )
    names = (
        ("latitude", "longitude")
        if crs.geographic
        else ("projection_y_coordinate", "projection_x_coordinate")
    )
    from odc.geo.xr import xr_coords

    native = xr_coords(geobox, always_yx=True)
    described = {}
    for dim, name, axis in zip(SPATIAL_DIMENSIONS, names, ("Y", "X"), strict=True):
        coord = raster[dim].copy()
        coord.attrs = {**native[dim].attrs, "standard_name": name, "axis": axis}
        described[dim] = coord
    return raster.assign_coords(described)


def _pass_on_band_attrs(
    dataset: xr.Dataset, reference: xr.Dataset | xr.DataArray
) -> xr.Dataset:
    """Give each variable the attrs of the reference's band of the same name."""
    bands = (
        dict(reference.data_vars)
        if isinstance(reference, xr.Dataset)
        else {reference.name: reference}
    )
    return dataset.assign(
        {
            name: variable.assign_attrs({**bands[name].attrs, **variable.attrs})
            for name, variable in dataset.data_vars.items()
            if name in bands
        }
    )


def _assign_reference_coords(
    dataset: xr.Dataset, reference: Reference, *, replaced: Iterable[Hashable]
) -> xr.Dataset:
    """Assign a reference's CRS and its coordinates on the dataset's dimensions.

    Args:
        dataset: Variables to place.
        reference: Grid, or xarray object passing on its coordinates.
        replaced: Coordinates the caller supplied, which the reference's do
            not override, except the grid's own `y` and `x`.

    Raises:
        TypeError: The reference is not a grid or an xarray object.
        ValueError: The dataset does not line up with the reference's
            coordinates, or an xarray reference has no locatable grid.
    """
    geobox, reference_coords = _coords_from_reference(reference)
    # The reference's CRS replaces any the variables carried in.
    dataset = dataset.drop_vars(
        [name for name, coord in dataset.coords.items() if _is_grid_mapping(coord)]
    )
    replaced = set(replaced) - set(SPATIAL_DIMENSIONS)
    passed_on = xr.Dataset(
        coords={
            name: coord
            for name, coord in reference_coords.items()
            if name not in replaced and set(coord.dims) <= set(dataset.dims)
        }
    )
    # Exact join: pixels off the reference's coordinates raise, not realign.
    dataset = xr.merge(
        [dataset, passed_on],
        join="exact",
        compat="no_conflicts",
        combine_attrs="override",
    )
    # odc points each variable at the CRS; the grid's own CRS coordinate goes
    # back on so attrs odc does not write, such as GCPs, survive.
    dataset = dataset.odc.assign_crs(geobox.crs).assign_coords(
        spatial_ref=passed_on.spatial_ref
    )
    # A GCP grid's axes count pixels, so they measure no CF coordinate.
    if isinstance(geobox, GeoBox):
        return write_cf_axes(dataset, geobox)
    return dataset


def _coords_from_reference(
    reference: Reference,
) -> tuple[GeoBox | GCPGeoBox, dict[Hashable, xr.Variable]]:
    """Return a reference's grid and the coordinates it passes on.

    Raises:
        TypeError: The reference is not a grid or an xarray object.
        ValueError: An xarray reference has no locatable grid.
    """
    from odc.geo.xr import xr_coords

    if isinstance(reference, (xr.DataArray, xr.Dataset)):
        geobox = reference.gs.geobox
        if geobox is None:
            raise ValueError("reference has no locatable grid")
        native = {
            name: coord.variable
            for name, coord in reference.coords.items()
            if not _is_grid_mapping(coord)
        }
    elif isinstance(reference, (GeoBox, GCPGeoBox)):
        geobox, native = reference, {}
    else:
        raise TypeError("reference must be a GeoBox, GCPGeoBox, DataArray or Dataset")

    grid = {
        name: coord.variable
        for name, coord in xr_coords(geobox, always_yx=True).items()
    }
    # Native labels win over the grid's: sliced values must stay exact to align.
    return geobox, grid | native


def _is_grid_mapping(coord: xr.DataArray) -> bool:
    """Return whether a coordinate is a CF grid mapping, the CRS's carrier."""
    carriers = {"crs_wkt", "spatial_ref", "grid_mapping_name"}
    return not coord.dims and not carriers.isdisjoint(coord.attrs)
