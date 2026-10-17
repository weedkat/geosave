"""Converting explicitly between geolocated rasters and vector features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata.conventions import SPATIAL_DIMENSIONS, TIME_COORDINATE

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from odc.geo import CRS

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core.anchor import GeoAnchor


def vectorize(
    flags: xr.DataArray,
    *,
    value_name: str = "value",
    mask: xr.DataArray | np.ndarray | None = None,
    connectivity: Literal[4, 8] = 4,
) -> GeoDataFrame:
    """Polygonize contiguous values from one geolocated flag plane.

    This operation computes lazy flags because geometry depends on their
    values.

    Args:
        flags: Two-dimensional categorical or flag array.
        value_name: Property column receiving each region's value.
        mask: Optional exact-grid mask selecting additional valid pixels.
        connectivity: Four- or eight-neighbour region connectivity.

    Returns:
        One row per contiguous flag region, in the raster CRS.

    Raises:
        ValueError: The array, mask, name, connectivity, or dtype is
            unsuitable for polygonization.

    Examples:
        >>> regions = vectorize(prediction["cover"], value_name="cover")
        >>> regions["cover"].tolist()
        [1, 2]
    """
    geobox = _read_plane_grid(flags)
    if value_name == "geometry":
        raise ValueError("value_name must not replace the geometry column")
    values = _narrow_dtype(np.asarray(flags.compute().values))
    valid = _read_valid_pixels(values, flags.gs.nodata, mask, geobox)
    return _polygonize(values, valid, geobox, value_name, connectivity)


def rasterize(
    vector: gpd.GeoDataFrame,
    like: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree,
    *,
    column: str | None = None,
    fill: int | float | bool = 0,
    dtype: DTypeLike | None = None,
    all_touched: bool = False,
) -> xr.DataArray:
    """Burn vector geometries or one numeric property onto an exact grid.

    Args:
        vector: Features to burn.
        like: Anchor or geolocated xarray object supplying the target grid.
        column: Property to burn. None returns a boolean presence mask.
        fill: Value outside geometries. A presence mask takes only 0 or False.
        dtype: Output dtype for property values.
        all_touched: Burn every touched pixel instead of pixel centres.

    Returns:
        Eager geolocated array on the exact target grid. A dated anchor
        dates it with a scalar `time` at the start of its timespan.

    Raises:
        KeyError: The selected property is absent.
        ValueError: A geometry is null, empty, or invalid, or values or fill
            cannot be represented by the dtype, or a presence mask is given
            a truthy fill.

    Examples:
        >>> rasterize(fields, scene).dtype
        dtype('bool')
        >>> rasterize(fields, scene, column="crop", dtype="uint8").name
        'crop'
    """
    from geosave.geodata.core.raster.create import array

    geobox = _read_burn_grid(like)
    frame = _validate_geometries(vector).to_crs(cast("CRS", geobox.crs))
    if column is None:
        pixels = _burn_presence(frame, geobox, fill, all_touched)
    else:
        pixels = _burn_column(frame, geobox, column, fill, dtype, all_touched)
    raster = array(pixels, geobox, dims=SPATIAL_DIMENSIONS).rename(column or "mask")
    return _assign_target_coords(raster, like)


# Rasterio polygonizes only these dtypes.
_POLYGONIZABLE = frozenset(
    np.dtype(name) for name in ("uint8", "uint16", "int16", "int32", "float32")
)


def _read_plane_grid(flags: xr.DataArray) -> GeoBox:
    """Read the grid of one two-dimensional spatial plane."""
    geobox = flags.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError("flags carry no locatable grid")
    if flags.ndim != 2 or tuple(flags.dims) != SPATIAL_DIMENSIONS:
        raise ValueError(
            "vectorize needs one two-dimensional spatial plane on "
            f"{SPATIAL_DIMENSIONS}; select extra dimensions from {flags.dims} first"
        )
    return geobox


def _narrow_dtype(values: np.ndarray) -> np.ndarray:
    """Narrow values onto a dtype rasterio polygonizes."""
    if values.dtype == np.bool_:
        return values.astype("uint8")
    if values.dtype.kind in "iu" and values.dtype not in _POLYGONIZABLE:
        # An argmax emits int64, which narrows to int32 when its values fit.
        bounds = np.iinfo("int32")
        if values.size and (values.min() < bounds.min or values.max() > bounds.max):
            raise ValueError(
                f"flags are {values.dtype} and hold values beyond int32, the "
                f"widest integer raster polygonization supports"
            )
        return values.astype("int32")
    if values.dtype not in _POLYGONIZABLE:
        raise ValueError(f"raster polygonization does not support dtype {values.dtype}")
    return values


def _read_valid_pixels(
    values: np.ndarray,
    fill: float | int | None,
    mask: xr.DataArray | np.ndarray | None,
    geobox: GeoBox,
) -> np.ndarray:
    """Read which pixels hold a finite value other than nodata, narrowed by `mask`."""
    valid = np.isfinite(values)
    if fill is not None:
        valid &= values != fill
    if mask is None:
        return valid

    if isinstance(mask, xr.DataArray):
        if mask.gs.geobox != geobox:
            raise ValueError("mask must be on the exact flags grid")
        mask_values = np.asarray(mask.compute().values)
    else:
        mask_values = np.asarray(mask)
    if mask_values.shape != values.shape:
        raise ValueError(
            f"mask shape {mask_values.shape} does not match flags {values.shape}"
        )
    return valid & mask_values.astype(bool)


def _polygonize(
    values: np.ndarray,
    valid: np.ndarray,
    geobox: GeoBox,
    value_name: str,
    connectivity: Literal[4, 8],
) -> GeoDataFrame:
    """Trace one polygon per contiguous region of equal value."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    features = list(
        shapes(
            values, mask=valid, transform=geobox.transform, connectivity=connectivity
        )
    )
    geometries = [shape(geometry) for geometry, _ in features]

    # Rasterio's values are numpy scalars, read out as plain Python numbers.
    if features:
        scalar = values.dtype.type
        region_values = [scalar(value).item() for _, value in features]
    else:
        # An empty column still says what type its values would hold.
        region_values = pd.Series(dtype=values.dtype)

    # Regions touching at a corner trace one self-touching ring.
    regions = gpd.GeoDataFrame(
        {value_name: region_values},
        geometry=gpd.GeoSeries(geometries, crs=geobox.crs).make_valid(),
    )
    return cast("GeoDataFrame", regions)


def _read_burn_grid(
    like: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree,
) -> GeoBox:
    """Read the exact grid geometries burn onto."""
    from geosave.geodata.core.anchor import GeoAnchor

    geobox = like.geobox if isinstance(like, GeoAnchor) else like.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError("rasterize target carries no regular grid CRS")
    return geobox


def _validate_geometries(vector: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Refuse a geometry naming no ground, which would burn wrong pixels silently."""
    geometries = vector.geometry
    if (
        geometries.isna().any()
        or geometries.is_empty.any()
        or not geometries.is_valid.all()
    ):
        raise ValueError(
            "the vector holds a null, empty, or invalid geometry; drop those "
            "rows or repair them with geometry.make_valid() before rasterizing"
        )
    return vector


def _burn_presence(
    frame: gpd.GeoDataFrame, geobox: GeoBox, fill: int | float | bool, all_touched: bool
) -> np.ndarray:
    """Burn True inside the geometries and False outside."""
    # A presence mask is True inside, so a truthy fill would mark every pixel.
    if fill:
        raise ValueError(
            f"a presence mask is True inside the geometries, so fill={fill!r} "
            f"would mark every pixel; pass column to burn values, or drop fill"
        )
    ones = np.ones(len(frame), dtype="uint8")
    return _burn(frame, ones, geobox, 0, all_touched).view(bool)


def _burn_column(
    frame: gpd.GeoDataFrame,
    geobox: GeoBox,
    column: str,
    fill: int | float | bool,
    dtype: DTypeLike | None,
    all_touched: bool,
) -> np.ndarray:
    """Burn one numeric property, refusing values its dtype stores as other numbers."""
    if column not in frame:
        raise KeyError(f"the vector has no {column!r} column")
    series = cast("pd.Series", frame[column])
    if series.isna().any():
        raise ValueError("rasterize values must not be null")

    column_values = np.asarray(series.tolist())
    output_dtype = _resolve_dtype(column_values, fill, dtype)
    values = _cast_losslessly(column_values, output_dtype, column)
    fill_value = _cast_losslessly(np.asarray([fill]), output_dtype, column)[0].item()
    return _burn(frame, values, geobox, fill_value, all_touched)


def _resolve_dtype(
    values: np.ndarray, fill: int | float | bool, dtype: DTypeLike | None
) -> np.dtype:
    """Read the numeric dtype burned values are stored in."""
    try:
        if dtype is None:
            output_dtype = np.dtype(np.result_type(values, type(fill)))
        else:
            output_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("rasterize needs a numeric raster dtype") from error
    if output_dtype.kind not in "iuf":
        raise ValueError(f"rasterize needs a numeric raster dtype, got {output_dtype}")
    return output_dtype


def _cast_losslessly(values: np.ndarray, dtype: np.dtype, column: str) -> np.ndarray:
    """Cast values to `dtype`, refusing any it stores as another number."""
    if values.dtype.kind not in "biuf":
        raise ValueError("rasterize needs a numeric raster dtype")
    # astype wraps and rounds, so an unheld -1 would burn as 65535 on uint16.
    with np.errstate(invalid="ignore", over="ignore"):
        cast_values = values.astype(dtype)
    if not np.array_equal(cast_values, values, equal_nan=True):
        raise ValueError(
            f"{column!r} or the fill holds values {dtype} stores as "
            f"other numbers; pass a dtype that holds them"
        )
    return cast_values


def _burn(
    frame: gpd.GeoDataFrame,
    values: np.ndarray,
    geobox: GeoBox,
    fill: int | float,
    all_touched: bool,
) -> np.ndarray:
    """Burn each geometry's value onto the grid, later rows winning overlaps."""
    from rasterio.features import rasterize as burn

    if frame.empty:
        return np.full(geobox.shape.yx, fill, dtype=values.dtype)
    pixels = burn(
        zip(frame.geometry, values, strict=True),
        out_shape=geobox.shape.yx,
        transform=geobox.transform,
        fill=fill,  # type: ignore
        dtype=values.dtype,
        all_touched=all_touched,
    )
    if pixels is None:
        raise RuntimeError("Rasterio returned no rasterized array")
    return pixels


def _assign_target_coords(
    raster: xr.DataArray, like: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree
) -> xr.DataArray:
    """Take the target's own coordinates, or date the raster by its anchor."""
    from geosave.geodata.core.anchor import GeoAnchor

    if isinstance(like, GeoAnchor):
        if like.timespan is None:
            return raster
        # A raster's date is one instant, so a span is dated by where it starts.
        return raster.assign_coords({TIME_COORDINATE: np.datetime64(like.timespan[0])})

    # Sliced coordinates drift by float error, so take the target's own to align.
    target = like.dataset if isinstance(like, xr.DataTree) else like
    coords = {}
    for dim in SPATIAL_DIMENSIONS:
        coords[dim] = target.coords[dim]
    return raster.assign_coords(coords)
