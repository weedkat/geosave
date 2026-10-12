"""Converting explicitly between geolocated rasters and vector features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from geosave_engine.geodata import GeoDataFrame
    from geosave_engine.geodata.core.anchor import GeoAnchor


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
    from rasterio.features import shapes
    from shapely.geometry import shape

    # Polygonization needs one realized numeric plane on a regular grid.
    geobox = flags.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError("flags carry no locatable grid")
    if flags.ndim != 2 or tuple(flags.dims) != SPATIAL_DIMENSIONS:
        raise ValueError(
            "vectorize needs one two-dimensional spatial plane on "
            f"{SPATIAL_DIMENSIONS}; select extra dimensions from {flags.dims} first"
        )
    if value_name == "geometry":
        raise ValueError("value_name must not replace the geometry column")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    # Rasterio polygonizes only these, so other types are narrowed onto them.
    supported = {
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("int32"),
        np.dtype("float32"),
    }
    values = np.asarray(flags.compute().values)
    if values.dtype == np.bool_:
        values = values.astype("uint8")
    if values.dtype.kind in "iu" and values.dtype not in supported:
        # An argmax emits int64, which narrows to int32 when its values fit.
        bounds = np.iinfo("int32")
        if values.size and (values.min() < bounds.min or values.max() > bounds.max):
            raise ValueError(
                f"flags are {values.dtype} and hold values beyond int32, the "
                f"widest integer raster polygonization supports"
            )
        values = values.astype("int32")
    if values.dtype not in supported:
        raise ValueError(f"raster polygonization does not support dtype {values.dtype}")

    # A pixel is polygonized where it holds a finite value other than nodata.
    valid = np.isfinite(values)
    fill = flags.gs.nodata
    if fill is not None:
        valid &= values != fill

    # A caller's mask narrows the valid pixels further.
    if mask is not None:
        if isinstance(mask, xr.DataArray):
            if mask.gs.geobox != geobox:
                raise ValueError("mask must be on the exact flags grid")
            selected = np.asarray(mask.compute().values)
        else:
            selected = np.asarray(mask)
        if selected.shape != values.shape:
            raise ValueError(
                f"mask shape {selected.shape} does not match flags {values.shape}"
            )
        valid &= selected.astype(bool)

    # Rasterio emits one geometry for each contiguous region.
    features = list(
        shapes(
            values,
            mask=valid,
            transform=geobox.transform,
            connectivity=connectivity,
        )
    )
    geometries = [shape(geometry) for geometry, _ in features]

    # Rasterio's values are numpy scalars, read out as plain Python numbers.
    scalar = values.dtype.type
    if features:
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
        Eager geolocated array on the exact target grid.

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
    from rasterio.features import rasterize as burn

    from geosave_engine.geodata.core.anchor import GeoAnchor
    from geosave_engine.geodata.core.constructors import array

    # The target owns the exact output grid; only burn geometry is reprojected.
    geobox = like.geobox if isinstance(like, GeoAnchor) else like.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError("rasterize target carries no regular grid CRS")
    # A geometry naming no ground burns wrong pixels without a word.
    shapes = vector.geometry
    if shapes.isna().any() or shapes.is_empty.any() or not shapes.is_valid.all():
        raise ValueError(
            "the vector holds a null, empty, or invalid geometry; drop those "
            "rows or repair them with geometry.make_valid() before rasterizing"
        )
    frame = vector.to_crs(geobox.crs)

    # A presence mask burns ones; a property burns its own values, checked to fit.
    output_dtype: np.dtype
    if column is None:
        # A presence mask is True inside, so a truthy fill would mark every pixel.
        if fill:
            raise ValueError(
                f"a presence mask is True inside the geometries, so fill={fill!r} "
                f"would mark every pixel; pass column to burn values, or drop fill"
            )
        output_dtype = np.dtype("uint8")
        values = np.ones(len(frame), dtype=output_dtype)
        output_name = "mask"
        output_fill: int | float | bool = False
    else:
        if column not in frame:
            raise KeyError(f"the vector has no {column!r} column")
        series = cast("pd.Series", frame[column])
        if series.isna().any():
            raise ValueError("rasterize values must not be null")
        try:
            output_dtype = np.dtype(
                dtype
                if dtype is not None
                else np.result_type(np.asarray(series.tolist()), type(fill))
            )
        except (TypeError, ValueError) as error:
            raise ValueError("rasterize needs a numeric raster dtype") from error
        if output_dtype.kind not in "iuf":
            raise ValueError(
                f"rasterize needs a numeric raster dtype, got {output_dtype}"
            )

        given = np.asarray([*series.tolist(), fill])
        if given.dtype.kind not in "biuf":
            raise ValueError("rasterize needs a numeric raster dtype")
        # astype wraps and rounds, so an unheld -1 would burn as 65535 on uint16.
        with np.errstate(invalid="ignore", over="ignore"):
            stored = given.astype(output_dtype)
        if not np.array_equal(stored, given, equal_nan=True):
            raise ValueError(
                f"{column!r} or the fill holds values {output_dtype} stores as "
                f"other numbers; pass a dtype that holds them"
            )
        values = stored[:-1]
        output_name = column
        output_fill = cast(
            "int | float | bool", np.asarray(fill).astype(output_dtype).item()
        )

    # Rasterio preserves row order, so later rows win where geometries overlap.
    if frame.empty:
        pixels = np.full(geobox.shape.yx, output_fill, dtype=output_dtype)
    else:
        pixels = burn(
            zip(frame.geometry, values, strict=True),
            out_shape=geobox.shape.yx,
            transform=geobox.transform,
            fill=output_fill,  # type: ignore
            dtype=output_dtype,
            all_touched=all_touched,
        )
        if pixels is None:
            raise RuntimeError("Rasterio returned no rasterized array")
    if column is None:
        pixels = pixels.view(bool)

    burned = array(pixels, geobox, dims=SPATIAL_DIMENSIONS).rename(output_name)

    # Sliced coordinates drift by float error, so take the target's own to align.
    if not isinstance(like, GeoAnchor):
        target = like.dataset if isinstance(like, xr.DataTree) else like
        burned = burned.assign_coords(
            {dim: target.coords[dim] for dim in SPATIAL_DIMENSIONS}
        )
    return burned
