from __future__ import annotations


import numpy as np
import pytest
import xarray as xr
from affine import Affine
from dask.callbacks import Callback
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from tests.conftest import build_dataset, build_tree
from tests.conftest import build_dataset as build_raster
from geosave.geodata.core.raster.transform.warp import Resampling


UTM = "EPSG:32633"
WGS84 = "EPSG:4326"


def utm_box(size: int = 256, resolution: int = 10) -> GeoBox:
    """Build a projected grid of `size` square pixels."""
    span = size * resolution
    return GeoBox.from_bbox(
        (300000, 5000000, 300000 + span, 5000000 + span), crs=UTM, resolution=resolution
    )


def scene() -> xr.Dataset:
    """Build a two-band uint16 raster on the projected grid."""
    box = utm_box()
    return build_raster(
        {
            "red": (("y", "x"), np.ones(box.shape, "uint16")),
            "nir": (("y", "x"), np.ones(box.shape, "uint16")),
        },
        box,
        nodata=0,
    )


def landcover() -> xr.Dataset:
    """Build a raster whose values are class codes."""
    box = utm_box()
    coded = build_raster({"landcover": (("y", "x"), np.ones(box.shape, "uint8"))}, box)
    return coded.gs.write_legend({1: "water", 2: "urban"}, target="landcover")


def test_a_warped_raster_reports_the_crs_it_landed_on() -> None:
    source = build_raster(
        {"elevation": (("y", "x"), np.ones(utm_box().shape, "float32"))},
        utm_box(),
    )

    warped = source.gs.reproject(WGS84)

    assert warped.gs.crs == "EPSG:4326"
    # odc's own Dataset path leaves the source CRS in spatial_ref's attrs.
    assert source.odc.reproject(WGS84).odc.crs.epsg == 32633


@pytest.mark.parametrize("resampling", ["bilinear", "cubic", "average"])
def test_a_blending_kernel_refuses_class_codes(resampling: Resampling) -> None:
    with pytest.raises(ValueError, match="class codes"):
        landcover().gs.reproject(WGS84, resampling=resampling)


def test_a_palette_band_refuses_a_blending_kernel_without_a_legend() -> None:
    raster = build_raster(
        {"cover": (("y", "x"), np.ones(utm_box().shape, "uint8"))}, utm_box()
    )
    painted = raster.assign(
        cover=attrs.stamp_attrs(raster.cover, attrs.GDALBand(colorinterp="palette"))
    )

    with pytest.raises(ValueError, match="class codes"):
        painted.gs.reproject(WGS84, resampling="bilinear")


@pytest.mark.parametrize("resampling", ["nearest", "mode"])
def test_a_value_preserving_kernel_accepts_class_codes(
    resampling: Resampling,
) -> None:
    from geosave.geodata.warnings import AssumedFillWarning

    # The class map carries no fill value, so uncovered pixels take uint8's.
    with pytest.warns(AssumedFillWarning, match="255"):
        warped = landcover().gs.reproject(WGS84, resampling=resampling)

    assert set(np.unique(warped.landcover.values)) <= {0, 1, 2, 255}


def _small(dtype: str, nodata: float | int | None) -> xr.Dataset:
    box = utm_box(size=8)
    return build_raster(
        {"red": (("y", "x"), np.full(box.shape, 5, dtype))}, box, nodata=nodata
    )


def _wider() -> GeoBox:
    """Build a grid reaching past `_small`'s, so a warp leaves uncovered pixels."""
    return GeoBox.from_bbox((300000, 5000000, 300160, 5000080), crs=UTM, resolution=10)


@pytest.mark.parametrize(
    ("dtype", "fill"), [("uint16", 65535), ("uint8", 255), ("int16", -32768)]
)
def test_reproject_gives_an_integer_with_no_fill_value_one_and_warns(
    dtype: str, fill: int
) -> None:
    from geosave.geodata.warnings import AssumedFillWarning

    source = _small(dtype, None)

    with pytest.warns(AssumedFillWarning, match=str(fill)):
        warped = source.gs.reproject(_wider())
    with pytest.warns(AssumedFillWarning, match=str(fill)):
        band = source.red.gs.reproject(_wider())

    assert warped.red.gs.nodata == fill
    assert sorted(np.unique(warped.red.values).tolist()) == sorted([5, fill])
    assert sorted(np.unique(band.values).tolist()) == sorted([5, fill])
    assert warped.red.dtype == np.dtype(dtype)


GRID = GeoBox((4, 4), Affine(10, 0, 300000, 0, -10, 5000000), "EPSG:32633")


def image(grid=GRID, *, dtype="uint16", fill: float = 0):
    pixels = np.arange(1, grid.shape.y * grid.shape.x + 1, dtype=dtype).reshape(
        grid.shape.yx
    )
    return build_dataset({"red": (("y", "x"), pixels)}, grid, nodata=fill)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("same_grid", [False, True])
def test_reproject_validates_bands_even_on_the_same_grid(empty, same_grid):
    source = image()
    if empty:
        source = source.drop_vars("red")
    else:
        source["profile"] = ("depth", [1, 2])
    target = GRID if same_grid else GRID.zoom_out(2)
    with pytest.raises(
        ValueError, match="no raster variables|profile.*spatial dimensions"
    ):
        source.gs.reproject(target)


@pytest.mark.parametrize("kind", ["array", "dataset"])
@pytest.mark.parametrize("lazy", [False, True])
def test_nonadjacent_spatial_dimensions_reproject_without_losing_axis_order(kind, lazy):
    source = image().expand_dims(time=[1, 2]).transpose("y", "time", "x")
    data = source.red if kind == "array" else source
    if lazy:
        data = data.chunk()
    with Callback(pretask=lambda *_: pytest.fail("reproject computed lazy pixels")):
        result = data.gs.reproject(GRID.zoom_out(2))
    band = result if kind == "array" else result.red
    assert band.dims == ("y", "time", "x")
    np.testing.assert_array_equal(band.isel(time=0).values, [[6, 8], [14, 16]])


@pytest.mark.parametrize(
    "kernel", ["nearest", "bilinear", "cubic", "average", "mode", "sum"]
)
@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857"])
@pytest.mark.parametrize("lazy", [False, True])
def test_cross_crs_pixels_match_explicit_rasterio_warp(kernel, crs, lazy, request):
    from rasterio.enums import Resampling
    from rasterio.warp import reproject as rio_reproject

    source = image(dtype="float32", fill=-999.0)
    source.red.values[1, 1] = -999.0
    target = GRID.to_crs(crs)
    expected = np.full(target.shape.yx, -999.0, dtype="float32")
    rio_reproject(
        source.red.values,
        expected,
        src_transform=GRID.transform,
        src_crs=str(GRID.crs),
        src_nodata=-999.0,
        dst_transform=target.transform,
        dst_crs=str(target.crs),
        dst_nodata=-999.0,
        resampling=Resampling[kernel],
    )
    if lazy:
        source = source.chunk({"x": 2, "y": 2})
    with Callback(pretask=lambda *_: pytest.fail("reproject computed lazy pixels")):
        result = source.gs.reproject(target, resampling=kernel)
    assert result.gs.geobox == target
    actual = result.red.compute().values
    # Mark only after grid, laziness and computation have succeeded, so a
    # known numerical discrepancy cannot hide a different regression.
    if lazy and crs == "EPSG:3857" and kernel in ("bilinear", "cubic", "average"):
        reason = (
            "ODC omits interpolation neighbours at chunk boundaries"
            if kernel in ("bilinear", "cubic")
            else "GDAL chunk-local averaging differs from the full-grid reference"
        )
        request.node.add_marker(
            pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_tree_to_dataset_warps_each_variable_with_its_named_kernel_lazily():
    labels = image().gs.write_legend({1: "land", 2: "water"}, target="red")
    labels.red.values[:] = [[1, 1, 2, 2]] * 4
    height = image(dtype="float32").rename(red="height")
    source = build_tree({"labels": labels.chunk(), "dem": height.chunk()})
    target = GRID.zoom_out(2)
    with Callback(pretask=lambda *_: pytest.fail("to_dataset computed lazy pixels")):
        result = source.gs.to_dataset(
            grid=target, resampling={"*": "average", "red": "mode"}
        )
    np.testing.assert_array_equal(result.red.values, [[1, 2], [1, 2]])
    np.testing.assert_array_equal(result.height.values, [[3.5, 5.5], [11.5, 13.5]])


def test_selecting_a_tree_node_supplies_an_exact_destination():
    reference = build_tree({"r10": image(), "r20": image(GRID.zoom_out(2))})
    assert reference.gs.geobox is None
    result = image(dtype="float32").gs.reproject(reference["r10"].gs.geobox)
    assert result.gs.geobox == GRID
