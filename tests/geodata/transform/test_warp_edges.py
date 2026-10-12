"""Pixel, metadata, and grid contracts at reprojection edge cases."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from dask.callbacks import Callback
from odc.geo.geobox import GeoBox

from geosave.geodata import raster, stack
from geosave.geodata.transform.warp import align, common_grid, reproject

from .conftest import grid_of

GRID = GeoBox((4, 4), Affine(10, 0, 300000, 0, -10, 5000000), "EPSG:32633")


def image(grid=GRID, *, dtype="uint16", fill: float = 0):
    pixels = np.arange(1, grid.shape.y * grid.shape.x + 1, dtype=dtype).reshape(
        grid.shape.yx
    )
    return raster({"red": (("y", "x"), pixels)}, grid, nodata=fill)


@pytest.mark.parametrize("kind", ["array", "dataset", "stack"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "dtype,fill",
    [("uint8", 0), ("int16", -999), ("float32", np.nan), ("float64", -999.0)],
)
def test_wider_grid_preserves_pixels_nodata_metadata_and_laziness(
    kind, lazy, dtype, fill
):
    source = image(dtype=dtype, fill=fill).expand_dims(
        time=np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    )
    source.attrs["title"] = "scene"
    source.red.attrs["units"] = "m"
    if lazy:
        source = source.chunk({"y": 2, "x": 2})
    data = source.red if kind == "array" else source
    if kind == "stack":
        data = stack({"image": source})
        data.attrs["title"] = "collection"
    target = GeoBox((6, 6), GRID.transform * Affine.translation(-1, -1), GRID.crs)

    with Callback(pretask=lambda *_: pytest.fail("reproject computed lazy pixels")):
        result = reproject(data, target)

    output = result.gs.rasters["image"] if isinstance(result, xr.DataTree) else result
    band = output if kind == "array" else output.red
    assert type(result) is type(data)
    assert band.gs.geobox == target
    assert band.dtype == np.dtype(dtype)
    assert (band.chunks is not None) is lazy
    assert band.attrs["units"] == "m"
    np.testing.assert_equal(band.attrs["_FillValue"], fill)
    xr.testing.assert_equal(band.time, source.time)
    expected = np.full((2, 6, 6), fill, dtype=dtype)
    expected[:, 1:5, 1:5] = np.arange(1, 17).reshape(4, 4)
    np.testing.assert_array_equal(band.compute().values, expected)
    if kind != "array":
        assert output.attrs["title"] == "scene"
    if kind == "stack":
        assert result.attrs["title"] == "collection"


@pytest.mark.parametrize(
    "extent,width", [("union", 6), ("intersection", 2), ("match", 4)]
)
@pytest.mark.parametrize("lazy", [False, True])
def test_align_overlapping_scenes_has_exact_pixels(extent, width, lazy):
    west = image()
    east = image(GRID.translate_pix(2, 0))
    if lazy:
        west, east = west.chunk(), east.chunk()
    with Callback(pretask=lambda *_: pytest.fail("align computed lazy pixels")):
        first, second = align([west, east], extent=extent)
    assert first.gs.geobox == second.gs.geobox
    assert first.sizes["x"] == width
    expected = {
        "union": ([1, 2, 3, 4, 0, 0], [0, 0, 1, 2, 3, 4]),
        "intersection": ([3, 4], [1, 2]),
        "match": ([1, 2, 3, 4], [0, 0, 1, 2]),
    }
    np.testing.assert_array_equal(first.red.isel(y=0).values, expected[extent][0])
    np.testing.assert_array_equal(second.red.isel(y=0).values, expected[extent][1])


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("same_grid", [False, True])
def test_reproject_validates_bands_even_on_the_same_grid(empty, same_grid):
    source = image()
    if empty:
        source = source.drop_vars("red")
    else:
        source["profile"] = ("depth", [1, 2])
    target = GRID if same_grid else GRID.zoom_out(2)
    with pytest.raises(ValueError, match="no data variable|profile.*spans"):
        reproject(source, target)


@pytest.mark.parametrize("resolution", [None, 10])
def test_stack_reproject_to_crs_covers_every_group(resolution):
    source = stack({"west": image(), "east": image(GRID.translate_pix(4, 0))})
    result = reproject(source, "EPSG:32633", resolution=resolution)
    assert grid_of(result).shape.yx == (4, 8)
    np.testing.assert_array_equal(
        result.gs.rasters["west"].red.values[:, :4], image().red
    )
    np.testing.assert_array_equal(
        result.gs.rasters["east"].red.values[:, 4:], image().red
    )


def test_native_resolution_refuses_comparing_different_crs_units():
    target = GeoBox(GRID.shape, GRID.transform, "EPSG:3857")
    with pytest.raises(ValueError, match="native.*CRS|CRS.*native"):
        reproject(image(), target, resolution="native")


@pytest.mark.parametrize(
    "extent,shape", [("union", (12, 12)), ("intersection", (6, 6)), ("match", (12, 12))]
)
def test_native_alignment_keeps_shared_extent_and_individual_resolutions(extent, shape):
    fine = GeoBox((7, 11), GRID.transform, GRID.crs)
    grids = [fine, fine.zoom_out(2), fine.zoom_out(3)]
    aligned = align([image(grid) for grid in grids], resolution="native", extent=extent)
    assert grid_of(aligned[0]).shape.yx == shape
    assert len({data.gs.bounds for data in aligned}) == 1
    for source, result in zip(grids, aligned, strict=True):
        assert grid_of(result).resolution == source.resolution


@pytest.mark.parametrize("kind", ["array", "dataset"])
@pytest.mark.parametrize("lazy", [False, True])
def test_nonadjacent_spatial_dimensions_reproject_without_losing_axis_order(kind, lazy):
    source = image().expand_dims(time=[1, 2]).transpose("y", "time", "x")
    data = source.red if kind == "array" else source
    if lazy:
        data = data.chunk()
    with Callback(pretask=lambda *_: pytest.fail("reproject computed lazy pixels")):
        result = reproject(data, GRID.zoom_out(2))
    band = result if kind == "array" else result.red
    assert band.dims == ("y", "time", "x")
    np.testing.assert_array_equal(band.isel(time=0).values, [[6, 8], [14, 16]])


@pytest.mark.parametrize("operation", [reproject, common_grid])
def test_empty_stack_has_a_clear_grid_error(operation):
    with pytest.raises(ValueError, match="grid|groups|no data variables"):
        if operation is reproject:
            operation(xr.DataTree(), GRID)
        else:
            operation([xr.DataTree()])


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
        result = reproject(source, target, resampling=kernel)
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


@pytest.mark.parametrize("kernel", ["gauss", "unknown"])
@pytest.mark.parametrize("same_grid", [False, True])
def test_unsupported_kernel_is_rejected_before_warping(kernel, same_grid):
    target = GRID if same_grid else GRID.zoom_out(2)
    with pytest.raises(ValueError, match="resampling"):
        reproject(image(), target, resampling=kernel)


def test_stack_variable_kernels_keep_class_codes_and_continuous_values_lazy():
    labels = image().gs.write_legend({1: "land", 2: "water"}, target="red")
    labels.red.values[:] = [[1, 1, 2, 2]] * 4
    source = stack({"labels": labels.chunk(), "dem": image(dtype="float32").chunk()})
    target = GRID.zoom_out(2)
    with Callback(pretask=lambda *_: pytest.fail("reproject computed lazy pixels")):
        result = reproject(
            source, target, resampling={"*": "average", "labels/red": "mode"}
        )
    np.testing.assert_array_equal(
        result.gs.rasters["labels"].red.values, [[1, 2], [1, 2]]
    )
    np.testing.assert_array_equal(
        result.gs.rasters["dem"].red.values, [[3.5, 5.5], [11.5, 13.5]]
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_all_nodata_and_single_pixel_sources(lazy):
    source = image(
        GeoBox((1, 1), GRID.transform, GRID.crs), dtype="float32", fill=np.nan
    )
    source.red.values[:] = np.nan
    if lazy:
        source = source.chunk()
    result = reproject(source, GRID)
    assert result.gs.geobox == GRID
    assert np.isnan(result.red.compute().values).all()


@pytest.mark.parametrize("factor", [2, 0.75])
def test_native_nesting_refuses_coarser_or_fractional_lattices(factor):
    target = GRID.zoom_out(factor)
    with pytest.raises(ValueError, match="nest|ratio|whole"):
        reproject(image(), target, resolution="native")


def test_reproject_refuses_a_target_grid_without_crs():
    target = GeoBox(GRID.shape, GRID.transform, None)
    with pytest.raises(ValueError, match="CRS"):
        reproject(image(), target)


def test_fractional_intersection_keeps_conservative_boundary_pixels():
    shifted = image(GRID.translate_pix(0.5, 0))
    grid = common_grid([image(), shifted], extent="intersection")
    # A half-pixel overlap is enclosed on the reference lattice, not clipped
    # into fractional cells or snapped inward to discard boundary coverage.
    assert grid == GRID


def test_native_stack_to_crs_preserves_resolutions_and_pixels():
    fine = GeoBox((6, 6), GRID.transform, GRID.crs)
    coarse = fine.zoom_out(3)
    source = stack({"fine": image(fine), "coarse": image(coarse)})
    result = reproject(source, "EPSG:32633", resolution="native")
    for name in ("fine", "coarse"):
        xr.testing.assert_identical(result.gs.rasters[name], source.gs.rasters[name])
