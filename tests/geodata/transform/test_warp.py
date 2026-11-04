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
from geosave.geodata.transform.warp import Resampling, align, reproject

from .conftest import grid_of

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

    warped = reproject(source, WGS84)

    assert warped.gs.crs == "EPSG:4326"
    # odc's own Dataset path leaves the source CRS in spatial_ref's attrs.
    assert source.odc.reproject(WGS84).odc.crs.epsg == 32633


@pytest.mark.parametrize("resampling", ["bilinear", "cubic", "average"])
def test_a_blending_kernel_refuses_class_codes(resampling: Resampling) -> None:
    with pytest.raises(ValueError, match="class codes"):
        reproject(landcover(), WGS84, resampling=resampling)


def test_a_palette_band_refuses_a_blending_kernel_without_a_legend() -> None:
    raster = build_raster(
        {"cover": (("y", "x"), np.ones(utm_box().shape, "uint8"))}, utm_box()
    )
    painted = raster.assign(
        cover=attrs.stamp_attrs(raster.cover, attrs.GDALBand(colorinterp="palette"))
    )

    with pytest.raises(ValueError, match="class codes"):
        reproject(painted, WGS84, resampling="bilinear")


@pytest.mark.parametrize("resampling", ["nearest", "mode"])
def test_a_value_preserving_kernel_accepts_class_codes(
    resampling: Resampling,
) -> None:
    from geosave.geodata.warnings import AssumedFillWarning

    # The class map carries no fill value, so uncovered pixels take uint8's.
    with pytest.warns(AssumedFillWarning, match="255"):
        warped = reproject(landcover(), WGS84, resampling=resampling)

    assert set(np.unique(warped.landcover.values)) <= {0, 1, 2, 255}


def test_matching_adopts_a_rasters_grid_without_naming_a_geobox() -> None:
    imagery = scene()
    dem_box = GeoBox.from_bbox((12.9, 45.0, 13.2, 45.2), crs=WGS84, resolution=0.0003)
    dem = build_raster(
        {"elevation": (("y", "x"), np.ones(dem_box.shape, "float32"))}, dem_box
    )

    assert reproject(dem, imagery).gs.geobox == imagery.gs.geobox


def test_warping_refuses_a_raster_carrying_no_grid() -> None:
    unreferenced = xr.Dataset({"band": (("y", "x"), np.zeros((4, 4), "uint8"))})

    with pytest.raises(ValueError, match="georegistered|locatable grid"):
        reproject(unreferenced, WGS84)


def offset_box(shift: float = 5.0, size: int = 256, resolution: int = 10) -> GeoBox:
    """Build a grid `shift` CRS units off the projected one's pixel edges."""
    span = size * resolution
    return GeoBox.from_bbox(
        (300000 + shift, 5000000, 300000 + shift + span, 5000000 + span),
        crs=UTM,
        resolution=resolution,
    )


def placed(box: GeoBox, fill: int = 7, name: str = "red") -> xr.Dataset:
    """Build a single-band raster holding `fill` on `box`."""
    return build_raster(
        {name: (("y", "x"), np.full(box.shape, fill, "uint16"))}, box, nodata=0
    )


def test_align_uses_the_requested_reference_grid() -> None:
    first, second = placed(utm_box()), placed(offset_box(), name="vv")

    on_first = align([first, second], target=first)
    on_second = align([first, second], target=second, extent="match")

    assert on_first[0].gs.geobox == first.gs.geobox
    assert on_second[0].gs.geobox == second.gs.geobox


def test_align_leaves_a_raster_already_on_the_grid_untouched() -> None:
    only = placed(utm_box(size=8))

    (aligned,) = align([only], target=only, extent="match")

    assert aligned.gs.geobox == only.gs.geobox
    assert np.array_equal(aligned.red.values, only.red.values)


def test_align_refuses_what_it_cannot_place() -> None:
    only = placed(utm_box(size=4))
    with pytest.raises(ValueError, match="pass at least one"):
        align([], target=only)
    with pytest.raises(ValueError, match="sits on no locatable grid"):
        align(
            [only, xr.Dataset({"red": (("y", "x"), np.zeros((4, 4), "uint16"))})],
            target=only,
        )
    with pytest.raises(ValueError, match="no locatable grid to land on"):
        align(
            [only], target=xr.Dataset({"red": (("y", "x"), np.zeros((4, 4), "uint16"))})
        )


def test_align_refuses_rasters_sharing_no_ground_to_intersect() -> None:
    here = placed(utm_box(size=4))
    far = placed(
        GeoBox.from_bbox((900000, 5000000, 900040, 5000040), crs=UTM, resolution=10),
        fill=9,
        name="vv",
    )

    with pytest.raises(ValueError, match="no ground in common"):
        align([here, far], target=here, extent="intersection")


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
        warped = reproject(source, _wider())
    with pytest.warns(AssumedFillWarning, match=str(fill)):
        band = reproject(source.red, _wider())

    assert warped.red.gs.nodata == fill
    assert sorted(np.unique(warped.red.values).tolist()) == sorted([5, fill])
    assert sorted(np.unique(band.values).tolist()) == sorted([5, fill])
    assert warped.red.dtype == np.dtype(dtype)


def test_reproject_records_nan_for_a_float_with_no_fill_value_and_warns() -> None:
    from geosave.geodata.warnings import AssumedFillWarning

    source = _small("float32", None)

    with pytest.warns(AssumedFillWarning, match="nan"):
        warped = reproject(source, _wider())

    assert np.isnan(warped.red.gs.nodata)
    assert np.isnan(warped.red.values).any()
    assert warped.red.dtype == np.dtype("float32")


def test_reproject_onto_a_geographic_grid_keeps_y_x_and_laziness() -> None:
    source = scene().isel(y=slice(0, 8), x=slice(0, 8)).chunk({"y": 2, "x": 2})

    warped = reproject(source, WGS84)

    assert warped.red.dims == ("y", "x")
    assert warped.gs.crs.epsg == 4326
    assert warped.red.chunks is not None
    assert warped.y.attrs["standard_name"] == "latitude"


def test_tree_reproject_resolves_numeric_resolution_once() -> None:
    source = build_tree({"image": scene().chunk()})
    warped = reproject(source, WGS84, resolution=0.001)
    assert grid_of(warped["image"].gs.to_dataset()).resolution.x == 0.001
    assert warped["image"].gs.to_dataset().red.chunks is not None


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rotation", [0, 30])
def test_alignment_target_is_independent_of_input_order(reverse, rotation) -> None:
    from affine import Affine

    origin = Affine.translation(300000, 5000000) * Affine.rotation(rotation)
    narrow = GeoBox((4, 4), origin * Affine.scale(10, -40), UTM)
    fine = GeoBox((4, 4), origin * Affine.scale(20, -10), UTM)
    rasters = [
        build_raster({"red": (("y", "x"), np.ones((4, 4), "uint16"))}, grid, nodata=0)
        for grid in (narrow, fine)
    ]
    if reverse:
        rasters.reverse()

    aligned = align(rasters, target=narrow)
    assert all(result.gs.geobox == narrow for result in aligned)


def test_tree_alignment_keeps_metadata_and_qualified_kernels():
    reference = placed(utm_box(size=4)).chunk({"y": 2, "x": 2})
    other = placed(offset_box(shift=20.0, size=4), name="vv").chunk({"y": 2, "x": 2})
    source = build_tree({"image": reference, "radar": other})
    source.attrs["title"] = "paired imagery"
    kernels = {"*": "nearest", "radar/vv": "bilinear"}

    from dask.callbacks import Callback

    with Callback(pretask=lambda *_: pytest.fail("alignment computed lazy pixels")):
        aligned = source.gs.align(target=reference, resampling=kernels)

    assert aligned.attrs["title"] == "paired imagery"
    assert aligned.gs.geobox == reference.gs.geobox
    assert aligned["radar"].gs.to_dataset().vv.chunks is not None
    # The source tree is left as it was.
    assert source.gs.geobox is None


GRID = GeoBox((4, 4), Affine(10, 0, 300000, 0, -10, 5000000), "EPSG:32633")


def image(grid=GRID, *, dtype="uint16", fill: float = 0):
    pixels = np.arange(1, grid.shape.y * grid.shape.x + 1, dtype=dtype).reshape(
        grid.shape.yx
    )
    return build_dataset({"red": (("y", "x"), pixels)}, grid, nodata=fill)


@pytest.mark.parametrize(
    "kind,lazy,dtype,fill",
    [
        ("array", False, "uint8", 0),
        ("array", True, "int16", -999),
        ("dataset", False, "int16", -999),
        ("dataset", True, "float32", np.nan),
        ("tree", False, "float32", np.nan),
        ("tree", True, "uint8", 0),
        ("dataset", False, "float64", -999.0),
        ("tree", True, "float64", -999.0),
    ],
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
    if kind == "tree":
        data = build_tree({"image": source})
        data.attrs["title"] = "collection"
    target = GeoBox((6, 6), GRID.transform * Affine.translation(-1, -1), GRID.crs)

    with Callback(pretask=lambda *_: pytest.fail("reproject computed lazy pixels")):
        result = reproject(data, target)

    output = result["image"].gs.to_dataset() if isinstance(result, xr.DataTree) else result
    band = output if kind == "array" else output.red
    assert type(result) is type(data)
    assert band.name == "red"
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
    if kind == "tree":
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
        first, second = align([west, east], target=west, extent=extent)
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
def test_tree_reproject_to_crs_covers_every_group(resolution):
    source = build_tree({"west": image(), "east": image(GRID.translate_pix(4, 0))})
    result = reproject(source, "EPSG:32633", resolution=resolution)
    assert grid_of(result).shape.yx == (4, 8)
    np.testing.assert_array_equal(
        result["west"].gs.to_dataset().red.values[:, :4], image().red
    )
    np.testing.assert_array_equal(
        result["east"].gs.to_dataset().red.values[:, 4:], image().red
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
    aligned = align(
        [image(grid) for grid in grids], target=fine, resolution="native", extent=extent
    )
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


def test_tree_variable_kernels_keep_class_codes_and_continuous_values_lazy():
    labels = image().gs.write_legend({1: "land", 2: "water"}, target="red")
    labels.red.values[:] = [[1, 1, 2, 2]] * 4
    source = build_tree(
        {"labels": labels.chunk(), "dem": image(dtype="float32").chunk()}
    )
    target = GRID.zoom_out(2)
    with Callback(pretask=lambda *_: pytest.fail("reproject computed lazy pixels")):
        result = reproject(
            source, target, resampling={"*": "average", "labels/red": "mode"}
        )
    np.testing.assert_array_equal(
        result["labels"].gs.to_dataset().red.values, [[1, 2], [1, 2]]
    )
    np.testing.assert_array_equal(
        result["dem"].gs.to_dataset().red.values, [[3.5, 5.5], [11.5, 13.5]]
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
    (aligned, _) = align([image(), shifted], target=GRID, extent="intersection")
    grid = aligned.gs.geobox
    # A half-pixel overlap is enclosed on the reference lattice, not clipped
    # into fractional cells or snapped inward to discard boundary coverage.
    assert grid == GRID


def test_native_tree_to_crs_preserves_resolutions_and_pixels():
    fine = GeoBox((6, 6), GRID.transform, GRID.crs)
    coarse = fine.zoom_out(3)
    source = build_tree({"fine": image(fine), "coarse": image(coarse)})
    result = reproject(source, "EPSG:32633", resolution="native")
    assert list(result.gs.rasters) == list(source.gs.rasters)
    for path, raster in source.gs.rasters.items():
        xr.testing.assert_identical(result.gs.rasters[path], raster)


def test_a_tree_target_with_no_root_grid_requires_an_explicit_group():
    target = build_tree({"r10": image(), "r20": image(GRID.zoom_out(2))})
    assert target.gs.geobox is None
    with pytest.raises(ValueError, match="select a reference group"):
        reproject(image(dtype="float32"), target)
    assert reproject(image(dtype="float32"), target["r10"].gs.to_dataset()).gs.geobox == GRID
