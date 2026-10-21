from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from dask.callbacks import Callback
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata import raster, tree
from geosave.geodata.core.raster.create import raster as build_raster
from geosave.geodata.core.raster.create import tree as build_tree
from geosave.geodata.transform.warp import Resampling, align, reproject

from .conftest import build, geobox, grid_of

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


def test_a_geobox_target_lands_exactly_on_it() -> None:
    target = utm_box().to_crs(WGS84)

    warped = reproject(scene(), target)

    assert warped.gs.geobox == target


@pytest.mark.parametrize("mode", ["direct", "accessor", "tree"])
def test_fill_warning_points_to_the_caller(mode: str) -> None:
    from geosave.geodata.warnings import AssumedFillWarning

    source = landcover()
    with pytest.warns(AssumedFillWarning) as caught:
        if mode == "direct":
            reproject(source, WGS84)
        elif mode == "accessor":
            source.gs.reproject(WGS84)
        else:
            reproject(build_tree({"cover": source}), WGS84)

    assert all(Path(warning.filename) == Path(__file__) for warning in caught)


def test_a_warped_raster_reports_the_crs_it_landed_on() -> None:
    source = build_raster(
        {"elevation": (("y", "x"), np.ones(utm_box().shape, "float32"))},
        utm_box(),
    )

    warped = reproject(source, WGS84)

    assert warped.gs.crs == "EPSG:4326"
    # odc's own Dataset path leaves the source CRS in spatial_ref's attrs.
    assert source.odc.reproject(WGS84).odc.crs.epsg == 32633


def test_a_warped_raster_trees_with_the_grid_it_targeted() -> None:
    imagery = scene()
    dem_box = GeoBox.from_bbox((12.9, 45.0, 13.2, 45.2), crs=WGS84, resolution=0.0003)
    dem = build_raster(
        {"elevation": (("y", "x"), np.ones(dem_box.shape, "float32"))}, dem_box
    )

    warped = reproject(dem, imagery, resampling="bilinear")

    assert build_tree({"optical": imagery, "dem": warped}).gs.groups == (
        "optical",
        "dem",
    )


def test_warping_keeps_the_raster_and_variable_attrs() -> None:
    source = scene().assign_attrs(title="Sentinel-2 Level-2A")

    warped = reproject(source, WGS84)

    assert warped.attrs["title"] == "Sentinel-2 Level-2A"
    assert warped.red.attrs["_FillValue"] == 0


def test_warping_a_band_returns_a_band() -> None:
    warped = reproject(scene().red, WGS84)

    assert isinstance(warped, xr.DataArray)
    assert warped.gs.crs.epsg == 4326


def test_warping_a_tree_leaves_every_group_on_one_grid() -> None:
    source = build_tree({"optical": scene(), "cover": landcover()})

    warped = reproject(source, WGS84)

    grids = {name: group.gs.geobox for name, group in warped.gs.rasters.items()}
    assert warped.gs.groups == ("optical", "cover")
    assert len(set(grids.values())) == 1
    assert warped.gs.crs == "EPSG:4326"


def test_warping_a_tree_onto_a_geobox_lands_every_group_on_it() -> None:
    target = utm_box().to_crs(WGS84)
    source = build_tree({"optical": scene(), "cover": landcover()})

    warped = reproject(source, target)

    assert all(group.gs.geobox == target for group in warped.gs.rasters.values())


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


def test_a_blending_kernel_names_the_group_holding_class_codes() -> None:
    source = build_tree({"optical": scene(), "cover": landcover()})

    with pytest.raises(ValueError, match=r"cover/landcover"):
        reproject(source, WGS84, resampling="bilinear")


def test_reproject_honours_a_requested_resolution() -> None:
    warped = reproject(scene(), WGS84, resolution=0.001)

    assert warped.gs.resolution.x == 0.001


def test_matching_adopts_a_rasters_grid_without_naming_a_geobox() -> None:
    imagery = scene()
    dem_box = GeoBox.from_bbox((12.9, 45.0, 13.2, 45.2), crs=WGS84, resolution=0.0003)
    dem = build_raster(
        {"elevation": (("y", "x"), np.ones(dem_box.shape, "float32"))}, dem_box
    )

    assert reproject(dem, imagery).gs.geobox == imagery.gs.geobox


def test_a_variable_off_the_grid_is_refused_by_name() -> None:
    source = scene()
    source["profile"] = ("depth", np.arange(3.0))

    with pytest.raises(ValueError, match="'profile' spans"):
        reproject(source, WGS84)


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


def test_align_puts_every_raster_on_one_exact_grid() -> None:
    aligned = align(
        [placed(utm_box()), placed(offset_box(), fill=9, name="vv")], target=utm_box()
    )

    assert aligned[0].gs.geobox == aligned[1].gs.geobox
    assert build_tree({"a": aligned[0], "b": aligned[1]}).gs.groups == ("a", "b")


def test_align_uses_the_requested_reference_grid() -> None:
    first, second = placed(utm_box()), placed(offset_box(), name="vv")

    on_first = align([first, second], target=first)
    on_second = align([first, second], target=second, extent="match")

    assert on_first[0].gs.geobox == first.gs.geobox
    assert on_second[0].gs.geobox == second.gs.geobox


def test_align_reaches_every_raster_when_it_unions() -> None:
    west = placed(utm_box(size=4))
    east = placed(offset_box(shift=20.0, size=4), fill=9, name="vv")

    aligned = align([west, east], target=west, extent="union")

    assert aligned[0].gs.geobox.shape.x > west.gs.geobox.shape.x
    assert aligned[0].gs.bounds.left == pytest.approx(west.gs.bounds.left)
    assert aligned[0].gs.bounds.right == pytest.approx(east.gs.bounds.right, abs=10)


def test_align_keeps_only_common_ground_when_it_intersects() -> None:
    west = placed(utm_box(size=4))
    east = placed(offset_box(shift=20.0, size=4), fill=9, name="vv")

    aligned = align([west, east], target=west, extent="intersection")

    assert aligned[0].gs.geobox.shape.x < west.gs.geobox.shape.x
    assert aligned[0].gs.geobox == aligned[1].gs.geobox


def test_align_carries_a_raster_across_crs_and_resolution() -> None:
    metric = placed(utm_box(size=8))
    coarse = build_raster(
        {
            "dem": (
                ("y", "x"),
                np.full((4, 4), 5.0, "float32"),
            )
        },
        GeoBox.from_bbox((300000, 5000000, 300080, 5000080), crs=UTM, resolution=20),
        nodata=-9999.0,
    )

    aligned = align([metric, coarse], target=metric, extent="match")

    assert aligned[1].gs.geobox == metric.gs.geobox
    assert aligned[1].gs.resolution == metric.gs.resolution


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


def test_reproject_keeps_a_nan_fill_value() -> None:
    source = _small("float32", np.nan)

    warped = reproject(source, _wider())
    band = reproject(source.red, _wider())

    assert np.isnan(warped.red.gs.nodata)
    assert np.isnan(band.gs.nodata)
    assert np.isnan(warped.red.values).any()


def test_reproject_keeps_a_bands_name() -> None:
    assert reproject(_small("uint16", 0).red, _wider()).name == "red"


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


def test_reproject_stays_quiet_for_a_variable_carrying_a_fill_value() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warped = reproject(_small("uint16", 0), _wider())

    assert sorted(np.unique(warped.red.values).tolist()) == [0, 5]


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


def test_unnamed_class_array_refuses_blending() -> None:
    with pytest.raises(ValueError, match="class codes"):
        reproject(landcover().landcover.rename(None), WGS84, resampling="bilinear")


def test_tree_reproject_resolves_numeric_resolution_once() -> None:
    source = build_tree({"image": scene().chunk()})
    warped = reproject(source, WGS84, resolution=0.001)
    assert grid_of(warped.gs.rasters["image"]).resolution.x == 0.001
    assert warped.gs.rasters["image"].red.chunks is not None


def test_union_alignment_covers_every_tree_group() -> None:
    source = scene()
    eastern = build_raster(
        {"red": (("y", "x"), np.full((256, 256), 2, dtype="uint16"))},
        source.gs.geobox.translate_pix(256, 0),
        nodata=0,
    )
    tree = build_tree({"west": source, "east": eastern})
    grid = tree.gs.align(target=source, extent="union").gs.geobox
    assert grid.shape.yx == (256, 512)
    assert grid.transform == source.gs.geobox.transform


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


def test_alignment_requires_an_explicit_target():
    source = placed(utm_box())
    with pytest.raises(TypeError, match="target"):
        align([source])
    with pytest.raises(TypeError, match="target"):
        build_tree({"image": source}).gs.align()


def test_alignment_defaults_to_exact_target_for_rasters_and_trees():
    reference = placed(utm_box(size=4))
    other = placed(offset_box(shift=20.0, size=4), name="vv")
    aligned = align([reference, other], target=reference)
    result = build_tree({"image": reference, "radar": other}).gs.align(target=reference)

    assert all(raster.gs.geobox == reference.gs.geobox for raster in aligned)
    assert result.gs.geobox == reference.gs.geobox
    np.testing.assert_array_equal(result.gs.rasters["radar"].vv, aligned[1].vv)


def test_tree_alignment_inplace_keeps_metadata_and_qualified_kernels():
    reference = placed(utm_box(size=4)).chunk({"y": 2, "x": 2})
    other = placed(offset_box(shift=20.0, size=4), name="vv").chunk({"y": 2, "x": 2})
    source = build_tree({"image": reference, "radar": other})
    source.attrs["title"] = "paired imagery"
    kernels = {"*": "nearest", "radar/vv": "bilinear"}
    expected = source.gs.align(target=reference, resampling=kernels)

    from dask.callbacks import Callback

    with Callback(pretask=lambda *_: pytest.fail("alignment computed lazy pixels")):
        assert (
            source.gs.align(target=reference, resampling=kernels, inplace=True) is None
        )

    assert source.attrs["title"] == "paired imagery"
    assert source.gs.geobox == reference.gs.geobox
    assert source.gs.rasters["radar"].vv.chunks is not None
    xr.testing.assert_equal(source.compute(), expected.compute())


GRID = GeoBox((4, 4), Affine(10, 0, 300000, 0, -10, 5000000), "EPSG:32633")


def image(grid=GRID, *, dtype="uint16", fill: float = 0):
    pixels = np.arange(1, grid.shape.y * grid.shape.x + 1, dtype=dtype).reshape(
        grid.shape.yx
    )
    return raster({"red": (("y", "x"), pixels)}, grid, nodata=fill)


@pytest.mark.parametrize("kind", ["array", "dataset", "tree"])
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
    if kind == "tree":
        data = tree({"image": source})
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
    source = tree({"west": image(), "east": image(GRID.translate_pix(4, 0))})
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


@pytest.mark.parametrize("operation", [reproject, align])
def test_empty_tree_has_a_clear_grid_error(operation):
    with pytest.raises(ValueError, match="grid|groups|no data variables"):
        if operation is reproject:
            operation(xr.DataTree(), GRID)
        else:
            operation([xr.DataTree()], target=GRID)


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
    source = tree({"labels": labels.chunk(), "dem": image(dtype="float32").chunk()})
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
    (aligned, _) = align([image(), shifted], target=GRID, extent="intersection")
    grid = aligned.gs.geobox
    # A half-pixel overlap is enclosed on the reference lattice, not clipped
    # into fractional cells or snapped inward to discard boundary coverage.
    assert grid == GRID


def test_native_tree_to_crs_preserves_resolutions_and_pixels():
    fine = GeoBox((6, 6), GRID.transform, GRID.crs)
    coarse = fine.zoom_out(3)
    source = tree({"fine": image(fine), "coarse": image(coarse)})
    result = reproject(source, "EPSG:32633", resolution="native")
    for name in ("fine", "coarse"):
        xr.testing.assert_identical(result.gs.rasters[name], source.gs.rasters[name])


def test_a_tree_target_with_no_root_grid_requires_an_explicit_group():
    target = tree({"r10": image(), "r20": image(GRID.zoom_out(2))})
    assert target.gs.geobox is None
    with pytest.raises(ValueError, match="select a reference group"):
        reproject(image(dtype="float32"), target)
    assert reproject(image(dtype="float32"), target.gs.rasters["r10"]).gs.geobox == GRID


def test_reproject_onto_a_new_grid_stays_lazy():
    source = image().chunk()
    with Callback(pretask=lambda *_: pytest.fail("reproject computed pixels")):
        result = reproject(source, GRID.zoom_out(2))
    assert isinstance(result.red.data, da.Array)


def test_reproject_preserves_tree_root_attrs():
    source = tree({"optical": build(geobox())})
    source.attrs = {"title": "training scene", "foreign": {"site": 7}}
    coarse = grid_of(source).zoom_out(2)
    result = reproject(source, coarse)
    assert result.gs.geobox == coarse
    assert result.attrs == source.attrs
