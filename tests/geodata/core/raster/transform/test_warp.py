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

from tests.geodata.conftest import build_raster as sample_raster
from tests.conftest import place


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
    return coded.gs.write_flags({1: "water", 2: "urban"}, target=["landcover"])


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


def test_a_colour_map_refuses_blending_without_flags() -> None:
    raster = build_raster(
        {"cover": (("y", "x"), np.ones(utm_box().shape, "uint8"))}, utm_box()
    )
    painted = raster.assign(
        cover=attrs.stamp_attrs(raster.cover, attrs.Display(color_map={1: "#00ff00"}))
    )

    with pytest.raises(ValueError, match="class codes"):
        painted.gs.reproject(WGS84, resampling="bilinear")


@pytest.mark.parametrize("resampling", ["nearest", "mode"])
def test_a_value_preserving_kernel_accepts_class_codes(
    resampling: Resampling,
) -> None:
    source = landcover().gs.write_nodata(255)
    warped = source.gs.reproject(WGS84, resampling=resampling)

    assert set(np.unique(warped.landcover.values)) <= {1, 2, 255}


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
def test_reproject_preserves_explicit_integer_nodata(dtype: str, fill: int) -> None:
    source = _small(dtype, fill)

    warped = source.gs.reproject(_wider())
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


@pytest.mark.parametrize("kind", ["array", "dataset"])
@pytest.mark.parametrize("lazy", [False, True])
def test_nonadjacent_spatial_dimensions_reproject_onto_trailing_spatial_axes(
    kind, lazy
):
    source = image().expand_dims(time=[1, 2]).transpose("y", "time", "x")
    data = source.red if kind == "array" else source
    if lazy:
        data = data.chunk()
    with Callback(pretask=lambda *_: pytest.fail("reproject computed lazy pixels")):
        result = data.gs.reproject(GRID.zoom_out(2))
    band = result if kind == "array" else result.red
    assert band.dims == ("time", "y", "x")
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


def test_different_resampling_methods_are_explicit_operations():
    labels = image().gs.write_flags({1: "land", 2: "water"}, target=["red"])
    labels.red.values[:] = [[1, 1, 2, 2]] * 4
    height = image(dtype="float32").rename(red="height")
    target = GRID.zoom_out(2)

    with Callback(pretask=lambda *_: pytest.fail("reprojection computed lazy pixels")):
        result = xr.merge(
            [
                labels.chunk().gs.reproject(target, resampling="mode"),
                height.chunk().gs.reproject(target, resampling="average"),
            ],
            join="exact",
        )

    np.testing.assert_array_equal(result.red.values, [[1, 2], [1, 2]])
    np.testing.assert_array_equal(result.height.values, [[3.5, 5.5], [11.5, 13.5]])


def test_selecting_a_tree_node_supplies_an_exact_destination():
    reference = build_tree({"r10": image(), "r20": image(GRID.zoom_out(2))})
    assert reference.gs.geobox is None
    result = image(dtype="float32").gs.reproject(reference["r10"].gs.geobox)
    assert result.gs.geobox == GRID


def test_dataset_reproject_preserves_temporal_bounds_and_source():
    source = image().expand_dims(time=[1, 2]).chunk()
    source = source.assign_coords(period=(("time", "bounds"), [[0, 1], [1, 2]]))
    original = source.copy(deep=False)

    with Callback(pretask=lambda *_: pytest.fail("reproject computed pixels")):
        result = source.gs.reproject(GRID.to_crs("EPSG:4326"))

    xr.testing.assert_identical(source, original)
    xr.testing.assert_equal(result.period.variable, source.period.variable)
    assert result.gs.crs.epsg == 4326
    assert hasattr(result.red.data, "dask")


@pytest.mark.parametrize("custom_dims", [False, True])
def test_generic_tree_reprojects_twice_preserving_coordinate_ownership(custom_dims):
    source = image().chunk()
    if custom_dims:
        source = source.rename(y="row", x="col")
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"title": "sensors"}),
            "run": xr.Dataset(coords={"acquired": np.datetime64("2025-01-01")}),
            "run/image": source,
        }
    )
    target = GRID.zoom_out(2)

    with Callback(pretask=lambda *_: pytest.fail("reproject computed pixels")):
        result = tree.gs.reproject(GRID).gs.reproject(target)

    assert result["run/image"].gs.geobox == target
    assert result["run/image"].to_dataset().red.dims == ("y", "x")
    assert result.attrs == tree.attrs
    assert "acquired" in result["run"].to_dataset(inherit=False).coords
    assert "acquired" not in result["run/image"].to_dataset(inherit=False).coords
    np.testing.assert_array_equal(
        result["run/image"].to_dataset().red.compute(), [[6, 8], [14, 16]]
    )
    assert tree["run/image"].gs.geobox == GRID


def test_reproject_selected_subtree_moves_its_data_bearing_root_and_children():
    source = image().rename(spatial_ref="crs")
    source.red.encoding["grid_mapping"] = "crs"
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"acquired": np.datetime64("2025-01-01")}),
            "run": source,
            "run/child": image().rename(red="nir"),
        }
    )
    target = GRID.zoom_out(2)

    result = tree["run"].gs.reproject(target)

    assert result.name == "run"
    assert result.gs.geobox == target
    assert result["child"].gs.geobox == target
    np.testing.assert_array_equal(result.to_dataset().red, [[6, 8], [14, 16]])
    np.testing.assert_array_equal(result["child"].to_dataset().nir, [[6, 8], [14, 16]])
    for raster in result.gs.datasets.values():
        for name in raster.gs.variables:
            assert raster[name].encoding["grid_mapping"] in raster.coords


def test_dataset_reproject_preserves_each_bands_nodata_and_values():
    source = image()
    source["nir"] = source.red.astype("float32").gs.write_nodata(-999)
    source = source.chunk()
    original = source.copy(deep=False)
    target = GeoBox((4, 8), GRID.affine, GRID.crs)

    with Callback(pretask=lambda *_: pytest.fail("reproject computed pixels")):
        result = source.gs.reproject(target)

    xr.testing.assert_identical(source, original)
    assert result.red.gs.nodata == 0
    assert result.nir.gs.nodata == -999
    np.testing.assert_array_equal(result.red[:, :4], source.red)
    np.testing.assert_array_equal(result.nir[:, :4], source.nir)
    assert (result.red[:, 4:] == 0).all().compute().item()
    assert (result.nir[:, 4:] == -999).all().compute().item()


def test_spatial_operations_preserve_independent_coordinates():
    scene = sample_raster().assign_coords(levels=("level", [1, 2, 3]))
    grid = scene.gs.geobox
    for result in (
        scene.pipe(place, grid),
        scene.gs.reproject(grid.to_crs("EPSG:3857")),
    ):
        xr.testing.assert_identical(result.levels.variable, scene.levels.variable)
        assert result.gs.geobox.crs is not None


def test_reprojection_names_axes_by_the_destination_grid():
    from pyproj import CRS

    scene = xr.Dataset(
        {
            "red": (
                ("latitude", "longitude"),
                np.ones((2, 2)),
                {"grid_mapping": "crs"},
            ),
        },
        coords={
            "latitude": [1.0, 0.0],
            "longitude": [0.0, 1.0],
            "crs": ((), 0, CRS.from_epsg(4326).to_cf()),
        },
    )
    target = scene.gs.geobox.to_crs("EPSG:3857")
    warped = scene.gs.reproject(target)
    assert warped.red.dims == ("y", "x")
    assert warped.gs.geobox == target
