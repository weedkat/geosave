"""Cutting rasters down to a region or a pixel window."""

from __future__ import annotations

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.transform.crop import crop, window
from tests.geodata.conftest import build_raster, titled
from tests.geodata.transform.conftest import grid_of


def test_crop_cuts_a_band_and_marks_outside_pixels_nodata() -> None:
    import geopandas as gpd
    import shapely
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.constructors import array

    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    band = array(np.ones(grid.shape, "uint16"), grid, dims=("y", "x"), nodata=0)
    triangle = gpd.GeoDataFrame(
        geometry=[shapely.Polygon([(0, 0), (20, 0), (0, 20)])], crs="EPSG:32633"
    )

    cropped = band.gs.crop(triangle)

    assert cropped.shape == (2, 2)
    assert cropped.dtype == np.dtype("uint16")
    assert (cropped.values == 0).any() and (cropped.values == 1).any()


def test_crop_refuses_a_raster_without_crs_and_names_the_accessor() -> None:
    import geopandas as gpd
    import shapely
    import xarray as xr

    band = xr.DataArray(np.ones((4, 4), "uint16"), dims=("y", "x"))
    square = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 2, 2)], crs="EPSG:32633")

    with pytest.raises(ValueError, match=r"gs\.write_crs"):
        crop(band, square)


def _triangle():
    import geopandas as gpd
    import shapely

    return gpd.GeoDataFrame(
        geometry=[shapely.Polygon([(0, 0), (30, 0), (0, 30)])], crs="EPSG:32633"
    )


def _band(times: int = 0):
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.constructors import array

    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    band = array(np.ones(grid.shape, "uint16"), grid, dims=("y", "x"), nodata=0).rename(
        "red"
    )
    if not times:
        return band
    return band.expand_dims(time=np.arange(times).astype("datetime64[D]")).copy()


def test_crop_masks_pixel_centres_inside_the_geometry() -> None:
    cropped = _band().gs.crop(_triangle())

    assert cropped.values.tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_masks_every_instant_of_a_time_axis() -> None:
    cropped = _band(times=2).gs.crop(_triangle())

    assert cropped.shape == (2, 3, 3)
    assert cropped.values[0].tolist() == cropped.values[1].tolist()
    assert cropped.values[1].tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_cuts_every_group_of_a_stack_and_keeps_root_attrs() -> None:
    from geosave_engine.geodata.core.stack import stack

    band = _band()
    tree = stack(
        {"optical": band.to_dataset(), "dem": band.rename("height").to_dataset()}
    )
    tree = titled(tree)

    cropped = crop(tree, _triangle())

    assert cropped.gs.groups == ("optical", "dem")
    assert grid_of(cropped).shape == (3, 3)
    assert cropped.gs.rasters["dem"].height.values.tolist() == [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
    assert cropped.attrs["title"] == "scene"


def test_crop_refuses_an_empty_vector() -> None:

    with pytest.raises(ValueError, match="empty vector"):
        _band().gs.crop(gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:32633")))


def test_crop_masks_a_geographic_grid_whose_coordinates_are_not_float_exact() -> None:
    import geopandas as gpd
    import shapely
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.constructors import array

    step = 1 / 3600
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.0 + 20 * step, 45.0 + 20 * step),
        crs="EPSG:4326",
        resolution=step,
    )
    band = array(np.ones(grid.shape, "uint16"), grid, dims=("y", "x"), nodata=0)
    corner = gpd.GeoDataFrame(
        geometry=[
            shapely.Polygon(
                [
                    (13.0 + 3.3 * step, 45.0 + 3.3 * step),
                    (13.0 + 9.7 * step, 45.0 + 3.3 * step),
                    (13.0 + 3.3 * step, 45.0 + 9.7 * step),
                ]
            )
        ],
        crs="EPSG:4326",
    )

    cropped = band.gs.crop(corner)

    assert cropped.dtype == np.dtype("uint16")
    assert (cropped.values == 0).any() and (cropped.values == 1).any()


def test_crop_masks_a_raster_whose_leading_axis_holds_no_dates() -> None:
    band = _band().expand_dims(time=[0, 1]).copy()

    cropped = band.gs.crop(_triangle())

    assert cropped.shape == (2, 3, 3)
    assert cropped.values[1].tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_reads_no_pixels_of_a_chunked_raster() -> None:
    cropped = _band(times=2).chunk().gs.crop(_triangle())

    assert cropped.chunks is not None


def test_crop_takes_a_vector_in_another_crs() -> None:
    import geopandas as gpd
    import shapely

    # Corners off the pixel edges, so a round trip through degrees moves no edge.
    triangle = gpd.GeoDataFrame(
        geometry=[shapely.Polygon([(3, 3), (27, 3), (3, 27)])], crs="EPSG:32633"
    )

    native = _band().gs.crop(triangle)
    geographic = _band().gs.crop(triangle.to_crs("EPSG:4326"))

    assert geographic.shape == native.shape
    assert geographic.values.tolist() == native.values.tolist()


def test_crop_takes_an_odc_geometry_carrying_its_own_crs() -> None:
    from odc.geo.geom import box

    cropped = _band().gs.crop(box(10, 10, 30, 30, "EPSG:32633"))

    assert cropped.shape == (2, 2)
    assert cropped.values.tolist() == [[1, 1], [1, 1]]


def test_crop_takes_a_geoseries() -> None:
    cropped = _band().gs.crop(_triangle().geometry)

    assert cropped.values.tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_keeps_every_disconnected_geometry_and_masks_the_gap() -> None:
    import geopandas as gpd
    import shapely

    corners = gpd.GeoDataFrame(
        geometry=[shapely.box(0, 0, 10, 10), shapely.box(30, 30, 40, 40)],
        crs="EPSG:32633",
    )

    cropped = _band().gs.crop(corners)

    assert cropped.shape == (4, 4)
    # North is the first row, so the box at y 30-40 is the top-right pixel.
    assert cropped.values.tolist() == [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]


def test_crop_refuses_a_bare_shapely_geometry_that_names_no_crs() -> None:
    import shapely

    with pytest.raises(TypeError, match="CRS"):
        _band().gs.crop(shapely.box(10, 10, 30, 30))  # pyright: ignore[reportArgumentType]


def test_crop_refuses_an_odc_geometry_that_names_no_crs() -> None:
    from odc.geo.geom import box

    with pytest.raises(TypeError, match="CRS"):
        _band().gs.crop(box(10, 10, 30, 30, None))


@pytest.mark.parametrize("kind", ["array", "dataset", "tree"])
def test_pixel_window_preserves_native_type_time_and_grid_lazily(kind):
    band = _band(times=2).chunk().assign_attrs(description="reflectance")
    if kind == "array":
        source = band
    elif kind == "dataset":
        source = band.to_dataset()
    else:
        source = stack({"image": band.to_dataset()})

    with Callback(pretask=lambda *_: pytest.fail("window computed pixels")):
        result = window(source, (2, 2), (3, 3), mode="edge")

    assert type(result) is type(source)
    if isinstance(result, xr.DataTree):
        output = result.gs.rasters["image"].red
    elif isinstance(result, xr.Dataset):
        output = result.red
    else:
        output = result
    assert output.shape == (2, 3, 3)
    assert isinstance(output.data, da.Array)
    assert output.attrs["description"] == "reflectance"
    np.testing.assert_array_equal(output.time, band.time)
    assert output.gs.geobox == grid_of(band).translate_pix(2, 2).crop((3, 3))
    np.testing.assert_array_equal(output.values, np.ones((2, 3, 3)))


def test_a_window_past_the_edge_is_filled_as_asked():
    data = build_raster(times=1).chunk()

    actual = window(data, (0, 1), (1, 1))

    assert actual.sizes["y"] == actual.sizes["x"] == 1
    assert actual.gs.geobox == data.gs.geobox.translate_pix(1, 0).crop((1, 1))
    assert actual.red.chunks is not None


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("mode", ["constant", "edge", "reflect"])
def test_window_negative_offset_pads_before_the_source(mode, lazy):
    import xarray as xr

    source = xr.DataArray(np.arange(1, 10).reshape(3, 3), dims=("y", "x"))
    if lazy:
        source = source.chunk()
    with Callback(pretask=lambda *_: pytest.fail("crop computed pixels")):
        result = window(source, (-1, -1), (3, 3), mode=mode, constant_value=0)
    expected = {
        "constant": [[0, 0, 0], [0, 1, 2], [0, 4, 5]],
        "edge": [[1, 1, 2], [1, 1, 2], [4, 4, 5]],
        "reflect": [[5, 4, 5], [2, 1, 2], [5, 4, 5]],
    }
    np.testing.assert_array_equal(result.values, expected[mode])


def test_window_refuses_wrap_even_without_padding():
    source = build_raster()
    with pytest.raises(ValueError, match="wrap"):
        window(source, (0, 0), (1, 1), mode="wrap")
