"""Cutting rasters down to a region or a pixel window."""

from __future__ import annotations

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tests.conftest import build_tree
from tests.geodata.conftest import titled


def _triangle():
    import shapely

    return gpd.GeoDataFrame(
        geometry=[shapely.Polygon([(0, 0), (30, 0), (0, 30)])], crs="EPSG:32633"
    )


def _band(times: int = 0):
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataarray

    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    band = build_dataarray(
        np.ones(grid.shape, "uint16"), grid, dims=("y", "x"), nodata=0
    ).rename("red")
    if not times:
        return band
    return band.expand_dims(time=np.arange(times).astype("datetime64[D]")).copy()


def test_crop_masks_pixel_centres_inside_the_geometry() -> None:
    cropped = _band().gs.crop(_triangle())

    assert cropped.dtype == np.dtype("uint16")
    assert cropped.values.tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_masks_every_instant_of_a_time_axis() -> None:
    cropped = _band(times=2).gs.crop(_triangle())

    assert cropped.shape == (2, 3, 3)
    assert cropped.values[0].tolist() == cropped.values[1].tolist()
    assert cropped.values[1].tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_cuts_every_group_of_a_tree_and_keeps_root_attrs() -> None:

    band = _band(times=1)
    tree = build_tree(
        {"optical": band.to_dataset(), "dem": band.rename("height").to_dataset()}
    )
    tree = titled(tree)

    # A tree is cut node by node through its accessor.
    cropped = tree.gs.crop(_triangle())

    assert tuple(cropped.children) == ("optical", "dem")
    assert cropped.gs.geobox.shape == (3, 3)
    assert cropped["dem"].gs.combine().height.isel(time=0).values.tolist() == [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
    assert cropped.attrs["title"] == "scene"


def test_crop_masks_a_geographic_grid_whose_coordinates_are_not_float_exact() -> None:
    import shapely
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataarray

    step = 1 / 3600
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.0 + 20 * step, 45.0 + 20 * step),
        crs="EPSG:4326",
        resolution=step,
    )
    band = build_dataarray(
        np.ones(grid.shape, "uint16"), grid, dims=("y", "x"), nodata=0
    )
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


def test_crop_reads_no_pixels_of_a_chunked_raster() -> None:
    cropped = _band(times=2).chunk().gs.crop(_triangle())

    assert cropped.chunks is not None


def test_crop_takes_a_vector_in_another_crs() -> None:
    import shapely

    # Corners off the pixel edges, so a round trip through degrees moves no edge.
    triangle = gpd.GeoDataFrame(
        geometry=[shapely.Polygon([(3, 3), (27, 3), (3, 27)])], crs="EPSG:32633"
    )

    native = _band().gs.crop(triangle)
    geographic = _band().gs.crop(triangle.to_crs("EPSG:4326"))

    assert geographic.shape == native.shape
    assert geographic.values.tolist() == native.values.tolist()


def _scene(times: int = 2):
    band = _band(times).chunk({"y": 2, "x": 2})
    band = band.gs.write_extent(band.gs.geobox.extent)
    return band.to_dataset().gs.write_extent(band.gs.geobox.extent)


@pytest.mark.parametrize("mode", ["constant", "reflect"])
@pytest.mark.parametrize("window", [(-1, -2, 3, 3), (1, 1, 2, 2), (2, 3, 4, 4)])
def test_crop_to_grid_matches_numpy_tile_padding_lazily(mode, window):
    scene = _scene()
    top, left, height, width = window
    grid = scene.gs.geobox.translate_pix(left, top).crop((height, width))
    with Callback(pretask=lambda *_: pytest.fail("crop_to_grid computed pixels")):
        chip = scene.gs.crop_to_grid(grid, mode=mode, constant_value=7)
    assert isinstance(chip.red.data, da.Array)
    assert chip.gs.geobox == grid
    values = scene.red.values
    inside = values[:, max(top, 0) : top + height, max(left, 0) : left + width]
    widths = (
        (0, 0),
        (max(-top, 0), max(top + height - 4, 0)),
        (max(-left, 0), max(left + width - 4, 0)),
    )
    if mode == "constant":
        expected = np.pad(inside, widths, mode=mode, constant_values=7)
    else:
        expected = np.pad(inside, widths, mode=mode)
    np.testing.assert_array_equal(chip.red.values, expected)
    assert chip.red.dtype == np.uint16


def test_crop_to_grid_keeps_model_context_and_clips_extent():
    scene = _scene()
    grid = scene.gs.geobox.translate_pix(3, 3).crop((2, 2))
    chip = scene.gs.crop_to_grid(grid, mode="reflect")
    np.testing.assert_array_equal(chip.time.values, scene.time.values)
    assert chip.time.attrs == scene.time.attrs
    assert chip.red.gs.nodata == 0
    inside = scene.gs.geobox.translate_pix(3, 3).crop((1, 1)).extent
    for extent in (chip.gs.region.extent, chip.red.gs.region.extent):
        assert extent.symmetric_difference(inside).area < 1e-6
    assert chip.spatial_ref.attrs["GeoTransform"].split()[::3] == ["30", "10"]


def test_crop_to_grid_rejects_unaligned_or_disjoint_grids():
    scene = _scene()
    grid = scene.gs.geobox
    with pytest.raises(ValueError, match="pixel-aligned"):
        scene.gs.crop_to_grid(grid.translate_pix(0.5, 0))
    with pytest.raises(ValueError, match="pixel-aligned"):
        scene.gs.crop_to_grid(grid.zoom_to(resolution=5))
    with pytest.raises(ValueError, match="overlap"):
        scene.gs.crop_to_grid(grid.translate_pix(10, 0))
    with pytest.raises(ValueError, match="grid"):
        xr.DataArray(np.ones((2, 2)), dims=("y", "x")).gs.crop_to_grid(grid)


def test_selected_chip_round_trips_through_geotiff(tmp_path):
    from geosave.geodata import open_raster

    scene = _scene(times=0)
    grid = scene.gs.geobox.translate_pix(2, 1).crop((3, 3))
    chip = scene.gs.crop_to_grid(grid, mode="edge")
    path = chip.red.gs.to_cog(tmp_path / "chip.tif")
    with open_raster(path) as restored:
        assert restored.gs.geobox == grid
        np.testing.assert_array_equal(restored.red.values, chip.red.values)
