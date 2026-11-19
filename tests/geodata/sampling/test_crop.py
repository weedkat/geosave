import random

import pytest
import xarray as xr
from dask.callbacks import Callback

import geosave as gs
from tests.geodata.sampling.conftest import GRID, build_scene, build_tree


def test_seeded_crop_shares_one_window_across_raster_types():
    tree = build_tree()
    scene = tree["image"].to_dataset()
    scene["red"].attrs.update(units="1", _FillValue=-9999.0)
    tree["image"] = scene
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        image = gs.random_crop(scene, (3, 5), seed=42)
        band = gs.random_crop(scene["red"], (3, 5), seed=42)
        group = gs.random_crop(tree, (3, 5), seed=42)
    assert not tasks
    assert image.gs.geobox.shape.yx == (3, 5)
    assert image.gs.geobox == band.gs.geobox == group.gs.common_geobox()
    xr.testing.assert_identical(image["red"], band)
    xr.testing.assert_identical(group["image/red"], band)
    assert band.attrs == scene["red"].attrs
    xr.testing.assert_equal(image, scene.sel(x=image.x, y=image.y))
    xr.testing.assert_equal(
        group["label/cover"], tree["label/cover"].sel(x=image.x, y=image.y)
    )


def test_random_origins_include_every_in_bounds_pixel():
    scene = build_scene()
    grids = {gs.random_crop(scene, 9, seed).gs.geobox for seed in range(20)}
    assert grids == {GRID[y : y + 9, x : x + 9] for y in (0, 1) for x in (0, 1)}
    state = random.getstate()
    gs.random_crop(scene, 9, seed=1)
    assert random.getstate() == state
    assert gs.random_crop(scene, 10).gs.geobox == GRID


@pytest.mark.parametrize("size", [0, -1, (11, 2), (2, 11)])
def test_crop_must_fit(size):
    with pytest.raises(ValueError, match="must fit"):
        gs.random_crop(build_scene(), size)
