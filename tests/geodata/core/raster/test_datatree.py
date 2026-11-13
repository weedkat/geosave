from __future__ import annotations


from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr

from geosave.geodata.core.raster.datatree import stac_catalog, stac_collection
from geosave.geodata import open_raster
from tests.conftest import build_tree
from geosave.geodata.io import storage
from tests.geodata.conftest import (
    _refuse,
    build_raster,
)
from tests.conftest import place

from shapely import from_wkt
from odc.geo.geom import Geometry


def build_stac_layout():
    """One collection, one Item, two assets: the layout stac.load_datatree builds."""
    scene = build_raster()
    return xr.DataTree.from_dict(
        {
            "s2/item-a": xr.Dataset(coords={"time": np.datetime64("2025-06-01", "ns")}),
            "s2/item-a/red": scene[["red"]],
            "s2/item-a/nir": scene[["nir"]],
        }
    )


def test_a_loaded_layout_is_a_stac_catalog_and_its_child_a_collection():
    tree = build_stac_layout()

    tree.gs.validate(stac_catalog)
    tree["s2"].gs.validate(stac_collection)
    tree["s2"].copy().gs.validate(stac_collection)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (
            {
                "s2/item-a": build_raster()[["red"]],
                "s2/item-a/nir": build_raster()[["nir"]],
            },
            "s2/item-a.*not a leaf",
        ),
        (
            {
                "s2/item-a/red": build_raster()[["red"]],
                "s2/nir": build_raster()[["nir"]],
            },
            "s2/nir.*Item/asset",
        ),
        ({"a/b/c/d": build_raster()[["red"]]}, "a/b/c/d.*Item/asset"),
    ],
    ids=["data on an Item node", "an asset without an Item", "too deep"],
)
def test_stac_catalog_refuses_and_names_the_node(layout, message):
    from geosave.geodata.errors import RasterValidationError

    with pytest.raises(RasterValidationError, match=message):
        xr.DataTree.from_dict(layout).gs.validate(stac_catalog)


def test_georeferenced_validation_checks_each_node():
    from geosave.geodata.errors import RasterValidationError

    scene = build_raster()
    tree = xr.DataTree.from_dict({"placed": scene})
    tree.gs.validate()

    pixels = xr.Dataset({"red": (("y", "x"), np.ones((2, 2)))})
    tree["nested/pixels"] = xr.DataTree(pixels)

    with pytest.raises(RasterValidationError) as caught:
        tree.gs.validate()

    assert caught.value.path == "nested/pixels"


def test_tree_region_combines_valid_footprints_in_different_crs() -> None:
    from odc.geo.geobox import GeoBox

    projected = build_raster(times=1)
    geographic = build_raster(times=1).pipe(
        place, GeoBox.from_bbox((111, -9, 111.01, -8.99), crs="EPSG:4326", shape=(2, 2))
    )
    tree = build_tree({"whole": projected, "geographic": geographic})
    expected = projected.gs.region.extent.union(
        geographic.gs.region.extent.to_crs(projected.gs.crs)
    )
    assert tree.gs.region.extent.geom.equals(expected.geom)


def test_tree_round_trip_preserves_groups_and_grid(
    tmp_path: Path, tree: xr.DataTree
) -> None:
    store = tree.gs.to_zarr(tmp_path / "tree.zarr")

    restored = open_raster(store)

    assert store == tmp_path / "tree.zarr"
    assert tuple(restored.children) == tuple(tree.children)
    assert restored.gs.geobox == tree.gs.geobox
    assert restored["optical"].gs.combine().red.dtype == np.dtype("uint16")


def test_a_single_store_is_not_a_tree(tmp_path, tree: xr.DataTree) -> None:
    store = tree["optical"].gs.combine().gs.to_zarr(tmp_path / "optical.zarr")

    with open_raster(store) as opened:
        assert isinstance(opened, xr.Dataset)


def test_a_tree_spans_from_its_earliest_group_to_its_latest() -> None:
    early = build_raster(times=2)
    late = early.assign_coords(time=early.time + np.timedelta64(1, "D"))

    timespan = build_tree({"a": early, "b": late}).gs.timespan

    assert timespan == (early.gs.timespan[0], late.gs.timespan[1])


def test_a_tree_lists_every_instant_any_group_reaches() -> None:
    early = build_raster(times=2)
    late = early.assign_coords(time=early.time + np.timedelta64(1, "D"))

    joint = build_tree({"a": early, "b": late}).gs.times

    assert joint is not None
    assert joint.strftime("%Y%m%d").tolist() == ["20250601", "20250602", "20250603"]


def test_a_tree_merges_onto_a_target_rasters_own_grid(tree: xr.DataTree) -> None:
    target = build_raster(crs="EPSG:32748").rename(red="other_red", nir="other_nir")

    merged = tree["optical"].gs.reproject(target.gs.geobox).gs.combine()

    assert merged.gs.geobox == target.gs.geobox


@pytest.mark.integration
def test_a_tree_round_trips_through_a_upath_carrying_its_options(remote) -> None:
    prefix, options = remote
    folder = storage.create_upath(f"{prefix}/s0", options)
    sample = build_tree(
        {"optical": build_raster(times=2), "label": build_raster(times=1)}
    )

    saved = sample.gs.to_zarr(folder / "scene.zarr")

    with open_raster(saved) as opened:
        assert tuple(opened.children) == ("optical", "label")


@pytest.mark.parametrize("method", ["to_netcdf", "to_zarr"])
def test_a_deferred_write_computes_to_the_eager_result(tmp_path, method):
    from dask.delayed import Delayed

    from dask.callbacks import Callback

    source = build_raster().chunk({"y": 1, "x": 1})
    tree = xr.DataTree.from_dict({"image": source, "label": source})
    suffix = ".nc" if method == "to_netcdf" else ".zarr"
    tasks = []
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        pending = getattr(tree.gs, method)(tmp_path / f"scene{suffix}", compute=False)
    assert tasks == []
    assert isinstance(pending, Delayed)
    assert list(tmp_path.iterdir()) == []

    saved = pending.compute()

    assert saved == tmp_path / f"scene{suffix}"
    with open_raster(saved) as restored:
        np.testing.assert_array_equal(
            restored["label"].to_dataset().red,
            tree["label"].to_dataset().red,
        )


def test_a_coarser_asset_averages_blocks_when_asked():
    tree = build_stac_catalog()
    coarse = tree["s2/2025-06-01/B11"].gs.geobox

    aligned = tree["s2"].gs.reproject(coarse, resampling="average")
    dataset = aligned.gs.combine()

    assert abs(dataset.gs.geobox.resolution.x) == 20
    fine = tree["s2/2025-06-01/B04"].dataset.B04.values
    assert float(dataset.B04.isel(time=0, y=0, x=0)) == float(fine[:2, :2].mean())


@pytest.mark.parametrize("shift", [0, 5])
def test_different_grids_require_alignment_without_reading_pixels(shift):
    from dask.callbacks import Callback

    from geosave.geodata.errors import RasterValidationError

    tree = build_stac_catalog().chunk()
    coarse = tree["s2/2025-06-01/B11"].gs.geobox
    moved = tree["s2"].gs.reproject(coarse)
    asset = moved["2025-06-06/B11"].to_dataset(inherit=False)
    moved["2025-06-06/B11"] = xr.DataTree(asset.assign_coords(x=asset.x + shift))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        if shift:
            with pytest.raises(RasterValidationError, match="reproject") as caught:
                moved.gs.combine()
            assert caught.value.path == "2025-06-06/B11"
        else:
            moved.gs.combine()
    assert tasks == []


def build_stac_catalog():
    """Two Items of one collection, each with a 10 m and a 20 m asset, plus a timeless DEM."""
    from tests.conftest import build_dataset
    from tests.geodata.conftest import geobox as make_geobox

    fine = make_geobox(shape=(4, 4))
    coarse = fine.zoom_out(2)
    nodes = {"s2": xr.Dataset(attrs={"title": "s2"})}
    for day, value in (("2025-06-01", 1.0), ("2025-06-06", 2.0)):
        when = np.datetime64(day, "ns")
        nodes[f"s2/{day}"] = xr.Dataset(coords={"time": when})
        nodes[f"s2/{day}/B04"] = build_dataset(
            {"B04": (("y", "x"), np.full((4, 4), value))}, fine
        )
        nodes[f"s2/{day}/B11"] = build_dataset(
            {"B11": (("y", "x"), np.full((2, 2), value))}, coarse
        )
    height = build_dataset({"height": (("y", "x"), np.zeros((4, 4)))}, fine)
    nodes["dem/cop/height"] = height
    return xr.DataTree.from_dict(nodes)


def test_items_merge_by_time_and_concat_along_it():
    tree = build_stac_catalog().filter(lambda node: "B11" not in node.data_vars)

    dataset = tree["s2"].gs.combine()

    assert dataset.sizes["time"] == 2
    assert list(dataset.data_vars) == ["B04"]
    np.testing.assert_array_equal(dataset.B04.isel(y=0, x=0).values, [1.0, 2.0])
    assert dataset.attrs["title"] == "s2"


def test_an_item_merges_its_assets_and_keeps_its_time_scalar():
    tree = build_stac_catalog()
    grid = tree["s2/2025-06-01/B04"].gs.geobox

    item = tree.gs.reproject(grid)["s2/2025-06-01"]
    assets = []
    for leaf in item.leaves:
        assets.append(leaf.to_dataset(inherit="all_coords"))
    dataset = xr.merge(assets, join="exact")

    assert set(dataset.data_vars) == {"B04", "B11"}
    assert dataset.time.ndim == 0
    assert dataset.gs.geobox == grid


def test_a_missing_asset_combines_as_nan():
    tree = build_stac_catalog()
    moved = tree.gs.reproject(tree["s2/2025-06-01/B04"].gs.geobox)
    gappy = moved.filter(lambda node: node.path != "/s2/2025-06-06/B11")

    dataset = gappy["s2"].gs.combine()

    assert not dataset.B11.isel(time=0).isnull().any()
    assert dataset.B11.isel(time=1).isnull().all()


def test_differing_grids_with_nothing_stated_name_the_grids():
    from geosave.geodata.errors import RasterValidationError

    with pytest.raises(RasterValidationError, match="2 grids.*reproject"):
        build_stac_catalog()["s2"].gs.combine()


def test_reproject_moves_every_leaf_lazily():
    tree = build_stac_catalog().chunk()
    grid = tree["s2/2025-06-01/B11"].gs.geobox

    with dask.config.set(scheduler=_refuse):
        moved = tree.gs.reproject(grid)

    for leaf in moved.leaves:
        assert leaf.to_dataset(inherit="all_coords").gs.geobox == grid
    assert moved["s2"].attrs == {"title": "s2"}


def test_a_reprojected_tree_chips_every_leaf_by_one_window():
    tree = build_stac_catalog()
    moved = tree.gs.reproject(tree["s2/2025-06-01/B04"].gs.geobox)

    chip = moved.isel(y=slice(0, 2), x=slice(0, 2))

    for leaf in chip.leaves:
        assert leaf.to_dataset(inherit="all_coords")[leaf.name].shape == (2, 2)
    times = chip["s2/2025-06-06"].to_dataset(inherit=False).time.values
    assert times == np.datetime64("2025-06-06")


def test_combine_after_reproject_stacks_the_collection_lazily():
    tree = build_stac_catalog().chunk()
    grid = tree["s2/2025-06-01/B04"].gs.geobox

    with dask.config.set(scheduler=_refuse):
        dataset = tree.gs.reproject(grid)["s2"].gs.combine()

    assert dataset.gs.geobox == grid
    assert set(dataset.data_vars) == {"B04", "B11"}
    assert dataset.sizes["time"] == 2


def test_items_over_different_extents_combine_onto_their_union():
    tree = build_stac_catalog().filter(lambda node: "B11" not in node.data_vars).chunk()
    late = tree["s2/2025-06-06/B04"].to_dataset(inherit=False)
    tree["s2/2025-06-06/B04"] = xr.DataTree(late.isel(x=slice(2, None)))

    with dask.config.set(scheduler=_refuse):
        dataset = tree["s2"].gs.combine()

    assert dataset.gs.geobox == tree["s2/2025-06-01/B04"].gs.geobox
    assert dataset.B04.dtype == np.dtype("float64")
    np.testing.assert_array_equal(dataset.B04.isel(time=0, y=0).values, [1, 1, 1, 1])
    np.testing.assert_array_equal(
        dataset.B04.isel(time=1, y=0).values, [np.nan, np.nan, 2, 2]
    )


def test_to_cog_writes_each_asset_at_its_node_path(tmp_path):
    scene = build_raster(times=2)
    tree = scene.gs.expand()

    files = tree.gs.to_cog(tmp_path)

    expected = []
    for leaf in tree.leaves:
        expected.append(tmp_path / f"{leaf.relative_to(tree)}.tif")
    assert [Path(file) for file in files] == expected
    with open_raster(files[2]) as restored:
        np.testing.assert_array_equal(restored.red, scene.red.isel(time=1))


def test_to_cog_refuses_a_tree_that_is_not_a_stac_catalog(tmp_path):
    from geosave.geodata.errors import RasterValidationError

    tree = xr.DataTree.from_dict({"s2/red": build_raster()[["red"]]})

    with pytest.raises(RasterValidationError, match="Item/asset"):
        tree.gs.to_cog(tmp_path)
    assert not list(tmp_path.iterdir())


def test_to_cog_defers_the_whole_batch(tmp_path):
    tree = build_raster(times=2).chunk().gs.expand()

    with dask.config.set(scheduler=_refuse):
        pending = tree.gs.to_cog(tmp_path, compute=False)
    assert not list(tmp_path.iterdir())
    # Two Items, each with a red and a nir asset.
    assert len(pending.compute()) == 4


@pytest.fixture
def city():
    return Geometry(from_wkt("POLYGON ((13 52, 14 52, 13 53, 13 52))"), "EPSG:4326")


def test_a_dataset_unions_variable_extents(city):
    left = build_raster()[["red"]].gs.write_extent(city)
    other = Geometry(from_wkt("POLYGON ((15 52, 16 52, 15 53, 15 52))"), "EPSG:4326")
    right = build_raster()[["nir"]].gs.write_extent(other.to_crs("EPSG:3857"))
    dataset = xr.Dataset({"red": left.red, "nir": right.nir})
    expected = city.union(other)
    assert dataset.gs.region.extent.geom.equals_exact(expected.geom, 1e-8)
