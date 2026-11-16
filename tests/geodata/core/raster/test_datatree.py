from __future__ import annotations


import dask
import numpy as np
import pytest
import xarray as xr

from geosave.geodata import open_raster_tree
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
    """One collection, one Item, two assets: a STAC tree layout."""
    scene = build_raster()
    return xr.DataTree.from_dict(
        {
            "s2/item-a": xr.Dataset(coords={"time": np.datetime64("2025-06-01", "ns")}),
            "s2/item-a/red": scene[["red"]],
            "s2/item-a/nir": scene[["nir"]],
        }
    )


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


@pytest.mark.integration
def test_a_tree_round_trips_through_a_upath_carrying_its_options(remote) -> None:
    prefix, options = remote
    folder = storage.create_upath(f"{prefix}/s0", options)
    sample = build_tree(
        {"optical": build_raster(times=2), "label": build_raster(times=1)}
    )

    saved = sample.gs.to_zarr(folder / "scene.zarr")

    with open_raster_tree(saved) as opened:
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
    with open_raster_tree(saved) as restored:
        np.testing.assert_array_equal(
            restored["label"].to_dataset().red,
            tree["label"].to_dataset().red,
        )


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


def test_general_tree_includes_data_on_parent_nodes():
    from geosave.geodata.core.raster.datatree import same_geobox

    scene = build_raster()
    tree = xr.DataTree.from_dict({"/": scene, "group": xr.Dataset()})
    tree.gs.validate(same_geobox)
    assert tree.gs.geobox == scene.gs.geobox
    assert tree.gs.spatial_dims == scene.gs.spatial_dims
    assert tree.gs.region.extent == scene.gs.region.extent


def test_empty_tree_has_no_common_grid():
    tree = xr.DataTree()
    assert tree.gs.geobox is None
    assert tree.gs.spatial_dims is None


def test_a_tree_converts_to_tensors_that_carry_their_grid():
    torch = pytest.importorskip("torch")
    from affine import Affine
    from odc.geo.geobox import GeoBox

    scene = build_raster(times=2).astype("float32")
    mask = (build_raster()[["red"]] > 1500).astype("uint8").rename(red="cover")
    tree = xr.DataTree.from_dict({"image": scene, "label": mask})

    sample = tree.gs.to_tensor(grid=True)

    assert list(sample) == ["image", "label", "transform", "crs"]
    assert sample["image"].shape == (2, 2, 2, 2)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].shape == (1, 2, 2)
    assert sample["label"].dtype == torch.uint8
    assert sample["crs"].item() == 32749
    grid = GeoBox(
        sample["image"].shape[-2:], Affine(*sample["transform"].tolist()), int(sample["crs"])
    )
    assert grid == scene.gs.geobox

    batch = torch.utils.data.default_collate([sample, sample])
    assert batch["image"].shape == (2, 2, 2, 2, 2)
    assert batch["transform"].shape == (2, 6)
    assert batch["crs"].tolist() == [32749, 32749]


def test_tensors_carry_only_the_pixels_unless_asked():
    pytest.importorskip("torch")

    sample = xr.DataTree.from_dict({"image": build_raster(times=2)}).gs.to_tensor()

    assert list(sample) == ["image"]


def test_context_names_add_their_readers_tensors():
    torch = pytest.importorskip("torch")
    from geosave.geodata import TENSOR_CONTEXT

    scene = build_raster(times=2)
    tree = xr.DataTree.from_dict({"image": scene})

    sample = tree.gs.to_tensor(context=["year", "centroid"])

    assert list(sample) == ["image", "year", "centroid"]
    torch.testing.assert_close(sample["centroid"], TENSOR_CONTEXT["centroid"](tree))
    with pytest.raises(KeyError, match="Unknown context"):
        tree.gs.to_tensor(context=["elevation"])


CUSTOM_CRS = "+proj=utm +zone=49 +south +ellps=GRS80 +towgs84=1,2,3 +units=m"


def test_tensors_need_one_grid_and_free_entry_names():
    pytest.importorskip("torch")
    from geosave.geodata.errors import RasterValidationError

    scene = build_raster()
    coarse = scene.coarsen(x=2, y=2).mean()
    with pytest.raises(RasterValidationError, match="GeoBoxes"):
        xr.DataTree.from_dict({"image": scene, "coarse": coarse}).gs.to_tensor()
    with pytest.raises(RasterValidationError):
        xr.DataTree().gs.to_tensor()
    with pytest.raises(ValueError, match="EPSG code"):
        tree = xr.DataTree.from_dict({"image": build_raster(crs=CUSTOM_CRS)})
        tree.gs.to_tensor(grid=True)
    with pytest.raises(ValueError, match="name of an entry"):
        xr.DataTree.from_dict({"crs": scene}).gs.to_tensor(grid=True)
    with pytest.raises(ValueError, match="name of an entry"):
        xr.DataTree.from_dict({"centroid": scene}).gs.to_tensor(context=["centroid"])
