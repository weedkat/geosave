from __future__ import annotations


from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr

from geosave.geodata import open_raster
from tests.conftest import build_tree
from geosave.geodata.io import storage
from tests.geodata.conftest import (
    _refuse,
    build_raster,
)
from tests.geodata.conftest import titled as _titled
from tests.geodata.core.raster.conftest import optical
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


@pytest.mark.parametrize(
    "select", ["/", "s2", "s2/item-a"], ids=["tree", "collection", "item"]
)
def test_stac_pattern_accepts_any_stac_subtree(select):
    tree = build_stac_layout()[select]

    tree.gs.validate(stac_pattern=True)
    tree.copy().gs.validate(stac_pattern=True)


def test_stac_pattern_accepts_a_collection_after_an_operation():
    collection = build_stac_layout()["s2"]

    collection.gs.reproject(build_raster().gs.geobox).gs.validate(stac_pattern=True)


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
            "depths 2 and 3",
        ),
        ({"a/b/c/d": build_raster()[["red"]]}, "a/b/c/d.*deeper"),
    ],
    ids=["data on an Item node", "leaves at two depths", "depth 4"],
)
def test_stac_pattern_refuses_and_names_the_node(layout, message):
    from geosave.geodata.errors import RasterValidationError

    with pytest.raises(RasterValidationError, match=message) as caught:
        xr.DataTree.from_dict(layout).gs.validate(stac_pattern=True)
    assert "load_datatree" in str(caught.value)


def test_any_tree_of_valid_rasters_validates_without_a_pattern():
    scene = build_raster()
    tree = xr.DataTree.from_dict({"fine": scene, "nested/deeper/coarse": scene})

    tree.gs.validate()


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


def test_grid_summary_tracks_cropped_coordinates_without_reading_pixels():
    from dask.callbacks import Callback

    source = build_raster().chunk({"y": 2, "x": 2})
    cropped = source.isel(y=slice(1, 3), x=slice(1, 4))
    tree = xr.DataTree.from_dict({"s2/native": source, "s2/crop": cropped})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        nodes = tree.gs.datasets
    assert tasks == []
    assert list(nodes) == ["s2/native", "s2/crop"]
    assert nodes["s2/native"].gs.geobox == source.gs.geobox
    assert nodes["s2/crop"].gs.geobox == cropped.gs.geobox
    assert (
        nodes["s2/native"].gs.geobox.transform != nodes["s2/crop"].gs.geobox.transform
    )


def test_one_groups_scalar_time_does_not_date_the_others() -> None:
    dated = build_raster(times=1).isel(time=0)
    undated = build_raster(times=0)[["red"]].rename(red="label")

    built = build_tree({"optical": dated, "label": undated})

    assert "time" not in built.coords
    assert "time" not in built["label"].gs.combine_stac().coords
    assert "time" in built["optical"].gs.combine_stac().coords


def test_group_extraction_preserves_all_root_coordinates():
    raster = build_raster(times=2).isel(time=0)
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords=raster.coords),
            "/image": raster.drop_vars(list(raster.coords)),
        }
    )
    # A data node with nothing between converts to itself: its time stays scalar.
    xr.testing.assert_equal(tree["image"].gs.combine_stac(), raster)
    mapped = tree.map_over_datasets(lambda ds: ds.copy())
    xr.testing.assert_equal(mapped["image"].gs.combine_stac(), raster)


def test_tree_region_combines_valid_footprints_in_different_crs() -> None:
    from odc.geo.geobox import GeoBox

    projected = build_raster()
    geographic = build_raster().pipe(
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
    assert restored.gs.groups == tree.gs.groups
    assert restored.gs.geobox == tree.gs.geobox
    assert restored["optical"].gs.combine_stac().red.dtype == np.dtype("uint16")


def test_every_group_opens_on_its_own(tmp_path: Path, tree: xr.DataTree) -> None:
    store = tree["optical"].gs.combine_stac().gs.to_netcdf(tmp_path / "optical.nc")

    solo = open_raster(store)

    assert store == tmp_path / "optical.nc"
    assert solo.gs.geobox == tree.gs.geobox
    assert "spatial_ref" in solo.coords


def test_a_single_store_is_not_a_tree(tmp_path, tree: xr.DataTree) -> None:
    store = tree["optical"].gs.combine_stac().gs.to_zarr(tmp_path / "optical.zarr")

    with open_raster(store) as opened:
        assert isinstance(opened, xr.Dataset)


def test_a_tree_spans_from_its_earliest_group_to_its_latest() -> None:
    dated = build_raster(times=2)
    timeless = build_raster()

    assert build_tree({"dem": timeless}).gs.timespan is None
    assert (
        build_tree({"optical": dated, "dem": timeless}).gs.timespan == dated.gs.timespan
    )


def test_a_tree_lists_every_instant_any_group_reaches() -> None:
    early = build_raster(times=2)
    late = early.assign_coords(time=early.time + np.timedelta64(1, "D"))

    mixed = build_tree({"optical": early, "dem": build_raster()}).gs.times
    joint = build_tree({"a": early, "b": late}).gs.times

    assert build_tree({"dem": build_raster()}).gs.times is None
    assert mixed is not None and mixed.equals(early.gs.times)
    assert joint is not None
    assert joint.strftime("%Y%m%d").tolist() == ["20250601", "20250602", "20250603"]


def test_a_tree_anchors_on_its_shared_grid_and_joint_span() -> None:
    dated = build_raster(times=2)
    scene = build_tree({"optical": dated, "dem": build_raster()})

    assert scene.gs.region.extent.geom.equals(scene.gs.geobox.extent.geom)
    assert scene.gs.region.timespan == dated.gs.timespan
    assert build_tree({"dem": build_raster()}).gs.region.timespan is None


def test_native_mapping_changes_every_grid_node_and_keeps_the_nodes_above() -> None:
    raster = build_raster()
    coarse = raster[["nir"]].isel(x=slice(0, 1))
    built = _titled(build_tree({"optical": raster[["red"]], "dem": coarse}))
    built["optical"].attrs["title"] = "Optical"

    renamed = built.map_over_datasets(
        lambda node: node.rename({name: f"{name}_raw" for name in node.data_vars}),
    )

    # Grids differ, so the root holds none and each node is mapped on its own.
    assert renamed.gs.geobox is None
    assert renamed.gs.variables == ("optical/r10m/red_raw", "dem/r10m/nir_raw")
    assert renamed.attrs["title"] == "scene"
    assert renamed["optical"].attrs["title"] == "Optical"


def test_a_tree_merges_onto_a_target_rasters_own_grid(tree: xr.DataTree) -> None:
    target = build_raster(crs="EPSG:32748").rename(red="other_red", nir="other_nir")

    merged = tree["optical"].gs.reproject(target.gs.geobox).gs.combine_stac()

    assert merged.gs.geobox == target.gs.geobox


def test_plot_takes_a_caption_without_asking_the_network(monkeypatch) -> None:
    from geosave.geodata.core.region import GeoRegion

    def refuse(_):
        raise AssertionError("an explicit caption must not reverse geocode")

    monkeypatch.setattr(GeoRegion, "locate", refuse)
    image = build_raster()

    drawn = build_tree({"red": image[["red"]], "nir": image[["nir"]]}).gs.plot(
        xlabel=""
    )

    assert len(drawn) == 2


def test_a_tree_converts_each_group_on_its_own_dtype(torch) -> None:
    from tests.conftest import build_tree

    scene = build_tree(
        {"optical": optical(), "height": optical()[["B04"]].astype("float32")}
    )

    tensors = scene.gs.to_tensor()
    assert tuple(tensors) == ("optical/r10m", "height/r10m")
    assert tensors["optical/r10m"].dtype is torch.uint16
    assert tensors["height/r10m"].dtype is torch.float32
    assert scene.gs.to_tensor(dtype="float32")["optical/r10m"].dtype is torch.float32


@pytest.mark.integration
def test_a_tree_round_trips_through_a_upath_carrying_its_options(remote) -> None:
    prefix, options = remote
    folder = storage.location(f"{prefix}/s0", options)
    sample = build_tree({"optical": build_raster(times=2), "label": build_raster()})

    saved = sample.gs.to_zarr(folder / "scene.zarr")

    with open_raster(saved) as opened:
        assert opened.gs.groups == ("optical", "label")


@pytest.mark.parametrize("method", ["to_cog", "to_netcdf", "to_zarr"])
def test_a_deferred_write_computes_to_the_eager_result(tmp_path, method):
    from dask.delayed import Delayed
    from upath import UPath

    from dask.callbacks import Callback

    source = build_raster().chunk({"y": 1, "x": 1})
    tree = xr.DataTree.from_dict({"image": source, "label": source})
    suffix = ".nc" if method == "to_netcdf" else ".zarr"
    tasks = []
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        if method == "to_cog":
            pending = tree.gs.to_cog(tmp_path, compute=False)
        else:
            pending = getattr(tree.gs, method)(
                tmp_path / f"scene{suffix}", compute=False
            )
    assert tasks == []
    assert isinstance(pending, Delayed)
    assert list(tmp_path.iterdir()) == []
    saved = pending.compute()
    if method == "to_cog":
        assert [str(path.relative_to(tmp_path)) for path in saved] == [
            "image.tif",
            "label.tif",
        ]
        assert all(isinstance(path, UPath) for path in saved)
        with open_raster(saved[0]) as restored:
            np.testing.assert_array_equal(
                restored.red, tree["image"].gs.combine_stac().red
            )
    else:
        assert saved == tmp_path / f"scene{suffix}"
        with open_raster(saved) as restored:
            np.testing.assert_array_equal(
                restored["label"].gs.combine_stac().red,
                tree["label"].gs.combine_stac().red,
            )


def build_mixed_collection():
    """A two-date collection: red at 10 m and swir at 20 m, one grid family."""
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataset

    times = np.array(["2024-03-05", "2024-04-05"], "datetime64[ns]")
    fine = GeoBox.from_bbox(
        (500000, 9000000, 500640, 9000640), crs="EPSG:32749", resolution=10
    )
    coarse = fine.zoom_out(2)
    red = np.arange(2 * 64 * 64, dtype="float32").reshape(2, 64, 64)
    swir = np.full((2, 32, 32), 7, "float32")
    return build_tree(
        {
            "s2/r10m": build_dataset(
                {"red": (("time", "y", "x"), red)}, fine, time=times
            ),
            "s2/r20m": build_dataset(
                {"swir": (("time", "y", "x"), swir)}, coarse, time=times
            ),
        }
    )


def test_a_collection_converts_onto_a_stated_resolution():
    tree = build_mixed_collection()
    fine = tree.gs.datasets["s2/r10m"].gs.geobox

    dataset = tree["s2"].gs.reproject(fine).gs.combine_stac()

    assert abs(dataset.odc.geobox.resolution.x) == 10
    assert set(dataset.data_vars) == {"red", "swir"}
    assert dataset.sizes["time"] == 2
    assert float(dataset.swir.isel(time=0).max()) == 7


def test_a_coarser_member_averages_blocks_when_asked():
    tree = build_mixed_collection()
    coarse = tree.gs.datasets["s2/r20m"].gs.geobox

    aligned = tree["s2"].gs.reproject(coarse, resampling="average")
    dataset = aligned.gs.combine_stac()

    assert abs(dataset.odc.geobox.resolution.x) == 20
    red = tree["s2/r10m"].dataset.red.isel(time=0).values
    assert float(dataset.red.isel(time=0, y=0, x=0)) == float(red[:2, :2].mean())


@pytest.mark.parametrize("shift", [0, 5])
def test_different_grids_require_alignment_without_reading_pixels(shift):
    from dask.callbacks import Callback

    from geosave.geodata.errors import RasterValidationError

    tree = build_mixed_collection().chunk()
    coarse = tree.gs.datasets["s2/r20m"]
    shifted = coarse.assign_coords(x=coarse.x + shift)
    broken = xr.DataTree.from_dict(
        {"s2/r10m": tree.gs.datasets["s2/r10m"], "s2/r20m": shifted}
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        with pytest.raises(RasterValidationError, match="reproject") as caught:
            broken["s2"].gs.combine_stac()
    assert caught.value.path == "r20m"
    assert tasks == []


def test_a_collections_metadata_reaches_its_dataset():
    tree = build_mixed_collection()
    tree["s2"].attrs["title"] = "Sentinel-2"
    fine = tree.gs.datasets["s2/r10m"].gs.geobox

    dataset = tree["s2"].gs.reproject(fine).gs.combine_stac()

    assert dataset.attrs["title"] == "Sentinel-2"


def test_nested_tree_zarr_preserves_parent_data_and_organization(nested_tree, tmp_path):
    saved = nested_tree.gs.to_zarr(tmp_path / "nested.zarr")
    with open_raster(saved) as restored:
        assert set(nested_tree.gs.datasets) == set(restored.gs.datasets)
        for path in nested_tree.gs.datasets:
            source = nested_tree[path].to_dataset(inherit="all_coords")
            loaded = restored[path].to_dataset(inherit="all_coords")
            assert loaded.gs.geobox == source.gs.geobox
            assert set(loaded.data_vars) == set(source.data_vars)
            for name in source.data_vars:
                np.testing.assert_array_equal(loaded[name].values, source[name].values)
        assert restored["s2/measurements"].attrs == nested_tree["s2/measurements"].attrs


def test_nodes_include_inherited_time_coordinates():
    source = build_raster(times=2)
    tree = xr.DataTree.from_dict(
        {
            "s2": xr.Dataset(coords={"time": source.time}),
            "s2/nested/native": source.drop_vars("time"),
        }
    )
    assert list(tree.gs.datasets["s2/nested/native"].data_vars) == ["red", "nir"]
    np.testing.assert_array_equal(
        tree.gs.datasets["s2/nested/native"].time, source.time
    )


def test_grid_nodes_have_no_collection_hierarchy_requirement():
    from dask.callbacks import Callback

    source = build_raster(times=2).chunk()
    tree = xr.DataTree.from_dict({"/": source, "qa": source.rename(red="quality")})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        nodes = tree.gs.datasets
    assert tasks == []
    assert set(nodes) == {".", "qa"}
    assert list(nodes["."].data_vars) == ["red", "nir"]
    assert nodes["qa"].gs.geobox == source.gs.geobox


def test_combine_reports_missing_rasters():
    with pytest.raises(ValueError, match="no raster variables"):
        xr.DataTree().gs.combine_stac()


def build_stac_tree():
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
    tree = build_stac_tree().filter(lambda node: "B11" not in node.data_vars)

    dataset = tree["s2"].gs.combine_stac()

    assert dataset.sizes["time"] == 2
    assert list(dataset.data_vars) == ["B04"]
    np.testing.assert_array_equal(dataset.B04.isel(y=0, x=0).values, [1.0, 2.0])
    assert dataset.attrs["title"] == "s2"


def test_an_item_merges_its_assets_and_keeps_its_time_scalar():
    tree = build_stac_tree()
    grid = tree["s2/2025-06-01/B04"].gs.geobox

    dataset = tree.gs.reproject(grid)["s2/2025-06-01"].gs.combine_stac()

    assert set(dataset.data_vars) == {"B04", "B11"}
    assert dataset.time.ndim == 0
    assert dataset.gs.geobox == grid


def test_collections_keep_independent_timelines():
    tree = build_stac_tree()

    with pytest.raises(ValueError, match="independent timelines"):
        tree.gs.reproject(tree["s2/2025-06-01/B04"].gs.geobox).gs.combine_stac()


def test_a_collection_refuses_an_item_missing_a_band():
    tree = build_stac_tree()
    moved = tree.gs.reproject(tree["s2/2025-06-01/B04"].gs.geobox)
    gappy = moved.filter(lambda node: node.path != "/s2/2025-06-06/B11")

    with pytest.raises(ValueError, match="same variables") as caught:
        gappy["s2"].gs.combine_stac()
    assert caught.value.path == "2025-06-06"


def test_a_timeless_item_combines_without_time():
    from tests.conftest import build_dataset
    from tests.geodata.conftest import geobox as make_geobox

    box = make_geobox(shape=(4, 4))
    height = build_dataset({"height": (("y", "x"), np.zeros((4, 4)))}, box)
    one = xr.DataTree.from_dict({"dem/cop/height": height})

    assert one["dem/cop"].gs.combine_stac().height.dims == ("y", "x")


def test_differing_grids_with_nothing_stated_name_the_grids():
    from geosave.geodata.errors import RasterValidationError

    with pytest.raises(RasterValidationError, match="2 grids.*reproject"):
        build_stac_tree()["s2"].gs.combine_stac()


def test_a_resolution_lands_every_node_on_one_grid_over_their_extent():
    tree = build_stac_tree()
    fine = tree["s2/2025-06-01/B04"].gs.geobox

    dataset = tree["s2"].gs.reproject(fine).gs.combine_stac()

    assert dataset.gs.geobox == fine
    assert set(dataset.data_vars) == {"B04", "B11"}
    assert dataset.sizes["time"] == 2


def test_reproject_moves_every_leaf_lazily():
    tree = build_stac_tree().chunk()
    grid = tree["s2/2025-06-01/B11"].gs.geobox

    with dask.config.set(scheduler=_refuse):
        moved = tree.gs.reproject(grid)

    for leaf in moved.leaves:
        assert leaf.to_dataset(inherit="all_coords").gs.geobox == grid
    assert moved["s2"].attrs == {"title": "s2"}


def test_a_reprojected_tree_chips_every_leaf_by_one_window():
    tree = build_stac_tree()
    moved = tree.gs.reproject(tree["s2/2025-06-01/B04"].gs.geobox)

    chip = moved.isel(y=slice(0, 2), x=slice(0, 2))

    for leaf in chip.leaves:
        assert leaf.to_dataset(inherit="all_coords")[leaf.name].shape == (2, 2)
    times = chip["s2/2025-06-06"].to_dataset(inherit=False).time.values
    assert times == np.datetime64("2025-06-06")


def test_combine_after_reproject_stacks_the_collection_lazily():
    tree = build_stac_tree().chunk()
    grid = tree["s2/2025-06-01/B04"].gs.geobox

    with dask.config.set(scheduler=_refuse):
        dataset = tree.gs.reproject(grid)["s2"].gs.combine_stac()

    assert dataset.gs.geobox == grid
    assert set(dataset.data_vars) == {"B04", "B11"}
    assert dataset.sizes["time"] == 2


def test_same_grid_names_nodes_on_another_grid():
    from geosave.geodata.errors import RasterValidationError

    with pytest.raises(RasterValidationError, match="grid") as caught:
        build_stac_tree()["s2"].gs.validate(same_grid=True)
    assert "B11" in caught.value.path


def test_combine_matches_the_catalog_loader_on_a_fresh_catalog(rows, grid):
    tree = rows.gs.to_datatree().gs.reproject(grid)
    optical = rows[rows["collection"] == "optical"]

    combined = tree["optical"].gs.combine_stac()
    loaded = optical.gs.to_dataset(grid)

    for name in ("red", "nir"):
        expected = loaded[name].transpose(*combined[name].dims)
        np.testing.assert_array_equal(combined[name].values, expected.values)


def test_to_cog_writes_one_file_per_asset_leaf(tmp_path):
    from tests.geodata.conftest import build_raster

    scene = build_raster(times=2)

    files = scene.gs.expand_stac().gs.to_cog(tmp_path)

    days = ["20250601T000000", "20250602T000000"]
    expected = []
    for day in days:
        for name in ("red", "nir"):
            expected.append(tmp_path / day / f"{name}.tif")
    assert [Path(f) for f in files] == expected
    with open_raster(files[2]) as restored:
        np.testing.assert_array_equal(restored.red.squeeze(), scene.red.isel(time=1))


def test_to_cog_refuses_existing_files_before_writing(tmp_path):
    from tests.geodata.conftest import build_raster

    tree = build_raster(times=2).gs.expand_stac(asset="image")
    first, second = tree.gs.to_cog(tmp_path)
    second.unlink()

    with pytest.raises(FileExistsError):
        tree.gs.to_cog(tmp_path)
    assert not second.exists()


def test_to_cog_defers_the_whole_batch(tmp_path):
    from tests.geodata.conftest import build_raster

    tree = build_raster(times=2).chunk().gs.expand_stac(asset="image")

    with dask.config.set(scheduler=_refuse):
        pending = tree.gs.to_cog(tmp_path, compute=False)
    assert not list(tmp_path.iterdir())
    assert len(pending.compute()) == 2


@pytest.fixture
def city():
    return Geometry(from_wkt("POLYGON ((13 52, 14 52, 13 53, 13 52))"), "EPSG:4326")


def test_extent_needs_no_grid_and_tree_nodes_stand_alone(city):
    raster = xr.Dataset({"red": (("y", "x"), np.zeros((2, 2)))})
    tree = xr.DataTree.from_dict({"a": raster, "b/deep": raster})
    written = tree.gs.write_extent(city)

    assert written.gs.region.extent == city
    assert written["b/deep"].to_dataset().red.gs.region.extent == city
    assert raster.attrs == {}
    assert written["a"].gs.geobox is None


def test_dataset_and_tree_union_variable_extents(city):
    left = build_raster()[["red"]].gs.write_extent(city)
    other = Geometry(from_wkt("POLYGON ((15 52, 16 52, 15 53, 15 52))"), "EPSG:4326")
    right = build_raster()[["nir"]].gs.write_extent(other.to_crs("EPSG:3857"))
    dataset = xr.Dataset({"red": left.red, "nir": right.nir})
    tree = xr.DataTree.from_dict({"left": left, "right": right})
    expected = city.union(other)
    assert dataset.gs.region.extent.geom.equals_exact(expected.geom, 1e-8)
    assert tree.gs.region.extent.geom.equals_exact(expected.geom, 1e-8)
