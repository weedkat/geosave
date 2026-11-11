from __future__ import annotations

from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from geosave.geodata import open_raster
from tests.conftest import build_tree
from geosave.geodata.io import rasterio as gdal
from geosave.geodata.io import storage
from tests.geodata.conftest import (
    build_raster,
    item_table,
    write_collection_cogs,
)
from tests.geodata.conftest import titled as _titled
from tests.geodata.core.raster.conftest import optical


def test_shared_time_requires_siblings_to_inherit_one_timeline():
    from geosave.geodata.errors import RasterValidationError

    scene = build_raster(times=2)
    siblings = xr.DataTree.from_dict({"s2/fine": scene, "s2/coarse": scene})
    with pytest.raises(RasterValidationError, match="own time"):
        siblings["s2"].gs.validate(shared_time=True)

    # Defined once on the parent, the time is inherited and agrees by construction.
    parent = xr.Dataset(coords={"time": scene.time})
    shared = xr.DataTree.from_dict(
        {
            "s2": parent,
            "s2/fine": scene.drop_vars("time"),
            "s2/coarse": scene.drop_vars("time"),
        }
    )
    shared["s2"].gs.validate(shared_time=True)
    with pytest.raises(ValueError, match="not aligned"):
        xr.DataTree.from_dict(
            {
                "s2": parent,
                "s2/fine": scene.assign_coords(
                    time=scene.time + np.timedelta64(1, "D")
                ),
            }
        )
    # A lone data node may carry its own time.
    xr.DataTree.from_dict({"s2/fine": scene})["s2"].gs.validate(shared_time=True)


def test_same_footprint_accepts_nested_grids_and_refuses_other_ground():
    from geosave.geodata.errors import RasterValidationError

    fine = build_raster()
    coarse_grid = fine.gs.geobox.zoom_out(2)
    coarse = xr.Dataset(
        {"swir": (("y", "x"), np.ones(coarse_grid.shape, "uint16"))}
    ).gs.write_grid(coarse_grid)
    xr.DataTree.from_dict({"fine": fine, "coarse": coarse}).gs.validate(
        same_footprint=True
    )

    # One pixel of edge difference is snapping; half a scene is other ground.
    wide_grid = fine.gs.geobox.zoom_to(shape=(10, 10))
    wide = xr.Dataset(
        {"red": (("y", "x"), np.ones(wide_grid.shape, "uint16"))}
    ).gs.write_grid(wide_grid)
    tree = xr.DataTree.from_dict({"wide": wide, "half": wide.isel(x=slice(0, 5))})
    with pytest.raises(RasterValidationError, match="different ground"):
        tree.gs.validate(same_footprint=True)


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


@pytest.mark.parametrize("time", [np.datetime64("NaT", "ns"), "not-a-date"])
def test_invalid_group_dates_are_rejected_before_any_cog_is_written(tmp_path, time):
    first = build_raster(times=1)
    second = build_raster(times=1).assign_coords(time=[time])
    tree = xr.DataTree.from_dict({"first": first, "second": second})
    with pytest.raises(ValueError):
        tree.gs.to_cog(tmp_path, split_asset=(), format_paths="{asset}.tif")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("compute", [True, False])
def test_a_later_invalid_cog_group_is_rejected_before_any_write(tmp_path, compute):
    valid = build_raster()
    invalid = build_raster().gs.write_nodata(0, target="red")
    tree = xr.DataTree.from_dict({"first": valid, "second": invalid})
    with pytest.raises(ValueError, match="fill value"):
        tree.gs.to_cog(
            tmp_path,
            split_asset=(),
            format_paths="{asset}.tif",
            compute=compute,
        )
    assert not list(tmp_path.iterdir())


def test_group_destinations_are_checked_together_before_writing(tmp_path):
    tree = xr.DataTree.from_dict({"first": build_raster(), "second": build_raster()})
    with pytest.raises(ValueError, match="collid"):
        tree.gs.to_cog(
            tmp_path,
            format_paths="same.tif",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("times", "split_vars", "template", "files"),
    [
        (0, False, "{asset}.tif", 1),
        (0, True, "{asset}.tif", 2),
        (2, False, "{asset}_{time:%Y%m%d}.tif", 2),
        (2, True, "{asset}_{time:%Y%m%d}.tif", 4),
    ],
)
def test_cog_export_retains_groups_with_repeated_band_names(
    tmp_path: Path, times: int, split_vars: bool, template: str, files: int
) -> None:
    image = build_raster(times=times)
    label = (image // 1000).astype("uint8")
    # Time the nodes share sits once on their parent, which they inherit.
    root = xr.Dataset(coords={"time": image.time}) if times else xr.Dataset()
    shared = [name for name in ("time",) if times]
    written = xr.DataTree.from_dict(
        {
            "/": root,
            "image.v2": image.drop_vars(shared),
            "label.v1": label.drop_vars(shared),
        }
    )

    saved = written.gs.to_cog(
        tmp_path, split_asset=("variable",) if split_vars else (), format_paths=template
    )

    assert len(saved) == 2 * files
    for group in ("image.v2", "label.v1"):
        paths = [
            Path(f)
            for f in saved
            if str(Path(f).relative_to(tmp_path)).startswith(group)
        ]
        assert len(paths) == files
        source = written[group].gs.to_dataset()
        for path in paths:
            with gdal.open_raster(path) as restored:
                expected = source.sel(time=restored.time) if times else source
                assert restored.gs.geobox == source.gs.geobox
                for name in restored.data_vars:
                    assert restored[name].dtype == source[name].dtype
                    np.testing.assert_array_equal(restored[name], expected[name])


def test_groups_sharing_a_path_refuse_before_writing(tmp_path: Path) -> None:
    image = build_raster()
    written = xr.DataTree.from_dict({"image": image, "label": image})

    with pytest.raises(ValueError, match=r"collide.*scene\.tif"):
        written.gs.to_cog(tmp_path, format_paths="scene.tif")
    assert not list(tmp_path.iterdir())


def test_tree_publication_rejects_nodes_with_different_observations(tmp_path):
    tree = xr.DataTree.from_dict(
        {
            "optical": build_raster(times=2),
            "dem": build_raster(),
        }
    )

    # The dated node keeps its own time, so the nodes share no timeline.
    with pytest.raises(ValueError, match="own time"):
        tree.gs.to_item(tmp_path)
    assert not list(tmp_path.iterdir())


def test_one_groups_scalar_time_does_not_date_the_others() -> None:
    dated = build_raster(times=1).isel(time=0)
    undated = build_raster(times=0)[["red"]].rename(red="label")

    built = build_tree({"optical": dated, "label": undated})

    assert "time" not in built.coords
    assert "time" not in built["label"].gs.to_dataset().coords
    assert "time" in built["optical"].gs.to_dataset().coords


def test_group_extraction_preserves_all_root_coordinates():
    raster = build_raster(times=2).isel(time=0)
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords=raster.coords),
            "/image": raster.drop_vars(list(raster.coords)),
        }
    )
    xr.testing.assert_equal(tree["image"].gs.to_dataset(), raster)
    mapped = tree.map_over_datasets(lambda ds: ds.copy())
    xr.testing.assert_equal(mapped["image"].gs.to_dataset(), raster)


def test_tree_region_combines_valid_footprints_in_different_crs() -> None:
    from odc.geo.geobox import GeoBox

    projected = build_raster()
    geographic = build_raster().gs.write_grid(
        GeoBox.from_bbox((111, -9, 111.01, -8.99), crs="EPSG:4326", shape=(2, 2))
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
    assert restored["optical"].gs.to_dataset().red.dtype == np.dtype("uint16")


def test_every_group_opens_on_its_own(tmp_path: Path, tree: xr.DataTree) -> None:
    store = tree["optical"].gs.to_dataset().gs.to_netcdf(tmp_path / "optical.nc")

    solo = open_raster(store)

    assert store == tmp_path / "optical.nc"
    assert solo.gs.geobox == tree.gs.geobox
    assert "spatial_ref" in solo.coords


def test_a_deferred_tree_write_resolves_to_its_stores(
    tmp_path, tree: xr.DataTree
) -> None:
    pending = tree.chunk().gs.to_zarr(tmp_path / "later.zarr", compute=False)

    assert not (tmp_path / "later.zarr").exists()
    (store,) = dask.compute(pending)

    assert store == tmp_path / "later.zarr"
    np.testing.assert_array_equal(
        open_raster(store)["optical"].gs.to_dataset().red,
        tree["optical"].gs.to_dataset().red,
    )


def test_a_single_store_is_not_a_tree(tmp_path, tree: xr.DataTree) -> None:
    store = tree["optical"].gs.to_dataset().gs.to_zarr(tmp_path / "optical.zarr")

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


def test_a_tree_reads_the_pixel_operations_through_gs(tree: xr.DataTree) -> None:
    import shapely

    left, bottom, right, top = tree.gs.bounds.bbox
    column = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
        crs=tree.gs.crs,
    )

    assert tree.gs.mask_and_scale().gs.groups == tree.gs.groups
    # One mask fits one grid, so a tree masks through its nodes, not as a whole.
    assert not hasattr(tree.gs, "mask")
    assert tree.gs.crop(column, mask=False).gs.geobox.shape == (2, 1)


def test_a_tree_merges_onto_a_target_rasters_own_grid(tree: xr.DataTree) -> None:
    target = build_raster(crs="EPSG:32748").rename(red="other_red", nir="other_nir")

    merged = tree["optical"].gs.to_dataset(grid=target.gs.geobox)

    assert merged.gs.geobox == target.gs.geobox
    assert not hasattr(tree.gs, "reproject")


def test_a_saved_tree_indexes_one_item_per_group_scene(tmp_path: Path) -> None:
    optical = build_raster(times=2)
    sample = build_tree({"optical": optical, "dem": build_raster()})

    saved = write_collection_cogs(
        sample,
        tmp_path,
        {"optical": "optical_{time:%Y%m%dT%H%M%S}.tif", "dem": "dem.tif"},
    )
    rows = item_table(saved, prefix="s0")

    assert rows["id"].tolist() == [
        "s0/optical_20250601T000000",
        "s0/optical_20250602T000000",
        "s0/dem",
    ]
    assert rows["collection"].tolist() == ["optical", "optical", "dem"]
    assert rows.iloc[2]["assets"]["image"]["href"] == str(tmp_path / "dem.tif")


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


def test_a_tree_keys_written_paths_by_node(tmp_path):
    image = build_raster(times=1)
    label = image[["red"]].rename(red="label")
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"time": image.time}),
            "optical": image.drop_vars("time"),
            "label": label.drop_vars("time"),
        }
    )

    files = tree.gs.to_cog(
        tmp_path,
        split_asset=(),
        format_paths="{asset}_{time:%Y%m%d}.tif",
    )

    assert [path.name for path in files] == [
        "optical_20250601.tif",
        "label_20250601.tif",
    ]


@pytest.mark.parametrize("method", ["to_cog", "to_netcdf"])
def test_a_deferred_write_computes_to_the_eager_result(tmp_path, method):
    from dask.delayed import Delayed
    from upath import UPath

    tree = xr.DataTree.from_dict({"image": build_raster(), "label": build_raster()})
    if method == "to_cog":
        pending = tree.gs.to_cog(
            tmp_path,
            format_paths="{asset}.tif",
            compute=False,
        )
    else:
        pending = tree.gs.to_netcdf(tmp_path / "scene.nc", compute=False)
    assert isinstance(pending, Delayed)
    assert list(tmp_path.iterdir()) == []
    saved = pending.compute()
    if method == "to_cog":
        assert [str(path.relative_to(tmp_path)) for path in saved] == [
            "image/red.tif",
            "image/nir.tif",
            "label/red.tif",
            "label/nir.tif",
        ]
        assert all(isinstance(path, UPath) for path in saved)
        with open_raster(saved[0]) as restored:
            np.testing.assert_array_equal(
                restored.red, tree["image"].gs.to_dataset().red
            )
    else:
        assert saved == tmp_path / "scene.nc"
        with open_raster(saved) as restored:
            np.testing.assert_array_equal(
                restored["label"].gs.to_dataset().red, tree["label"].gs.to_dataset().red
            )


@pytest.mark.parametrize(
    ("method", "backend"),
    [
        ("to_cog", "geotiff"),
        ("to_zarr", "zarr"),
        ("to_netcdf", "netcdf"),
    ],
)
def test_deferred_writes_reach_formats_with_lazy_pixels(
    tmp_path, monkeypatch, method, backend
):
    from geosave.geodata.io.formats import geotiff
    from geosave.geodata.io.formats import netcdf
    from geosave.geodata.io.formats import zarr

    source = build_raster().chunk({"y": 1, "x": 1})
    tree = xr.DataTree.from_dict({"image": source, "label": source})
    module = {"geotiff": geotiff, "zarr": zarr, "netcdf": netcdf}[backend]
    # The step that runs when the delayed result computes, and receives the pixels.
    name = "write_cog" if backend == "geotiff" else "_write"
    write = getattr(module, name)
    chunks = []

    def save(raster, destination, *args, **options):
        rasters = (
            raster.gs.datasets.values() if isinstance(raster, xr.DataTree) else [raster]
        )
        for each in rasters:
            chunks.append(each.red.chunks)
        return write(raster, destination, *args, **options)

    monkeypatch.setattr(module, name, save)
    if method == "to_cog":
        pending = tree.gs.to_cog(
            tmp_path,
            split_asset=(),
            format_paths="{asset}.tif",
            compute=False,
        )
    else:
        suffix = ".zarr" if method == "to_zarr" else ".nc"
        pending = getattr(tree.gs, method)(tmp_path / f"scene{suffix}", compute=False)
    assert chunks == []
    pending.compute()
    assert chunks == [source.red.chunks, source.red.chunks]
    if method == "to_cog":
        with open_raster(tmp_path / "image.tif") as restored:
            np.testing.assert_array_equal(restored.red, source.red)
    else:
        with open_raster(tmp_path / f"scene{suffix}") as restored:
            np.testing.assert_array_equal(
                restored["image"].gs.to_dataset().red, source.red
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


def test_a_collection_converts_after_explicit_fine_grid_alignment():
    tree = build_mixed_collection()

    dataset = tree["s2"].gs.to_dataset(grid="r10m")

    assert abs(dataset.odc.geobox.resolution.x) == 10
    assert set(dataset.data_vars) == {"red", "swir"}
    assert dataset.sizes["time"] == 2
    assert float(dataset.swir.isel(time=0).max()) == 7


def test_a_coarser_member_averages_blocks_when_asked():
    tree = build_mixed_collection()

    dataset = tree["s2"].gs.to_dataset(grid="r20m", resampling="average")

    assert abs(dataset.odc.geobox.resolution.x) == 20
    red = tree["s2/r10m"].dataset.red.isel(time=0).values
    assert float(dataset.red.isel(time=0, y=0, x=0)) == float(red[:2, :2].mean())


def test_organizational_nesting_merges_on_a_relative_grid_path():
    tree = build_mixed_collection()
    both = build_tree(
        {
            **tree.gs.datasets,
            "dem/r10m": tree.gs.datasets["s2/r10m"].rename(red="height"),
        }
    )

    dataset = both.gs.to_dataset(grid="s2/r20m")

    assert set(dataset.data_vars) == {"red", "swir", "height"}
    assert dataset.odc.geobox == both["s2/r20m"].gs.geobox


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
        with pytest.raises(RasterValidationError, match="grid=") as caught:
            broken["s2"].gs.to_dataset()
    assert caught.value.path == "r20m"
    assert tasks == []


def test_a_collections_metadata_reaches_its_dataset():
    tree = build_mixed_collection()
    tree["s2"].attrs["title"] = "Sentinel-2"

    dataset = tree["s2"].gs.to_dataset(grid="r10m")

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


def test_to_dataset_includes_lazy_root_and_deeply_nested_rasters():
    raster = build_mixed_collection().gs.datasets["s2/r10m"].chunk({"y": 16, "x": 16})
    tree = xr.DataTree.from_dict(
        {"/": raster, "organization/sensor/detail": raster.rename(red="height")}
    )

    from dask.callbacks import Callback

    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        dataset = tree.gs.to_dataset()

    assert tasks == []
    assert set(dataset.data_vars) == {"red", "height"}
    assert dataset.red.chunks is not None
    assert dataset.height.chunks is not None
    xr.testing.assert_equal(dataset.red, raster.red)
    xr.testing.assert_equal(dataset.height, raster.red.rename("height"))


@pytest.mark.parametrize(
    ("tree", "message"),
    [
        (xr.DataTree(), "no raster variables"),
        (
            xr.DataTree(xr.Dataset({"red": (("y", "x"), np.ones((2, 2)))})),
            "write_grid",
        ),
    ],
)
def test_to_dataset_reports_missing_rasters_or_grids(tree, message):
    with pytest.raises(ValueError, match=message):
        tree.gs.to_dataset()


def test_group_grid_merges_siblings_on_one_grid_and_keeps_others_apart():
    from tests.conftest import build_dataset
    from tests.geodata.conftest import geobox as make_geobox

    fine = make_geobox(shape=(4, 4))
    coarse = fine.zoom_out(2)
    tree = xr.DataTree.from_dict(
        {
            "/s2": xr.Dataset(attrs={"title": "s2"}),
            "/s2/B04": build_dataset({"B04": (("y", "x"), np.ones((4, 4)))}, fine),
            "/s2/B08": build_dataset({"B08": (("y", "x"), np.ones((4, 4)))}, fine),
            "/s2/B11": build_dataset({"B11": (("y", "x"), np.ones((2, 2)))}, coarse),
        }
    )

    grouped = tree.gs.group_grid()

    assert set(grouped.gs.datasets) == {"s2/r10m", "s2/r20m"}
    assert list(grouped["s2/r10m"].data_vars) == ["B04", "B08"]
    assert grouped["s2/r10m"].gs.geobox == fine
    assert grouped["s2"].attrs == {"title": "s2"}


def test_group_grid_refuses_two_grids_with_one_pixel_size():
    from tests.conftest import build_dataset
    from tests.geodata.conftest import geobox as make_geobox

    fine = make_geobox(shape=(4, 4))
    shifted = make_geobox(shape=(2, 2))
    tree = xr.DataTree.from_dict(
        {
            "/s2/B04": build_dataset({"B04": (("y", "x"), np.ones((4, 4)))}, fine),
            "/s2/B05": build_dataset({"B05": (("y", "x"), np.ones((2, 2)))}, shifted),
        }
    )

    with pytest.raises(ValueError, match="r10m"):
        tree.gs.group_grid()
