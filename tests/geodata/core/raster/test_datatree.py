from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr
from upath import UPath

from geosave.geodata import open_raster, open_tree
from geosave.geodata.core.raster.datatree import map_groups
from geosave.geodata.core.raster.create import tree as build_tree
from geosave.geodata.io import gdal
from tests.geodata.conftest import build_raster, item_table
from tests.geodata.conftest import titled as _titled
from tests.geodata.core.raster.conftest import optical
from tests.geodata.conftest import flat


@pytest.mark.parametrize(
    ("times", "template", "files"),
    [
        (0, "{collection}.tif", 1),
        (0, "{collection}/{band}.tif", 2),
        (2, "{collection}_{time:%Y%m%d}.tif", 2),
        (2, "{collection}/{time:%Y%m%d}/{band}.tif", 4),
    ],
)
def test_cog_export_retains_groups_with_repeated_band_names(
    tmp_path: Path, times: int, template: str, files: int
) -> None:
    image = build_raster(times=times)
    label = (image // 1000).astype("uint8")
    written = build_tree({"image.v2": image, "label.v1": label})

    scenes = written.gs.to_cog(f"{tmp_path}/{template}")

    layers = [scene.collection for scene in scenes for _ in scene.written]
    saved = flat(scenes)
    assert list(dict.fromkeys(layers)) == ["image.v2", "label.v1"]
    for group in ("image.v2", "label.v1"):
        paths = [Path(f) for f, layer in zip(saved, layers) if layer == group]
        assert len(paths) == files
        source = written.gs.rasters[group]
        for path in paths:
            assert group in str(path.relative_to(tmp_path))
            with gdal.open_raster(path) as restored:
                expected = source.sel(time=restored.time) if times else source
                assert restored.gs.geobox == source.gs.geobox
                for name in restored.data_vars:
                    assert restored[name].dtype == source[name].dtype
                    np.testing.assert_array_equal(restored[name], expected[name])


def test_groups_sharing_a_path_refuse_before_writing(tmp_path: Path) -> None:
    image = build_raster()
    written = build_tree({"image": image, "label": image})

    with pytest.raises(
        ValueError, match=r"2 layers would share .*scene\.tif; add \{collection\}"
    ):
        flat(written.gs.to_cog(f"{tmp_path}/scene.tif"))
    assert not list(tmp_path.iterdir())


def test_groups_take_a_template_each(tmp_path: Path) -> None:
    written = build_tree({"optical": build_raster(times=2), "dem": build_raster()})

    scenes = written.gs.to_cog(
        {
            "optical": f"{tmp_path}/optical_{{time:%Y%m%d}}.tif",
            "dem": f"{tmp_path}/dem.tif",
        }
    )

    assert [(s.collection, Path(f)) for s in scenes for f in s.written.values()] == [
        ("optical", tmp_path / "optical_20250601.tif"),
        ("optical", tmp_path / "optical_20250602.tif"),
        ("dem", tmp_path / "dem.tif"),
    ]


def test_time_on_an_undated_group_points_to_one_template_per_group(
    tmp_path: Path,
) -> None:
    written = build_tree({"optical": build_raster(times=2), "dem": build_raster()})

    with pytest.raises(ValueError, match="'dem' has no dates.*a template per group"):
        flat(written.gs.to_cog(f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif"))


def test_a_group_with_a_scalar_time_fills_the_time_field(tmp_path: Path) -> None:
    scene = build_raster(times=1).isel(time=0)
    written = build_tree({"optical": scene})

    saved = flat(written.gs.to_cog(f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif"))

    assert [f.name for f in saved] == ["optical_20250601.tif"]


def test_one_groups_scalar_time_does_not_date_the_others() -> None:
    dated = build_raster(times=1).isel(time=0)
    undated = build_raster(times=0)[["red"]].rename(red="label")

    built = build_tree({"optical": dated, "label": undated})

    assert "time" not in built.coords
    assert "time" not in built["label"].coords
    assert "time" in built["optical"].coords


def test_a_tree_reads_back_from_what_its_cog_export_returns(tmp_path: Path) -> None:
    image = build_raster(times=2)
    written = build_tree({"image": image, "label": (image // 1000).astype("uint8")})

    restored = open_tree(
        written.gs.to_cog(f"{tmp_path}/{{collection}}/{{time:%Y%m%d}}.tif")
    )

    assert restored.gs.groups == ("image", "label")
    np.testing.assert_array_equal(
        restored.gs.rasters["label"].red, written.gs.rasters["label"].red
    )


def test_group_extraction_preserves_all_root_coordinates():
    raster = build_raster(times=2).isel(time=0)
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords=raster.coords),
            "/image": raster.drop_vars(list(raster.coords)),
        }
    )
    xr.testing.assert_equal(tree.gs.rasters["image"], raster)
    mapped = map_groups(tree, lambda ds: ds.copy())
    xr.testing.assert_equal(mapped.gs.rasters["image"], raster)


def test_tree_carries_the_shared_grid_at_its_root() -> None:
    raster = build_raster()

    built = build_tree({"optical": raster[["red"]], "dem": raster[["nir"]]})

    assert built.gs.groups == ("optical", "dem")
    assert built.gs.geobox == raster.gs.geobox
    expected = ("y", "x", "spatial_ref")
    assert set(expected) <= set(built.dataset.coords)
    assert not built.dataset.data_vars


def test_tree_names_are_not_restricted_to_identifiers() -> None:
    raster = build_raster()

    built = build_tree({"sentinel-2-l2a": raster[["red"]], "ndvi.v2": raster[["nir"]]})

    assert built.gs.groups == ("sentinel-2-l2a", "ndvi.v2")


def test_tree_refuses_an_empty_mapping() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_tree({})


def test_tree_refuses_a_name_spelling_a_path() -> None:
    raster = build_raster()

    with pytest.raises(ValueError, match="spell paths rather than group names"):
        build_tree(
            {"sentinel-2/r10": raster[["red"]], "sentinel-2/r20": raster[["nir"]]}
        )


def test_groups_covering_different_ground_leave_the_root_bare() -> None:
    whole = build_raster()
    corner = build_raster().isel(y=slice(0, 1), x=slice(0, 1))

    loose = build_tree({"whole": whole, "corner": corner})

    assert loose.gs.groups == ("whole", "corner")
    assert loose.gs.geobox is None
    assert loose.gs.rasters["corner"].gs.geobox == corner.gs.geobox


def test_groups_in_different_crs_leave_the_root_bare() -> None:
    projected = build_raster(crs="EPSG:32749")
    geographic = build_raster(crs="EPSG:4326")

    loose = build_tree({"projected": projected, "geographic": geographic})

    assert loose.gs.geobox is None
    assert loose.gs.rasters["geographic"].gs.crs.epsg == 4326


def test_an_unplaced_raster_leaves_the_root_bare() -> None:
    unplaced = xr.Dataset({"red": (("y", "x"), np.zeros((2, 2), "uint16"))})

    loose = build_tree({"unplaced": unplaced})

    assert loose.gs.groups == ("unplaced",)
    assert loose.gs.geobox is None


def test_a_tree_without_a_shared_grid_names_no_anchor() -> None:
    loose = build_tree(
        {"whole": build_raster(), "geographic": build_raster(crs="EPSG:4326")}
    )

    with pytest.raises(ValueError, match="publishes no grid at its root"):
        _ = loose.gs.anchor


def test_xarray_refuses_a_misaligned_group_after_construction(
    tree: xr.DataTree,
) -> None:
    shifted = build_raster()
    shifted = shifted.assign_coords(y=shifted.y.values + 12_345.0)

    with pytest.raises(ValueError, match="not aligned"):
        tree["shifted"] = xr.DataTree(shifted)


def test_tree_round_trip_preserves_groups_and_grid(
    tmp_path: Path, tree: xr.DataTree
) -> None:
    items = tree.gs.to_zarr(tmp_path / "tree")

    restored = open_tree(items)

    assert [file.path for file in flat(items)] == [
        tmp_path / "tree" / f"{group}.zarr" for group in tree.gs.groups
    ]
    assert restored.gs.groups == tree.gs.groups
    assert restored.gs.geobox == tree.gs.geobox
    assert restored["optical"].dataset.red.dtype == np.dtype("uint16")


def test_each_group_is_a_scene_naming_its_collection(
    tmp_path, tree: xr.DataTree
) -> None:
    scenes = tree.gs.to_zarr(tmp_path / "tree")

    assert [scene.collection for scene in scenes] == list(tree.gs.groups)
    rows = item_table(scenes, datetime=datetime(2025, 6, 1, tzinfo=UTC))
    assert rows["collection"].tolist() == list(tree.gs.groups)


def test_every_group_opens_on_its_own(tmp_path: Path, tree: xr.DataTree) -> None:
    store = flat(tree.gs.to_netcdf(tmp_path / "solo"))[0]

    solo = open_raster(store)

    assert store.path == tmp_path / "solo/optical.nc"
    assert solo.gs.geobox == tree.gs.geobox
    assert "spatial_ref" in solo.coords


def test_a_saved_group_is_not_replaced_unasked(tmp_path, tree: xr.DataTree) -> None:
    tree.gs.to_zarr(tmp_path / "tree")

    with pytest.raises(FileExistsError):
        tree.gs.to_zarr(tmp_path / "tree")

    assert len(tree.gs.to_zarr(tmp_path / "tree", overwrite=True)) == len(
        tree.gs.groups
    )


def test_a_deferred_tree_write_resolves_to_its_stores(
    tmp_path, tree: xr.DataTree
) -> None:
    pending = tree.chunk().gs.to_zarr(tmp_path / "later", compute=False)

    assert not (tmp_path / "later").exists()
    (scenes,) = dask.compute(pending)

    assert all(scene.pending is None for scene in scenes)
    np.testing.assert_array_equal(open_tree(scenes)["optical"].red, tree["optical"].red)


def test_a_tree_round_trips_through_a_upath_folder(
    bucket: str, tree: xr.DataTree
) -> None:
    folder = UPath(bucket) / "s0"

    tree.gs.to_zarr(folder)

    assert set(open_tree(folder).gs.groups) == set(tree.gs.groups)


def test_a_single_store_is_not_a_tree(tmp_path, tree: xr.DataTree) -> None:
    store = flat(tree.gs.to_zarr(tmp_path / "tree"))[0]

    with pytest.raises(ValueError, match="open_raster"):
        open_tree(store.path)


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

    assert scene.gs.anchor.geobox == scene.gs.geobox
    assert scene.gs.anchor.timespan == dated.gs.timespan
    assert build_tree({"dem": build_raster()}).gs.anchor.timespan is None


def test_map_groups_changes_every_group_and_keeps_the_root_attrs() -> None:
    raster = build_raster()
    built = build_tree({"optical": raster[["red"]], "dem": raster[["nir"]]})
    titled = _titled(built)

    halved = map_groups(titled, lambda group: group.isel(x=slice(0, 1)))

    assert halved.gs.groups == ("optical", "dem")
    assert halved.gs.geobox.shape.x == 1
    assert halved.attrs["title"] == "scene"


def test_map_groups_rebuilds_a_tree_that_shares_no_grid() -> None:
    raster = build_raster()
    coarse = raster[["nir"]].isel(x=slice(0, 1))
    built = build_tree({"optical": raster[["red"]], "dem": coarse})
    titled = _titled(built)

    renamed = map_groups(
        titled,
        lambda group: group.rename({name: f"{name}_raw" for name in group.data_vars}),
    )

    assert renamed.gs.geobox is None
    assert renamed.gs.variables == ("optical/red_raw", "dem/nir_raw")
    assert renamed.attrs["title"] == "scene"


def test_a_tree_reads_the_pixel_operations_through_gs(tree: xr.DataTree) -> None:
    import geopandas as gpd
    import shapely

    valid = np.array([[True, False], [True, True]])
    left, bottom, right, top = tree.gs.bounds.bbox
    column = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
        crs=tree.gs.crs,
    )

    assert tree.gs.unpack().gs.groups == tree.gs.groups
    assert tree.gs.to_nan().gs.groups == tree.gs.groups
    assert tree.gs.mask(valid, fill=0).gs.rasters["optical"].red.values[0, 1] == 0
    assert tree.gs.crop(column, mask=False).gs.geobox.shape == (2, 1)


def test_a_tree_reprojects_onto_a_target_rasters_own_grid(
    tree: xr.DataTree,
) -> None:
    target = build_raster(crs="EPSG:32748")

    warped = tree.gs.reproject(target)

    assert warped.gs.groups == tree.gs.groups
    assert warped.gs.geobox == target.gs.geobox
    assert tree.gs.reproject("EPSG:3857").gs.crs.epsg == 3857


def test_a_saved_tree_indexes_one_item_per_group_scene(tmp_path: Path) -> None:
    optical = build_raster(times=2)
    sample = build_tree({"optical": optical, "dem": build_raster()})

    saved = sample.gs.to_cog(
        {
            "optical": f"{tmp_path}/optical_{{time:%Y%m%dT%H%M%S}}.tif",
            "dem": f"{tmp_path}/dem.tif",
        }
    )
    rows = item_table(saved, prefix="s0")

    assert rows["id"].tolist() == [
        "s0/optical_20250601T000000",
        "s0/optical_20250602T000000",
        "s0/dem",
    ]
    assert rows["collection"].tolist() == ["optical", "optical", "dem"]
    assert rows.iloc[2]["assets"]["image"]["href"] == str(tmp_path / "dem.tif")


def test_shared_geographic_tree_spans_y_x() -> None:
    from affine import Affine
    from odc.geo import GeoBox
    from odc.geo.xr import xr_coords

    dims = ("y", "x")
    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    scene = xr.Dataset(
        {"red": (dims, np.ones((2, 3)))}, coords=xr_coords(grid, dims=dims)
    )
    scene.red.encoding["grid_mapping"] = "spatial_ref"

    tree = build_tree({"optical": scene, "dem": scene.rename({"red": "elevation"})})

    assert tree.gs.geobox == grid
    assert tree["optical"].dataset.red.dims == dims
    assert tree["dem"].dataset.elevation.dims == dims


def test_plot_takes_a_caption_without_asking_the_network(monkeypatch) -> None:
    from geosave.geodata.core.anchor import GeoAnchor

    def refuse(_):
        raise AssertionError("an explicit caption must not reverse geocode")

    monkeypatch.setattr(GeoAnchor, "locate", refuse)
    image = build_raster()

    drawn = build_tree({"red": image[["red"]], "nir": image[["nir"]]}).gs.plot(
        xlabel=""
    )

    assert len(drawn) == 2


def test_plot_leaves_room_between_columns_for_each_colorbar() -> None:
    image = build_raster()

    drawn = build_tree({"red": image[["red"]], "nir": image[["nir"]]}).gs.plot(
        cols=2, xlabel=""
    )

    assert drawn.opts.get("plot", backend="matplotlib").kwargs["hspace"] == 0.9


def test_bands_tree_as_one_variable_groups() -> None:
    from odc.geo.geobox import GeoBox

    from geosave.geodata.core.raster.create import raster as build
    from geosave.geodata.core.raster.create import tree as build_tree

    box = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32749", resolution=10)
    label = build({"label": (("y", "x"), np.zeros((2, 2), "uint8"))}, box)["label"]
    unnamed = label.copy().rename(None)

    tree = build_tree({"label": label, "prediction": unnamed})

    assert tree.gs.groups == ("label", "prediction")
    assert list(tree["label"].dataset.data_vars) == ["label"]
    assert list(tree["prediction"].dataset.data_vars) == ["prediction"]


def test_a_tree_converts_each_group_on_its_own_dtype(torch) -> None:
    from geosave.geodata.core.raster.create import tree

    scene = tree({"optical": optical(), "height": optical()[["B04"]].astype("float32")})

    tensors = scene.gs.to_tensor()
    assert tuple(tensors) == ("optical", "height")
    assert tensors["optical"].dtype is torch.uint16
    assert tensors["height"].dtype is torch.float32
    assert scene.gs.to_tensor(dtype="float32")["optical"].dtype is torch.float32
