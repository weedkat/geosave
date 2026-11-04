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


@pytest.mark.parametrize("time", [np.datetime64("NaT", "ns"), "not-a-date"])
def test_invalid_group_dates_are_rejected_before_any_cog_is_written(tmp_path, time):
    first = build_raster(times=1)
    second = build_raster(times=1).assign_coords(time=[time])
    tree = build_tree({"first": first, "second": second})
    with pytest.raises(ValueError):
        tree.gs.to_cog(tmp_path, path_format="{collection}.tif")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("compute", [True, False])
def test_a_later_invalid_cog_group_is_rejected_before_any_write(tmp_path, compute):
    valid = build_raster()
    invalid = build_raster().gs.write_nodata(0, target="red")
    tree = build_tree({"first": valid, "second": invalid})
    with pytest.raises(ValueError, match="fill value"):
        tree.gs.to_cog(tmp_path, path_format="{collection}.tif", compute=compute)
    assert not list(tmp_path.iterdir())


def test_group_destinations_are_checked_together_before_writing(tmp_path):
    tree = build_tree({"first": build_raster(), "second": build_raster()})
    with pytest.raises(ValueError, match=r"collid.*\{collection\}"):
        tree.gs.to_cog(
            tmp_path, path_format={"first": "same.tif", "second": "same.tif"}
        )
    assert not list(tmp_path.iterdir())


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

    saved = written.gs.to_cog(tmp_path, path_format=template)

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
    written = build_tree({"image": image, "label": image})

    with pytest.raises(ValueError, match=r"collide.*scene\.tif"):
        written.gs.to_cog(tmp_path, path_format="scene.tif")
    assert not list(tmp_path.iterdir())


def test_groups_take_a_template_each(tmp_path: Path) -> None:
    written = build_tree({"optical": build_raster(times=2), "dem": build_raster()})

    saved = written.gs.to_cog(
        tmp_path, path_format={"optical": "optical_{time:%Y%m%d}.tif", "dem": "dem.tif"}
    )

    assert saved == [
        tmp_path / "optical_20250601.tif",
        tmp_path / "optical_20250602.tif",
        tmp_path / "dem.tif",
    ]


def test_time_on_an_undated_group_reports_which_group(
    tmp_path: Path,
) -> None:
    written = build_tree({"optical": build_raster(times=2), "dem": build_raster()})

    with pytest.raises(ValueError, match="'dem' has no dates"):
        written.gs.to_cog(tmp_path, path_format="{collection}_{time:%Y%m%d}.tif")


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


def test_tree_anchor_combines_valid_footprints_in_different_crs() -> None:
    from odc.geo.geobox import GeoBox

    projected = build_raster()
    geographic = build_raster().gs.write_grid(
        GeoBox.from_bbox((111, -9, 111.01, -8.99), crs="EPSG:4326", shape=(2, 2))
    )
    tree = build_tree({"whole": projected, "geographic": geographic})
    expected = projected.gs.anchor.geometry.union(
        geographic.gs.anchor.geometry.to_crs(projected.gs.crs)
    )
    assert tree.gs.anchor.geometry.geom.equals(expected.geom)


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

    assert scene.gs.anchor.geometry.geom.equals(scene.gs.geobox.extent.geom)
    assert scene.gs.anchor.timespan == dated.gs.timespan
    assert build_tree({"dem": build_raster()}).gs.anchor.timespan is None


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

    valid = np.array([[True, False], [True, True]])
    left, bottom, right, top = tree.gs.bounds.bbox
    column = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
        crs=tree.gs.crs,
    )

    assert tree.gs.unpack().gs.groups == tree.gs.groups
    assert tree.gs.to_nan().gs.groups == tree.gs.groups
    assert tree.gs.mask(valid, fill=0)["optical"].gs.to_dataset().red.values[0, 1] == 0
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
    from geosave.geodata.core.anchor import GeoAnchor

    def refuse(_):
        raise AssertionError("an explicit caption must not reverse geocode")

    monkeypatch.setattr(GeoAnchor, "locate", refuse)
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


def test_a_tree_keys_written_paths_by_group(tmp_path: Path) -> None:
    from tests.conftest import build_tree

    image = build_raster(times=1)
    label = image.isel(time=0, drop=True)[["red"]].rename(red="label")
    sample = build_tree({"optical": image, "label": label})

    files = sample.gs.to_cog(
        tmp_path,
        path_format={
            "optical": "{collection}_{time:%Y%m%d}.tif",
            "label": "{collection}.tif",
        },
    )

    assert files[0].name.startswith("optical_")
    assert files[1].name == "label.tif"


@pytest.mark.parametrize("method", ["to_cog", "to_netcdf"])
def test_a_deferred_write_computes_to_the_eager_result(tmp_path, method):
    from dask.delayed import Delayed
    from upath import UPath

    tree = build_tree({"image": build_raster(), "label": build_raster()})
    if method == "to_cog":
        pending = tree.gs.to_cog(
            tmp_path, path_format="{collection}_{band}.tif", compute=False
        )
    else:
        pending = tree.gs.to_netcdf(tmp_path / "scene.nc", compute=False)
    assert isinstance(pending, Delayed)
    assert list(tmp_path.iterdir()) == []
    saved = pending.compute()
    if method == "to_cog":
        assert [path.name for path in saved] == [
            "image_red.tif",
            "image_nir.tif",
            "label_red.tif",
            "label_nir.tif",
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
    tree = build_tree({"image": source, "label": source})
    module = {"geotiff": geotiff, "zarr": zarr, "netcdf": netcdf}[backend]
    # The step that runs when the delayed result computes, and receives the pixels.
    name = "write_cog" if backend == "geotiff" else "_write"
    write = getattr(module, name)
    chunks = []

    def save(raster, destination, *args, **options):
        rasters = (
            raster.gs.rasters.values() if isinstance(raster, xr.DataTree) else [raster]
        )
        for each in rasters:
            chunks.append(each.red.chunks)
        return write(raster, destination, *args, **options)

    monkeypatch.setattr(module, name, save)
    if method == "to_cog":
        pending = tree.gs.to_cog(
            tmp_path, path_format="{collection}.tif", compute=False
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


def test_a_bare_dataset_becomes_its_collections_grid_node(tree):
    rebuilt = build_tree({"optical": tree.gs.rasters[next(iter(tree.gs.rasters))]})

    assert list(rebuilt.gs.rasters) == ["optical/" + rebuilt["optical"].gs.groups[0]]
    assert rebuilt.gs.groups == ("optical",)


def test_rasters_round_trip_through_create_datatree():
    tree = build_mixed_collection()

    rebuilt = build_tree(tree.gs.rasters)

    assert list(rebuilt.gs.rasters) == ["s2/r10m", "s2/r20m"]
    for path, raster in tree.gs.rasters.items():
        xr.testing.assert_identical(rebuilt.gs.rasters[path], raster)


def test_a_collection_converts_on_its_finest_grid():
    tree = build_mixed_collection()

    dataset = tree["s2"].gs.to_dataset()

    assert abs(dataset.odc.geobox.resolution.x) == 10
    assert set(dataset.data_vars) == {"red", "swir"}
    assert dataset.sizes["time"] == 2
    assert float(dataset.swir.isel(time=0).max()) == 7


def test_a_coarser_member_averages_blocks_when_asked():
    tree = build_mixed_collection()

    dataset = tree["s2"].gs.to_dataset(res="r20m", resampling="average")

    assert abs(dataset.odc.geobox.resolution.x) == 20
    red = tree["s2/r10m"].dataset.red.isel(time=0).values
    assert float(dataset.red.isel(time=0, y=0, x=0)) == float(red[:2, :2].mean())


def test_several_collections_refuse_listing_them():
    tree = build_mixed_collection()
    both = build_tree(
        {**tree.gs.rasters, "dem/r10m": tree.gs.rasters["s2/r10m"].rename(red="height")}
    )

    with pytest.raises(ValueError, match=r"s2.*dem"):
        both.gs.to_dataset()


def test_nodes_that_do_not_nest_point_to_align():

    tree = build_mixed_collection()
    shifted = tree.gs.rasters["s2/r20m"].assign_coords(
        x=tree.gs.rasters["s2/r20m"].x + 5
    )
    broken = build_tree({"s2/r10m": tree.gs.rasters["s2/r10m"], "s2/r20m": shifted})

    with pytest.raises(ValueError, match="align"):
        broken["s2"].gs.to_dataset()


def test_a_collections_metadata_reaches_its_dataset():
    tree = build_mixed_collection()
    tree["s2"].attrs["title"] = "Sentinel-2"

    dataset = tree["s2"].gs.to_dataset()

    assert dataset.attrs["title"] == "Sentinel-2"
