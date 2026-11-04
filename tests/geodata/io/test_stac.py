"""STAC asset reading preserves file pixels and applies catalog descriptions."""

from __future__ import annotations

from datetime import UTC, datetime

import dask
import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from geosave.geodata import (
    open_vector,
)
from geosave.geodata.io.stac import open_scene
from geosave.geodata.stac.catalog import from_items
from tests.conftest import build_dataset
from tests.geodata.conftest import (
    _refuse,
    build_raster,
    create_file_items,
    item_table,
    read_items,
)
from tests.geodata.stac.conftest import create_native_item


def test_a_scene_closes_every_opened_asset_once(monkeypatch):
    from geosave.geodata.io import openers

    closed = []

    def read(path, **options):
        raster = build_raster()[["red"]].rename(red=str(path).split(".")[0])
        raster.set_close(lambda: closed.append(str(path)))
        return raster

    monkeypatch.setattr(openers, "open_raster", read)
    entry = pystac.Item("one", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    for name in ("first", "second"):
        entry.add_asset(name, pystac.Asset(f"{name}.tif", roles=["data"]))
    result = open_scene(entry)
    assert closed == []
    result.close()
    result.close()
    assert sorted(closed) == ["first.tif", "second.tif"]


@pytest.mark.parametrize("failure", ["open", "bands", "time"])
def test_a_scene_failure_closes_assets_already_opened(monkeypatch, failure):
    from geosave.geodata.io import openers
    from geosave.geodata.stac import item as translate_item

    closed = []

    def read(path, **options):
        if failure == "open" and str(path) == "second.tif":
            raise OSError("second asset failed")
        raster = build_raster()[["red"]].rename(red=str(path).split(".")[0])
        raster.set_close(lambda: closed.append(str(path)))
        return raster

    monkeypatch.setattr(openers, "open_raster", read)
    entry = pystac.Item("one", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    for name in ("first", "second"):
        entry.add_asset(name, pystac.Asset(f"{name}.tif", roles=["data"]))
    if failure == "bands":
        monkeypatch.setattr(
            translate_item,
            "read_variable_names",
            lambda _: {"first": ("first",), "second": ("red", "nir")},
        )
    if failure == "time":
        calls = []

        def fail_second(raster, entry):
            calls.append(raster)
            if len(calls) == 2:
                raise ValueError("time interpretation failed")
            return raster

        monkeypatch.setattr(translate_item, "stamp_item_time", fail_second)
    with pytest.raises((ValueError, OSError)):
        open_scene(entry)
    assert (
        closed == ["first.tif"]
        if failure == "open"
        else sorted(closed) == ["first.tif", "second.tif"]
    )


def test_native_paths_publish_actual_band_metadata(tmp_path):
    from tests.geodata.conftest import build_raster

    path = tmp_path / "image.tif"
    source = build_raster(times=1, packed=True)
    source.gs.to_cog(path.parent, path_format=path.name)
    items = list(create_file_items([path], collection="s2"))
    assert len(items) == 1
    assert items[0].collection_id == "s2"
    assert set(items[0].assets) == {"image"}
    with from_items(items).iloc[0].gs.to_datatree() as tree:
        restored = tree.gs.to_dataset()
        assert restored.red.dtype == source.red.dtype
        assert (
            restored.red.gs.packing.scale_factor == source.red.gs.packing.scale_factor
        )


def test_catalog_retains_observation_properties_across_native_publication(tmp_path):
    import geosave as gs
    from geosave.geodata.stac.item import stamp_properties_coordinate
    from tests.geodata.conftest import build_raster

    source = stamp_properties_coordinate(
        build_raster(times=1), [{"eo:cloud_cover": 12.0, "platform": "test-satellite"}]
    )
    rows = source.gs.to_catalog(tmp_path, path_format="{band}.tif", collection="s2")
    saved = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")
    item = gs.open_vector(saved).iloc[0].gs.to_item()
    assert item.properties["eo:cloud_cover"] == 12.0
    assert item.properties["platform"] == "test-satellite"


@pytest.fixture
def item(tmp_path):
    path = tmp_path / "image.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32749",
        transform=from_origin(500_000, 9_000_020, 10, 10),
        nodata=0,
    ) as tiff:
        tiff.write(np.stack([np.full((2, 2), value, "uint8") for value in (1, 2, 3)]))

    entry = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    entry.add_asset("visual", pystac.Asset(str(path), roles=["data"]))
    return entry


def test_unlisted_bands_use_the_asset_key_without_reading_pixels(item):
    original = item.to_dict()

    raster = open_scene(item).gs.to_dataset()

    assert list(raster.data_vars) == ["visual.1", "visual.2", "visual.3"]
    assert dask.is_dask_collection(raster["visual.3"])
    assert raster["visual.3"].gs.nodata == 0
    assert int(raster["visual.3"].max()) == 3
    assert item.to_dict() == original


def test_band_count_mismatch_reports_the_file(item):
    item.assets["visual"].extra_fields["bands"] = [{"name": "red"}, {"name": "green"}]

    with pytest.raises(ValueError, match=r"image.tif holds 3 bands.*names 2"):
        open_scene(item)


def test_native_composite_load_retains_bounds(tmp_path):
    from geosave.geodata.transform import temporal

    ds = temporal.reduce(temporal.resample(build_raster(times=1), "MS"), "median")
    original = next(
        create_file_items(ds.gs.to_cog(tmp_path, path_format="original.tif"))
    )
    loaded = open_scene(original).gs.to_dataset()
    rebuilt = next(
        create_file_items(loaded.gs.to_cog(tmp_path, path_format="rebuilt.tif"))
    )
    assert rebuilt.datetime == original.datetime
    assert rebuilt.properties["start_datetime"] == original.properties["start_datetime"]
    assert rebuilt.properties["end_datetime"] == original.properties["end_datetime"]


@pytest.mark.integration
def test_an_item_table_of_cogs_loads_back(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)
    paths = source.gs.to_cog(
        prefix, path_format="forest_{time:%Y%m%d}.tif", storage_options=options
    )
    # GDAL reads a remote TIFF with the keys in its environment, not fsspec options.
    rows = item_table(paths)

    saved = rows.gs.to_geoparquet(f"{prefix}/catalog.parquet", storage_options=options)

    rows = open_vector(saved, storage_options=options)
    assert rows.iloc[0]["assets"]["image"]["href"] == str(paths[0])
    with dask.config.set(scheduler=_refuse):
        restored = read_items(rows)
    np.testing.assert_array_equal(restored.red.values, source.red.values)
    restored.close()


def test_only_data_assets_load(tmp_path):
    entry, source = create_native_item(tmp_path)

    loaded = read_items(from_items([entry]))

    # The thumbnail is skipped; the data asset names its bands by its key.
    assert list(loaded.data_vars) == [
        f"reflectance.{index}" for index in range(1, len(source.data_vars) + 1)
    ]
    loaded.close()


@pytest.fixture
def catalog(tmp_path):
    """A dated optical collection and a timeless DEM on one grid, catalogued."""
    from tests.conftest import build_tree
    from tests.geodata.conftest import write_collection_cogs

    tree = build_tree(
        {"optical": build_raster(times=2), "dem": build_raster()[["red"]]}
    )
    files = write_collection_cogs(tree, tmp_path)
    return item_table(files, prefix="c")


def test_collections_open_as_groups_in_first_appearance_order(catalog):
    tree = catalog.gs.to_datatree()

    assert tree.gs.groups == tuple(dict.fromkeys(catalog["collection"]))
    assert list(tree.gs.rasters) == ["optical/r10m", "dem/r10m"]
    assert tree["optical"].gs.to_dataset().sizes["time"] == 2
    assert "time" not in tree["dem"].gs.to_dataset().dims
    assert tree["optical"].gs.to_dataset().red.chunks is not None


def test_an_item_without_a_collection_cannot_name_its_group(catalog):
    rows = catalog.copy()
    rows["collection"] = None

    with pytest.raises(ValueError, match="collection"):
        rows.gs.to_datatree()


def test_rows_on_two_dates_stack_along_time(tmp_path):
    scene = build_raster(times=2)
    rows = scene.gs.to_catalog(
        tmp_path, path_format="{time:%Y%m%d}/{band}.tif", collection="s2"
    )

    loaded = rows.gs.to_datatree()["s2"].gs.to_dataset()

    assert loaded.sizes["time"] == 2
    for name in scene.data_vars:
        np.testing.assert_array_equal(loaded[name].values, scene[name].values)


def write_mixed_scene(tmp_path, crs_by_name=None):
    """Write a red band at 10 m and a swir band at 20 m as one Item's assets."""
    from odc.geo.geobox import GeoBox

    crs_by_name = crs_by_name or {}
    paths = {}
    for name, resolution in [("red", 10), ("swir", 20)]:
        crs = crs_by_name.get(name, "EPSG:32749")
        grid = GeoBox.from_bbox(
            (500000, 9000000, 500640, 9000640), crs=crs, resolution=resolution
        )
        raster = build_dataset(
            {name: (("y", "x"), np.full(grid.shape, resolution, "uint16"))},
            grid,
            time=np.datetime64("2024-03-05"),
        )
        paths[name] = raster.gs.to_cog(tmp_path, path_format=f"{name}.tif")[0]
    return next(iter(create_file_items(paths, id_format="mixed", collection="s2")))


def test_a_mixed_resolution_row_opens_on_its_native_grids(tmp_path):
    entry = write_mixed_scene(tmp_path)

    scene = open_scene(entry)

    assert list(scene.gs.rasters) == ["r10m", "r20m"]
    assert scene["r10m"].gs.to_dataset().odc.geobox.shape == (64, 64)
    assert scene["r20m"].gs.to_dataset().odc.geobox.shape == (32, 32)
    assert int(scene["r20m"].gs.to_dataset().swir.max()) == 20


def test_a_single_grid_row_converts_to_its_scene(tmp_path):
    scene = build_raster(times=1)
    rows = scene.gs.to_catalog(tmp_path, path_format="{band}.tif", collection="s2")

    with rows.iloc[0].gs.to_datatree() as tree:
        loaded = tree.gs.to_dataset()

        for name in scene.data_vars:
            np.testing.assert_array_equal(
                loaded[name].values, scene[name].isel(time=0).values
            )


# A 1024² raster in Web Mercator, 10 m pixels, its COGs holding overviews 2 and 4.
PIXEL_SIZE = 10.0


@pytest.fixture
def pyramid_item(tmp_path):
    from odc.geo.geobox import GeoBox

    grid = GeoBox.from_bbox(
        (0, 0, 1024 * PIXEL_SIZE, 1024 * PIXEL_SIZE),
        crs="EPSG:3857",
        resolution=PIXEL_SIZE,
    )
    values = np.arange(1024 * 1024, dtype="uint16").reshape(1024, 1024) % 1000 + 1
    values[:8, :8] = 0
    scene = build_dataset(
        {"red": (("y", "x"), values), "nir": (("y", "x"), values * 2)},
        grid,
        nodata=0,
        time=np.datetime64("2024-03-05"),
    )
    files = scene.gs.to_cog(
        tmp_path, path_format="{band}.tif", blocksize=256, overview_count=2
    )
    paths = {}
    for path in files:
        paths[path.stem] = path
    return next(iter(create_file_items(paths, id_format="pyramid")))


def record_overview_levels(monkeypatch) -> list:
    """Record the `overview_level` of each file opened, None for the base image."""
    from geosave.geodata.io import openers

    levels = []
    original = openers.open_raster

    def open_and_record(source, **options):
        levels.append(options.get("overview_level"))
        return original(source, **options)

    monkeypatch.setattr(openers, "open_raster", open_and_record)
    return levels


def read_pixels(entry, geobox, variables=None):
    return from_items([entry]).iloc[0].gs.read_pixels(geobox, variables=variables)


def create_view(left, bottom, right, top, pixel_size):
    from odc.geo.geobox import GeoBox

    return GeoBox.from_bbox(
        (left, bottom, right, top), crs="EPSG:3857", resolution=pixel_size
    )


def test_read_pixels_reads_the_overview_the_view_needs(pyramid_item, monkeypatch):
    levels = record_overview_levels(monkeypatch)
    view = create_view(0, 0, 10240, 10240, 4 * PIXEL_SIZE)

    pixels = read_pixels(pyramid_item, view, variables=["red"])

    assert pixels.odc.geobox == view
    assert pixels["red"].dtype == np.float32
    # The base image's header, then overview factor 4: level index 1 of [2, 4]
    assert levels == [None, 1]


def test_read_pixels_reads_the_base_image_when_zoomed_in(pyramid_item, monkeypatch):
    levels = record_overview_levels(monkeypatch)
    view = create_view(1000, 1000, 2000, 2000, PIXEL_SIZE)

    pixels = read_pixels(pyramid_item, view, variables=["red"])

    assert levels == [None]
    assert pixels["red"].shape == (100, 100)
    assert np.isfinite(pixels["red"].values).all()


def test_read_pixels_outside_the_raster_opens_nothing(pyramid_item, monkeypatch):
    levels = record_overview_levels(monkeypatch)
    view = create_view(50000, 50000, 60000, 60000, PIXEL_SIZE * 10)

    pixels = read_pixels(pyramid_item, view, variables=["red"])

    assert pixels.odc.geobox == view
    assert np.isnan(pixels["red"].values).all()
    assert levels == []


def test_read_pixels_opens_only_assets_holding_the_variables(pyramid_item, monkeypatch):
    levels = record_overview_levels(monkeypatch)
    view = create_view(1000, 1000, 2000, 2000, PIXEL_SIZE)

    pixels = read_pixels(pyramid_item, view, variables=["nir"])

    assert list(pixels.data_vars) == ["nir"]
    assert len(levels) == 1


def test_read_pixels_turns_nodata_into_nan(pyramid_item):
    view = create_view(0, 10240 - 160, 160, 10240, PIXEL_SIZE)

    pixels = read_pixels(pyramid_item, view, variables=["red"])

    # The raster's top-left 8×8 pixels hold nodata
    assert np.isnan(pixels["red"].values[:8, :8]).all()
    assert np.isfinite(pixels["red"].values[8:, 8:]).all()


def test_read_pixels_states_the_grid_it_reads_onto(tmp_path):
    from odc.geo.geobox import GeoBox

    grid = GeoBox.from_bbox(
        (500000, 9000000, 500640, 9000640), crs="EPSG:32749", resolution=10
    )
    scene = build_dataset(
        {"red": (("y", "x"), np.ones((64, 64), "uint16"))},
        grid,
        time=np.datetime64("2024-03-05"),
    )
    entry = next(
        iter(
            create_file_items(
                scene.gs.to_cog(tmp_path, path_format="red.tif"), id_format="utm"
            )
        )
    )
    view = GeoBox.from_bbox(
        grid.extent.to_crs("EPSG:3857").boundingbox, crs="EPSG:3857", shape=(32, 32)
    )

    pixels = read_pixels(entry, view)

    assert pixels.odc.geobox == view


def write_png_item(tmp_path, stated_shape=(48, 64)):
    """Write a PNG, which holds no grid, and an Item whose asset states one."""
    from odc.geo.geobox import GeoBox

    from geosave.geodata.stac.extensions import projection

    path = tmp_path / "label.png"
    with rasterio.open(
        path, "w", driver="PNG", width=64, height=48, count=1, dtype="uint8"
    ) as png:
        png.write(np.arange(64 * 48, dtype="uint8").reshape(1, 48, 64))
    entry = pystac.Item("label", None, None, datetime(2024, 3, 5, tzinfo=UTC), {})
    entry.add_asset("label", pystac.Asset(str(path), roles=["data"]))
    rows, columns = stated_shape
    grid = GeoBox.from_bbox(
        (500000, 9000000, 500000 + columns * 10, 9000000 + rows * 10),
        crs="EPSG:32749",
        resolution=10,
    )
    projection.write_asset(entry.assets["label"], grid)
    return entry, grid


def test_an_asset_states_the_grid_a_png_lacks(tmp_path):
    entry, grid = write_png_item(tmp_path)

    scene = open_scene(entry).gs.to_dataset()

    assert scene.odc.geobox == grid


def test_a_file_that_nothing_places_is_refused_naming_its_asset(tmp_path):
    entry, _ = write_png_item(tmp_path, stated_shape=(10, 10))

    with pytest.raises(ValueError, match="label.*proj"):
        open_scene(entry)


def test_a_files_own_grid_wins_over_its_asset(tmp_path):
    from odc.geo.geobox import GeoBox

    from geosave.geodata.stac.extensions import projection

    grid = GeoBox.from_bbox(
        (500000, 9000000, 500640, 9000640), crs="EPSG:32749", resolution=10
    )
    scene = build_dataset(
        {"red": (("y", "x"), np.ones(grid.shape, "uint16"))},
        grid,
        time=np.datetime64("2024-03-05"),
    )
    entry = next(
        iter(
            create_file_items(
                {"red": scene.gs.to_cog(tmp_path, path_format="red.tif")[0]},
                id_format="red",
                collection="s2",
            )
        )
    )
    elsewhere = GeoBox.from_bbox(
        (600000, 9000000, 600640, 9000640), crs="EPSG:32749", resolution=10
    )
    projection.write_asset(entry.assets["red"], elsewhere)

    with pytest.warns(UserWarning, match="states a grid"):
        opened = open_scene(entry).gs.to_dataset()

    assert opened.odc.geobox == grid
