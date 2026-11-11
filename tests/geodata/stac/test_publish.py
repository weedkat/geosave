"""Publication splits rasters into Items and assets; asset reading keeps file pixels."""

from __future__ import annotations


from datetime import UTC, datetime

import dask
import dask.array as da
import numpy as np
import pystac
import pytest
import rasterio
import xarray as xr
from dask.callbacks import Callback
from rasterio.transform import from_origin

from geosave.geodata import (
    open_vector,
)
from geosave.geodata.stac.load import load_datatree
from geosave.geodata.stac.publish import create_stac_item, split_items
from geosave.geodata.stac.catalog import create_catalog
from tests.conftest import build_dataset
from tests.geodata.conftest import (
    _refuse,
    assign_cells,
    build_raster,
    create_file_items,
    create_test_collection,
    item_table,
    read_items,
)
from tests.geodata.stac.conftest import create_native_item


def test_a_catalog_loads_one_node_per_asset_whatever_the_asset_order(tmp_path):
    from affine import Affine
    from odc.geo.geobox import GeoBox

    base = Affine(10, 0, 500000, 0, -10, 9000020)
    grids = {
        "red": GeoBox((4, 4), base, "EPSG:32749"),
        "nir": GeoBox((4, 4), base, "EPSG:32749"),
        "swir": GeoBox((2, 2), base * Affine.scale(2), "EPSG:32749"),
    }
    sources = {}
    for value, (name, grid) in enumerate(grids.items(), start=1):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=1,
            dtype="uint16",
            crs=str(grid.crs),
            transform=grid.transform,
        ) as file:
            file.write(np.full(tuple(grid.shape), value, dtype="uint16"), 1)
        sources[name] = path
    entry = create_stac_item(
        sources,
        id="first",
        input_datetime=datetime(2025, 1, 1, tzinfo=UTC),
        with_proj=True,
        with_raster=True,
        with_eo=True,
    )
    entry.set_collection(create_test_collection("s2", description="S2"))
    second = entry.clone()
    second.id = "second"
    second.datetime = datetime(2025, 1, 2, tzinfo=UTC)
    second.assets = dict(reversed(list(second.assets.items())))

    tree = create_catalog([entry, second]).gs.to_datatree(groupby="id")
    assert set(tree.gs.datasets) == {"s2/red", "s2/nir", "s2/swir"}
    assert tree["s2"].sizes["time"] == 2
    for name, grid in grids.items():
        node = tree[f"s2/{name}"]
        assert node.gs.geobox == grid
        np.testing.assert_array_equal(
            node[name].values, np.full((2, *grid.shape), list(grids).index(name) + 1)
        )


def test_native_paths_publish_actual_band_metadata(tmp_path):
    from tests.geodata.conftest import build_raster

    path = tmp_path / "image.tif"
    source = build_raster(times=1, packed=True)
    source.gs.to_cog(path.parent, split_asset=(), format_paths=path.name)
    items = list(create_file_items([path], collection="s2"))
    assert len(items) == 1
    assert items[0].collection_id == "s2"
    assert set(items[0].assets) == {"image"}
    with create_catalog(items).iloc[0].gs.to_datatree() as tree:
        restored = tree.gs.to_dataset()
        assert restored.red.dtype == source.red.dtype
        assert (
            restored.red.gs.packing.scale_factor == source.red.gs.packing.scale_factor
        )


def test_catalog_retains_observation_properties_across_native_publication(tmp_path):
    import geosave as gs
    from geosave.geodata.stac.raster import stamp_properties_coordinate
    from tests.geodata.conftest import build_raster, create_test_collection

    source = stamp_properties_coordinate(
        build_raster(times=1), [{"eo:cloud_cover": 12.0, "platform": "test-satellite"}]
    )
    rows = source.gs.to_catalog(
        tmp_path,
        format_paths="{variable}.tif",
        collection=create_test_collection("s2", description="S2"),
    )
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

    return create_stac_item(
        {"visual": path},
        id="scene",
        input_datetime=datetime(2025, 1, 1, tzinfo=UTC),
        with_proj=True,
        with_raster=True,
        with_eo=True,
    )


def test_unlisted_bands_use_the_asset_key_without_reading_pixels(item):
    original = item.to_dict()

    raster = load_datatree([item]).gs.to_dataset()

    assert list(raster.data_vars) == ["visual.1", "visual.2", "visual.3"]
    assert dask.is_dask_collection(raster["visual.3"])
    assert raster["visual.3"].gs.nodata == 0
    assert int(raster["visual.3"].max()) == 3
    assert item.to_dict() == original


def test_native_composite_load_retains_bounds(tmp_path):

    # Scenes on June 1 and 2, so the composite spans both.
    ds = build_raster(times=2).gs.mask_and_scale().gs.composite("MS", "median")
    original = next(
        create_file_items(
            ds.gs.to_cog(tmp_path, split_asset=(), format_paths="original.tif")
        )
    )
    loaded = load_datatree([original]).gs.to_dataset()
    rebuilt = next(
        create_file_items(
            loaded.gs.to_cog(tmp_path, split_asset=(), format_paths="rebuilt.tif")
        )
    )
    assert original.datetime is None
    assert rebuilt.datetime == original.datetime
    assert rebuilt.properties["start_datetime"] == original.properties["start_datetime"]
    assert rebuilt.properties["end_datetime"] == original.properties["end_datetime"]


@pytest.mark.integration
def test_an_item_table_of_cogs_loads_back(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)
    paths = source.gs.to_cog(
        prefix, format_paths="forest_{time:%Y%m%d}.tif", storage_options=options
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

    loaded = read_items(create_catalog([entry]))

    # The thumbnail is skipped; the data asset's bands keep the names it states.
    assert list(loaded.data_vars) == list(source.data_vars)
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
    assert list(tree.gs.datasets) == ["optical/image", "dem/red"]
    assert tree["optical"].gs.to_dataset().sizes["time"] == 2
    # The DEM Item is dated, so it loads as one time step, as stac.load_dataset does.
    assert tree["dem"].gs.to_dataset().sizes["time"] == 1
    assert tree["optical"].gs.to_dataset().red.chunks is not None


def test_an_item_without_a_collection_cannot_name_its_group(catalog):
    rows = catalog.copy()
    rows["collection"] = None

    with pytest.raises(ValueError, match="collection"):
        rows.gs.to_datatree()


def test_rows_on_two_dates_stack_along_time(tmp_path):
    scene = build_raster(times=2)
    rows = scene.gs.to_catalog(
        tmp_path,
        format_paths="{time:%Y%m%d}/{variable}.tif",
        collection=create_test_collection("s2", description="S2"),
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
        paths[name] = raster.gs.to_cog(tmp_path, format_paths=f"{name}.tif")[0]
    return next(iter(create_file_items(paths, id_format="mixed", collection="s2")))


def test_a_mixed_resolution_row_opens_on_its_native_grids(tmp_path):
    entry = write_mixed_scene(tmp_path)

    scene = load_datatree([entry])

    assert list(scene.gs.datasets) == ["s2/red", "s2/swir"]
    assert scene["s2/red"].gs.to_dataset().odc.geobox.shape == (64, 64)
    assert scene["s2/swir"].gs.to_dataset().odc.geobox.shape == (32, 32)
    assert int(scene["s2/swir"].gs.to_dataset().swir.max()) == 20


def test_a_single_grid_row_converts_to_its_scene(tmp_path):
    scene = build_raster(times=1)
    rows = scene.gs.to_catalog(
        tmp_path,
        format_paths="{variable}.tif",
        collection=create_test_collection("s2", description="S2"),
    )

    with rows.iloc[0].gs.to_datatree() as tree:
        loaded = tree.gs.to_dataset()

        for name in scene.data_vars:
            np.testing.assert_array_equal(
                loaded[name].isel(time=0).values, scene[name].isel(time=0).values
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
        tmp_path, format_paths="{variable}.tif", blocksize=256, overview_count=2
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
    return create_catalog([entry]).iloc[0].gs.read_pixels(geobox, variables=variables)


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
                scene.gs.to_cog(tmp_path, format_paths="red.tif"), id_format="utm"
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

    scene = load_datatree([entry]).gs.to_dataset()

    assert scene.odc.geobox == grid


def test_create_stac_item_describes_saved_assets(tmp_path):
    import rasterio
    from geosave.geodata.io.formats.geotiff import write_cog
    from geosave.geodata.stac.publish import create_stac_item

    dataset = build_raster(times=1)
    paths = {}
    for name in dataset.data_vars:
        paths[name] = write_cog(dataset[[name]], tmp_path / f"{name}.tif")

    entry = create_stac_item(
        paths,
        id="scene",
        input_datetime=datetime(2025, 6, 1, tzinfo=UTC),
        with_proj=True,
        with_raster=True,
        with_eo=True,
    )

    assert set(entry.assets) == {"red", "nir"}
    assert entry.datetime == datetime(2025, 6, 1, tzinfo=UTC)
    for name, asset in entry.assets.items():
        with rasterio.open(paths[name]) as raster:
            assert asset.extra_fields["proj:shape"] == [raster.height, raster.width]
            assert (
                asset.extra_fields["raster:bands"][0]["data_type"] == raster.dtypes[0]
            )
            assert asset.extra_fields["eo:bands"][0]["name"] == name
    assert "proj:shape" not in entry.properties


def test_saved_item_preserves_nominal_time_from_interval_properties(tmp_path):
    from copy import deepcopy
    from geosave.geodata.io.formats.geotiff import write_cog
    from geosave.geodata.stac.publish import create_stac_item

    path = write_cog(build_raster(times=1), tmp_path / "composite.tif")
    properties = {
        "datetime": "2025-06-15T00:00:00Z",
        "start_datetime": "2025-06-01T00:00:00Z",
        "end_datetime": "2025-07-01T00:00:00Z",
        "platform": "s2a",
    }
    before = deepcopy(properties)
    entry = create_stac_item({"image": path}, id="composite", properties=properties)

    assert entry.datetime == datetime(2025, 6, 15, tzinfo=UTC)
    assert entry.to_dict()["properties"]["datetime"] == "2025-06-15T00:00:00Z"
    assert entry.properties["platform"] == "s2a"
    assert properties == before


def _scene():
    source = build_raster(times=2).chunk()
    return source.assign(dem=source.red.isel(time=0, drop=True).astype("float32"))


def test_items_follow_time_and_assets_follow_variables_without_computing():
    scene = _scene()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        items = split_items({".": scene}, ("time",), ("variable",))
    assert tasks == []
    assert [list(item) for item in items] == [["red", "nir", "dem"]] * 2
    for position, item in enumerate(items):
        assert item["red"].time.item() == scene.time.values[position].item()
        assert list(item["red"].data_vars) == ["red"]
        assert isinstance(item["red"].red.data, da.Array)
    # A static variable joins every Item rather than being split by time.
    # It carries the Item's time as a scalar label, like every asset of that Item.
    xr.testing.assert_identical(items[1]["dem"].dem.drop_vars("time"), scene.dem)


def test_a_lone_raster_written_whole_is_the_sole_asset():
    items = split_items({".": _scene()}, ("time",), ())
    assert [list(item) for item in items] == [["asset"], ["asset"]]
    assert list(items[0]["asset"].data_vars) == ["red", "nir", "dem"]


def test_tree_assets_keep_their_node_path():
    scene = build_raster(times=2)
    rasters = {"s2/r10m": scene, "label": scene[["red"]].rename(red="mask")}
    by_variable = split_items(rasters, ("time",), ("variable",))
    assert list(by_variable[0]) == ["s2/r10m/red", "s2/r10m/nir", "label/mask"]
    whole = split_items(rasters, ("time",), ())
    assert list(whole[0]) == ["s2/r10m", "label"]
    (single,) = split_items({"s2": scene.isel(time=[0])}, ("time",), ())
    assert list(single) == ["s2"]


def test_asset_dimensions_add_their_labels_to_the_name():
    reflectance = build_raster(times=1).gs.to_dataarray().rename("reflectance")
    (item,) = split_items(
        {".": reflectance.to_dataset()}, ("time",), ("variable", "band")
    )
    assert list(item) == ["reflectance_red", "reflectance_nir"]
    assert item["reflectance_red"].band.item() == "red"
    (item,) = split_items({".": reflectance.to_dataset()}, ("time",), ("band",))
    assert list(item) == ["red", "nir"]


def test_absent_split_dimensions_are_skipped():
    static = build_raster()
    (item,) = split_items({".": static}, ("time",), ("variable", "level"))
    assert list(item) == ["red", "nir"]


def test_repeated_asset_names_are_rejected():
    scene = build_raster(times=1).gs.to_dataarray().to_dataset(name="image")
    scene = scene.assign_coords(band=["red", "red"])
    with pytest.raises(ValueError, match="repeats"):
        split_items({".": scene}, ("time",), ("band",))


def test_variable_assets_keep_cf_time_bounds():
    scene = build_raster(times=2)
    starts = scene.time.values
    ends = starts + np.timedelta64(1, "D")
    scene = assign_cells(scene, list(zip(starts, ends, strict=True)))
    items = split_items({".": scene}, ("time",), ("variable",))
    assert items[1]["nir"].end_datetime.values == ends[1]
