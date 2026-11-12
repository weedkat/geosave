from __future__ import annotations

from tests.conftest import write_cogs

from tests.conftest import combine_item


import dask.array as da
import numpy as np
import pytest
import rasterio

from geosave.geodata.stac.catalog import create_catalog
from tests.conftest import build_dataset
from tests.geodata.conftest import create_file_items
import xarray as xr

from geosave.geodata.io.formats import geotiff
from tests.geodata.conftest import build_raster


def test_an_item_row_opens_lazily_on_its_native_grid(items, wide_grid):
    radar_row = items[items["collection"] == "s1"].iloc[0]

    raster = combine_item(radar_row.gs.to_datatree())

    assert isinstance(raster, xr.Dataset)
    assert raster.gs.geobox == wide_grid
    assert isinstance(raster.vv.data, da.Array)
    assert raster.vv.dtype == "uint16"


def test_boundary_record_has_an_accessor_but_is_not_a_stac_item():
    import geopandas as gpd
    import pytest
    from shapely.geometry import box

    boundary = gpd.GeoDataFrame(
        {"name": ["area"]}, geometry=[box(0, 0, 1, 1)], crs=4326
    )
    record = boundary.iloc[0]

    with pytest.raises(ValueError, match="STAC Item operations need.*to_catalog"):
        record.gs.to_item()


def test_an_item_row_reads_its_pixels_onto_a_grid(items, wide_grid):
    import numpy as np

    radar_row = items[items["collection"] == "s1"].iloc[0]

    pixels = radar_row.gs.read_pixels(wide_grid, variables=["vv"])

    assert pixels.odc.geobox == wide_grid
    stored = (
        combine_item(radar_row.gs.to_datatree())
        .vv.gs.mask_and_scale()
        .astype("float32")
        .values
    )
    np.testing.assert_array_equal(pixels.vv.values, stored)


def test_a_row_on_mixed_grids_reads_each_asset_as_a_leaf(tmp_path):
    from geosave.geodata.stac.catalog import create_catalog
    from tests.geodata.stac.test_publish import write_mixed_scene

    rows = create_catalog([write_mixed_scene(tmp_path)])

    row = rows.iloc[0].gs.to_datatree()
    tree = rows.gs.to_datatree()

    # A row opens as a one-row catalog does.
    assert row.isomorphic(tree)
    assert [leaf.name for leaf in tree.leaves] == ["red", "swir"]
    fine = tree.gs.datasets["s2/" + rows.iloc[0]["id"] + "/red"].gs.geobox
    scene = tree["s2"].gs.reproject(fine).gs.combine_stac()
    assert abs(scene.odc.geobox.resolution.x) == 10


def test_record_validation_checks_geometry_and_item_fields_independently():
    import geopandas as gpd
    import pytest
    from shapely.geometry import Point

    from geosave.geodata.errors import VectorValidationError

    rows = gpd.GeoDataFrame(geometry=[Point(0, 0), None], index=["site", "missing"])
    assert rows.iloc[0].gs.validate() is None
    with pytest.raises(VectorValidationError, match="STAC Item operations need"):
        rows.iloc[0].gs.validate(stac_item=True)
    assert rows.iloc[1].gs.validate(geometry=False) is None
    with pytest.raises(VectorValidationError, match="Record 'missing'.*missing"):
        rows.iloc[1].gs.validate()


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
    files = write_cogs(
        scene, tmp_path, format_paths="{variable}.tif", blocksize=256, overview_count=2
    )
    paths = {}
    for path in files:
        paths[path.stem] = path
    return next(iter(create_file_items(paths, id_format="pyramid")))


def record_overview_levels(monkeypatch) -> list:
    """Record the `overview_level` of each file opened, None for the base image."""
    levels = []
    original = rasterio.open

    def open_and_record(source, *args, **options):
        levels.append(options.get("overview_level"))
        return original(source, *args, **options)

    monkeypatch.setattr(rasterio, "open", open_and_record)
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
                write_cogs(scene, tmp_path, format_paths="red.tif"), id_format="utm"
            )
        )
    )
    view = GeoBox.from_bbox(
        grid.extent.to_crs("EPSG:3857").boundingbox, crs="EPSG:3857", shape=(32, 32)
    )

    pixels = read_pixels(entry, view)

    assert pixels.odc.geobox == view


def test_asset_window_masks_float32_missing_codes_without_scaling(tmp_path):
    from geosave.geodata.stac.catalog import create_catalog

    source = build_raster()[["red"]].astype("float32")
    source.red.data = np.array([[0.1, 2], [3, 4]], dtype="float32")
    source.red.attrs = {
        "missing_value": np.float32(0.1),
        "scale_factor": 2.0,
        "add_offset": 10.0,
    }
    path = tmp_path / "float.tif"
    geotiff.write_cog(source, path)
    entry = next(iter(create_file_items({"red": path})))

    window = create_catalog([entry]).iloc[0].gs.read_pixels(source.gs.geobox)
    decoded = source.gs.mask_and_scale()

    assert window.gs.geobox == source.gs.geobox
    np.testing.assert_array_equal(np.isnan(window.red), np.isnan(decoded.red))
    np.testing.assert_allclose(window.red, [[np.nan, 2], [3, 4]], equal_nan=True)
    np.testing.assert_allclose(decoded.red, [[np.nan, 14], [16, 18]], equal_nan=True)
