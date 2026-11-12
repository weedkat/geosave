from tests.conftest import write_cogs
import dask.array as da
import numpy as np
import pytest
from dask.callbacks import Callback

from geosave.geodata.io.formats.geotiff import write_cog
from geosave.geodata.viz.explore import RasterLayer


@pytest.fixture(autouse=True)
def restore_plot_backend():
    import holoviews as hv

    backend = hv.Store.current_backend
    hv.extension("bokeh")
    yield
    hv.Store.set_current_backend(backend)


def create_view(x_range, y_range, shape=(600, 900)):
    from geosave.geodata.viz.explore.mercator import create_view as create_map_view

    return create_map_view(x_range, y_range, shape)


def create_layer_view(layer):
    box = layer.bbox
    return create_view((box.left, box.right), (box.bottom, box.top))


def count_reads(monkeypatch) -> list:
    """Record each Dataset a layer reprojects onto its view."""
    from geosave.geodata.core.raster.dataset import GeoDataset

    read_data = []
    original = GeoDataset.reproject

    def read_and_record(self, how, **options):
        read_data.append(self._data)
        return original(self, how, **options)

    monkeypatch.setattr(GeoDataset, "reproject", read_and_record)
    return read_data


def create_mercator_data():
    import xarray as xr
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox.from_bbox((0, 0, 40960, 40960), crs=3857, shape=(4096, 4096))
    y, x = da.indices((4096, 4096), chunks=(256, 256))
    values = xr.DataArray(
        (x + y).astype("float32"), dims=("y", "x"), coords=xr_coords(grid)
    )
    return xr.Dataset({"value": values})


def test_pan_inside_the_cache_reads_nothing(monkeypatch):
    read_data = count_reads(monkeypatch)
    layer = RasterLayer(create_mercator_data())
    layer.draw(create_view((10240, 20480), (10240, 17067)))

    # A fifth of the view sideways stays inside the half-view padding
    moved = layer.draw(create_view((12288, 22528), (10240, 17067)))

    assert len(read_data) == 1
    values = moved.values()[0].dimension_values(2, flat=False)
    assert float(np.nanmax(values)) > float(np.nanmin(values))


def test_zooming_past_a_level_reads_again(monkeypatch):
    read_data = count_reads(monkeypatch)
    layer = RasterLayer(create_mercator_data())
    layer.draw(create_view((10240, 20480), (10240, 17067)))

    layer.draw(create_view((12800, 17920), (12800, 16213)))

    assert len(read_data) == 2


def test_contrast_change_reads_nothing(monkeypatch):
    read_data = count_reads(monkeypatch)
    layer = RasterLayer(create_mercator_data())
    view = create_view((10240, 20480), (10240, 17067))
    layer.draw(view)

    layer.clim = (0, 100)
    layer.draw(view)

    assert len(read_data) == 1


def test_changing_the_date_reads_again(monthly, monkeypatch):
    read_data = count_reads(monkeypatch)
    layer = RasterLayer(monthly, "B04")
    view = create_layer_view(layer)
    layer.draw(view)

    layer.time = monthly.time.values[1]
    layer.draw(view)

    assert len(read_data) == 2


def test_a_read_covers_only_the_shown_date(monthly, monkeypatch):
    read_data = count_reads(monkeypatch)
    layer = RasterLayer(monthly, "B04")

    layer.draw(create_layer_view(layer))

    assert "time" not in read_data[0].dims


def test_row_layer_reads_its_rows_shown_bands(catalog, monkeypatch):
    import odc.stac

    from geosave.geodata.core.vector.item import GeoCatalogItem

    requested = []
    original = GeoCatalogItem.read_pixels

    def read_and_record(self, geobox, *, variables=None):
        requested.append(list(variables))
        return original(self, geobox, variables=variables)

    monkeypatch.setattr(GeoCatalogItem, "read_pixels", read_and_record)
    row = catalog.iloc[1]
    (parsed,) = odc.stac.parse_items([row.gs.to_item()])
    layer = RasterLayer(catalog.iloc[[1]].gs.to_dataset(parsed.geoboxes()[0]), row=row)

    image = layer.draw(create_layer_view(layer)).values()[0]

    assert requested == [["B04"]]
    assert np.isfinite(image.dimension_values(2)).any()


def _render_pixels(layer):
    import numpy as np

    image = layer.draw(create_layer_view(layer)).values()[0]
    return np.asarray(image.dimension_values(2, flat=False))


def test_geographic_file_renders_through_its_windows(tmp_path):
    import numpy as np
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataset as build_raster
    from geosave.geodata.io import open_raster

    grid = GeoBox.from_bbox((100, 0, 101, 1), "EPSG:4326", shape=(64, 64))
    values = np.random.default_rng(0).random((64, 64)).astype("float32")
    path = tmp_path / "geographic.tif"
    write_cog(build_raster({"value": (("y", "x"), values)}, grid), path)

    pixels = _render_pixels(RasterLayer(open_raster(path)))

    assert np.isfinite(pixels).any()


def test_utm_tile_across_the_antimeridian_renders_both_sides():
    import numpy as np
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataset as build_raster

    grid = GeoBox.from_bbox(
        (120000, 8000000, 220000, 8100000), "EPSG:32701", shape=(100, 100)
    )
    values = np.ones((100, 100), "float32")
    layer = RasterLayer(build_raster({"value": (("y", "x"), values)}, grid))

    pixels = _render_pixels(layer)
    box = layer.bbox
    middle = (box.bottom + box.top) / 2

    assert np.isfinite(pixels).any()
    assert layer.identify(box.left + 5000, middle) == {"value": "1.0"}
    assert layer.identify(box.right - 5000, middle) == {"value": "1.0"}


def test_longitudes_past_180_are_refused_with_the_fix():
    import numpy as np
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataset as build_raster

    grid = GeoBox.from_bbox((0, -60, 360, 60), "EPSG:4326", resolution=10)
    values = np.ones(grid.shape, "float32")
    with pytest.raises(ValueError, match="wrap them into -180..180"):
        RasterLayer(build_raster({"t2m": (("y", "x"), values)}, grid))


def test_steps_select_an_axis_besides_time(scene):
    import numpy as np
    import xarray as xr

    levels = xr.concat(
        [scene["B04"] * 0 + level for level in (1, 2, 3)], dim="level"
    ).assign_coords(level=[850, 500, 200])
    layer = RasterLayer(levels.to_dataset(name="t"))

    assert layer.axis_labels == {"level": ["850", "500", "200"]}
    assert np.nanmax(_render_pixels(layer)) == 1
    layer.axis_positions = {"level": 2}
    assert np.nanmax(_render_pixels(layer)) == 3


def _stretch_rgb(layer, sampled):
    import holoviews as hv

    hv.extension("bokeh")
    return layer._stretch(
        layer._draw_pixels(sampled), layer.clim, layer.stretch_percentiles
    ).values()[0]


def test_rgb_balances_channels_without_changing_source(scene):
    import numpy as np

    original = scene.copy(deep=True)
    image = _stretch_rgb(RasterLayer(scene), scene)
    red = image.dimension_values("R", flat=False)
    np.testing.assert_allclose(red, image.dimension_values("G", flat=False), atol=1e-6)
    np.testing.assert_allclose(red, image.dimension_values("B", flat=False), atol=1e-6)
    assert np.nanmin(red) == 0
    assert np.nanmax(red) == 1
    assert scene.identical(original)


def test_rgb_uses_scene_contrast_even_when_first_view_is_a_crop(scene):
    import numpy as np

    # A dark first viewport must use the same mapping as the whole scene.
    dark = scene.where(scene.B04 < 0.5)
    layer = RasterLayer(scene)
    first = _stretch_rgb(layer, dark)
    whole = _stretch_rgb(layer, scene)
    for dim in ("R", "G", "B"):
        cropped = first.dimension_values(dim, flat=False)
        full = whole.dimension_values(dim, flat=False)
        mask = np.isfinite(cropped)
        np.testing.assert_allclose(cropped[mask], full[mask])


def test_rgb_contrast_refreshes_for_date_and_steps_without_slider_reads(scene):
    import numpy as np
    import xarray as xr

    data = xr.concat([scene, scene + 10], dim="time").assign_coords(
        time=np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]")
    )
    data = xr.concat([data, data + 20], dim="level").chunk({"x": 16, "y": 16})
    layer = RasterLayer(data)
    first = _stretch_rgb(layer, layer.select_shown_bands().compute())
    layer.time = data.time.values[1]
    layer.axis_positions = {"level": 1}
    selected = layer.select_shown_bands().compute()
    second = _stretch_rgb(layer, selected)
    for dim in ("R", "G", "B"):
        np.testing.assert_allclose(
            first.dimension_values(dim), second.dimension_values(dim), atol=1e-5
        )
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        layer.stretch_percentiles = (10, 90)
        _stretch_rgb(layer, selected)
    assert tasks == []


def test_rgb_manual_range_overrides_balancing(scene):
    import numpy as np

    layer = RasterLayer(scene)
    layer.clim = (0, 3)
    image = _stretch_rgb(layer, scene)
    for dim, name in zip(("R", "G", "B"), layer.rgb, strict=True):
        # HoloViews dimension_values follows ascending y order.
        np.testing.assert_allclose(
            np.sort(image.dimension_values(dim)),
            np.sort(scene[name].values.ravel() / 3),
        )


def test_rgb_scene_overview_is_bounded(scene, tmp_path):
    import numpy as np
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataset as build_raster
    from geosave.geodata.io import open_raster

    grid = GeoBox.from_bbox((0, 0, 5120, 5120), crs=3857, shape=(512, 512))
    data = build_raster(
        {
            name: (("y", "x"), np.tile(scene[name].values, (8, 8)))
            for name in scene.gs.rgb
        },
        grid,
    ).gs.write_rgb(*scene.gs.rgb)
    path = tmp_path / "rgb.tif"
    write_cogs(data, path.parent, format_paths=path.name)
    loaded = open_raster(path)
    layer = RasterLayer(loaded)
    sampled = loaded.isel(x=slice(0, 32), y=slice(0, 32)).compute()
    image = _stretch_rgb(layer, sampled)
    assert layer._overview.sizes == {"x": 256, "y": 256}
    assert isinstance(loaded.B04.data, da.Array)
    np.testing.assert_allclose(
        image.dimension_values("R"), image.dimension_values("B"), atol=1e-6
    )
    loaded.close()


def test_rgb_flat_and_missing_channels_keep_nodata_transparent(scene):
    import numpy as np

    data = scene.copy(deep=True)
    data.B04.values[:] = 5
    data.B03.values[:] = np.nan
    image = _stretch_rgb(RasterLayer(data), data)
    assert np.all(image.dimension_values("R") == 0)
    assert np.isnan(image.dimension_values("G")).all()
    assert np.all(image.dimension_values("A") == 0)


def test_rgb_band_change_updates_the_live_plot(scene):
    import holoviews as hv
    import numpy as np

    hv.extension("bokeh")
    layer = RasterLayer(scene)
    layer.clim = (0, 3)
    first = layer.draw(None).values()[0]
    blue = first.dimension_values("B").copy()
    layer.rgb = ("B02", "B03", "B04")
    second = layer.draw(None).values()[0]
    np.testing.assert_allclose(second.dimension_values("R"), blue)


def test_a_band_without_limits_fits_its_shown_values(scene):
    import numpy as np

    layer = RasterLayer(scene, "B04")
    plot = layer._draw_pixels(layer.read_view_pixels(create_layer_view(layer)))
    layer._stretch(plot, None, (0.0, 100.0))

    values = plot.values()[0].dimension_values(layer.band)
    assert layer.fitted_clim == (float(np.nanmin(values)), float(np.nanmax(values)))


def test_returning_to_a_zoom_reuses_pixels_within_budget(monkeypatch):
    import xarray as xr

    reads = count_reads(monkeypatch)
    layer = RasterLayer(create_mercator_data())
    whole = create_view((10240, 20480), (10240, 17067), (100, 150))
    detail = create_view((12800, 17920), (12800, 16213), (100, 150))
    first = layer.read_view_pixels(whole)
    layer.read_view_pixels(detail)
    returned = layer.read_view_pixels(whole)
    assert len(reads) == 2
    xr.testing.assert_identical(returned, first)


def test_zero_cache_budget_releases_windows_and_reads_again(monkeypatch):
    reads = count_reads(monkeypatch)
    layer = RasterLayer(create_mercator_data())
    view = create_view((10240, 20480), (10240, 17067), (100, 150))
    first = layer.read_view_pixels(view)
    layer.cache_bytes = 0
    assert layer.read_view_pixels(view) is first
    detail = create_view((12800, 17920), (12800, 16213), (100, 150))
    layer.read_view_pixels(detail)
    layer.read_view_pixels(view)
    assert len(reads) == 3
    layer.clear_cache()
    assert layer._cached_pixels is None


def test_identify_uses_declared_spatial_dimensions():
    scene = create_mercator_data().isel(y=slice(0, 2), x=slice(0, 2))
    scene = scene.rename(y="row", x="col")
    scene.row.attrs["axis"] = "Y"
    scene.col.attrs["axis"] = "X"
    layer = RasterLayer(scene)
    assert layer.identify(float(scene.col[1]), float(scene.row[0])) == {"value": "1.0"}


def test_multiband_item_viewport_uses_the_loaded_band_names(tmp_path):
    from datetime import UTC, datetime

    from geosave.geodata.io.formats.geotiff import write_cog
    from geosave.geodata.stac.catalog import create_catalog
    from geosave.geodata.stac.publish import create_stac_item
    from tests.geodata.conftest import build_raster

    source = build_raster()
    source.red.data = np.full_like(source.red.data, 2)
    source.nir.data = np.full_like(source.nir.data, 4)
    path = write_cog(source, tmp_path / "image.tif")
    entry = create_stac_item(
        {"image": path},
        id="scene",
        input_datetime=datetime(2025, 1, 1, tzinfo=UTC),
        with_proj=True,
        with_raster=True,
    )
    entry.assets["image"].extra_fields["bands"] = [{"name": "red"}, {"name": "nir"}]
    catalog = create_catalog([entry])
    row = catalog.iloc[0]
    scene = catalog.gs.to_dataset(source.gs.geobox)
    name = list(scene.data_vars)[1]
    layer = RasterLayer(scene, row=row, band=name)

    pixels = layer._read_pixels(source.gs.geobox)

    assert list(pixels.data_vars) == [name]
    assert pixels.gs.geobox == source.gs.geobox
    assert pixels[name].dims == ("y", "x")
    np.testing.assert_array_equal(pixels[name], 4)


def test_dataset_viewport_keeps_calculations_on_file_pixels(tmp_path):
    from geosave.geodata.io import open_raster
    from geosave.geodata.io.formats.geotiff import write_cog
    from tests.geodata.conftest import build_raster

    source = build_raster()[["red"]]
    source.red.data = np.full_like(source.red.data, 2)
    path = write_cog(source, tmp_path / "red.tif")
    with open_raster(path) as loaded:
        changed = loaded.assign(red=loaded.red * 3)
        layer = RasterLayer(changed)

        pixels = layer._read_pixels(source.gs.geobox)

    np.testing.assert_array_equal(pixels.red, 6)
