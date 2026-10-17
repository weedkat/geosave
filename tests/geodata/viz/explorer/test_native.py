"""Behavior required from the native Panel explorer replacement."""

import dask.array as da
import panel as pn
import pytest
from dask.callbacks import Callback

from geosave.geodata.viz.explorer import Explorer, RasterLayer


def test_native_layout_and_independent_raster_controls(scene, catalog):
    viewer = Explorer(scene, catalog, basemap="None")
    assert isinstance(viewer.map, pn.pane.HoloViews)
    assert not viewer.__panel__().select(pn.pane.IPyWidget)
    first = viewer.layers[0]
    assert first.display == "RGB"
    assert first.rgb == scene.gs.rgb
    first.display = "cover"
    viewer.table.selection = [0]
    opened = viewer.layers[-1]
    viewer.selected = opened
    viewer.selected = first
    assert first.display == "cover"
    viewer.table.selection = []
    assert opened not in viewer.layers
    viewer.close()


def test_explicit_selection_and_first_spatial_band(scene):
    assert RasterLayer(scene, "cover").display == "cover"
    rgb = RasterLayer(scene, rgb=("B02", "B03", "B04"))
    assert rgb.rgb == ("B02", "B03", "B04")
    plain = scene.copy()
    # Drop only declared display metadata using its owner API.
    from geosave.geodata.attrs import GDALBand, stamp_attrs

    for name in plain.data_vars:
        plain[name] = stamp_attrs(
            plain[name], GDALBand.replace(plain[name].attrs, colorinterp=None)
        )
    plain["scalar"] = 1
    first = RasterLayer(plain)
    assert first.display == "B04"
    with pytest.raises(ValueError):
        RasterLayer(scene, "cover", rgb=("B02", "B03", "B04"))
    with pytest.raises(KeyError):
        RasterLayer(scene, "missing")


def test_sidebar_selection_and_visibility_follow_layer_membership(scene, fields):
    viewer = Explorer(scene, fields, basemap="None")
    raster, vector = viewer.layers
    # Menu payloads contain strings and positions, never Python layer objects.
    menu = viewer._menus["Vectors"]
    menu.items = list(menu.items)
    menu.active = 0
    assert viewer.selected is vector
    viewer.selected = raster
    assert viewer._menus["Rasters"].active == (0,)
    raster.visible = False
    assert vector.visible
    assert viewer._menus["Rasters"].items[0]["actions"][0]["icon"] == "visibility_off"
    viewer.remove(raster)
    assert viewer.selected is vector
    assert viewer._menus["Vectors"].active == (0,)
    assert not viewer._menus["Rasters"].items
    viewer.close()


def test_opacity_does_not_execute_dask_or_read_other_rasters(scene):
    lazy = scene.chunk({"y": 16, "x": 16}) * 2
    viewer = Explorer(lazy, scene, basemap="None")
    viewer.map.get_root()
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        viewer.layers[0].alpha = 0.2
    assert tasks == []
    assert isinstance(lazy.B04.data, da.Array)
    viewer.close()


def test_band_rgb_switches_use_native_elements(scene):
    viewer = Explorer(scene, basemap="None")
    viewer.map.get_root()
    layer = viewer.layers[0]
    layer.display = "cover"
    assert layer.legend is not None
    layer.display = "RGB"
    viewer.map.get_root()
    viewer.close()


def test_computed_window_executes_only_intersecting_chunks():
    import numpy as np
    import xarray as xr
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords
    from geosave.geodata.viz.explorer.raster import read_window

    grid = GeoBox.from_bbox(
        (0, 0, 1024, 1024), crs=3857, shape=(1024, 1024), tight=True
    )
    values = da.random.default_rng(0).random((1024, 1024), chunks=(128, 128))
    data = xr.Dataset(
        {"value": xr.DataArray(values * 2, dims=("y", "x"), coords=xr_coords(grid))}
    )
    target = GeoBox.from_bbox((0, 0, 256, 256), crs=3857, shape=(64, 64), tight=True)
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(str(key))):
        sampled = read_window(data, target)
    assert sum("random" in key for key in tasks) == 4
    assert sampled.value.shape == (64, 64)
    assert np.isfinite(sampled.value).all()
    assert isinstance(data.value.data, da.Array)


def test_catalog_selection_after_render_and_disjoint_view(scene, catalog):
    viewer = Explorer(scene, catalog, basemap="None")
    viewer.map.get_root()
    viewer.table.selection = [0]
    viewer._ranges.event(x_range=(0, 1000), y_range=(0, 1000))
    viewer.table.selection = []
    assert len(viewer.layers) == 2
    viewer.close()


def test_catalog_and_band_changes_keep_the_map_figure(scene, catalog):
    from bokeh.models import Plot

    viewer = Explorer(scene, catalog, basemap="None")
    root = viewer.map.get_root()
    figure = root.select_one({"type": Plot})
    assert figure.match_aspect
    viewer.table.selection = [0]
    assert root.select_one({"type": Plot}) is figure
    viewer.layers[0].display = "B04"
    assert root.select_one({"type": Plot}) is figure
    viewer.table.selection = []
    assert root.select_one({"type": Plot}) is figure
    viewer.close()


def test_catalog_filter_sort_pagination_keep_row_identity(catalog):
    viewer = Explorer(catalog, basemap="None")
    table = viewer.table
    table.page_size = 1
    table.sorters = [{"field": "id", "dir": "desc"}]
    table.page = 2
    table.filters = [{"field": "id", "type": "=", "value": catalog.iloc[2]["id"]}]
    table.selection = [2]
    assert viewer.layers[-1].name.startswith(catalog.iloc[2]["id"])
    table.filters = []
    assert table.selection == [2]
    table.selection = []
    assert len(viewer.layers) == 1
    viewer.close()


def test_opacity_and_vector_style_do_not_compute_other_rasters(scene, fields):
    viewer = Explorer(scene.chunk({"y": 16, "x": 16}) * 2, fields, basemap="None")
    viewer.map.get_root()
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        viewer.layers[1].column = "area"
        viewer.layers[0].visible = False
        viewer.layers[0].alpha = 0.2
    assert tasks == []
    viewer.close()


def test_static_and_explorer_backends_in_both_orders(scene):
    import holoviews as hv

    for static_first in [True, False]:
        if static_first:
            figure = scene.gs.plot("B04")
            assert hv.render(figure, backend="matplotlib") is not None
        viewer = scene.gs.explore(basemap="None")
        viewer.map.get_root()
        figure = scene.gs.plot("B04")
        assert hv.render(figure, backend="matplotlib") is not None
        viewer.layers[0].display = "B04"
        viewer.map.get_root()
        viewer.close()


def test_nodata_and_decoded_cog_round_trip(scene, tmp_path):
    import numpy as np
    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    data = scene[["B04"]].copy()
    data["B04"].values[0, 0] = -999
    data = data.gs.write_nodata(-999)
    path = tmp_path / "packed.tif"
    data.gs.to_cog(path)
    loaded = open_raster(path)
    result = read_window(loaded, loaded.odc.geobox)
    assert np.isnan(result.B04.values[0, 0])
    np.testing.assert_allclose(result.B04.values[1:], data.B04.values[1:])


def test_derived_file_values_are_used_and_source_rgb_is_unchanged(scene, tmp_path):
    import numpy as np
    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    path = tmp_path / "rgb.tif"
    scene[["B04", "B03", "B02"]].gs.to_cog(path)
    loaded = open_raster(path)
    derived = loaded * 2
    sampled = read_window(derived, derived.odc.geobox)
    np.testing.assert_allclose(sampled.B04, loaded.B04.compute() * 2)
    layer = RasterLayer(loaded)
    layer.display = "B03"
    assert loaded.gs.rgb == ("B04", "B03", "B02")


def test_changed_values_with_source_encoding_are_authoritative(scene, tmp_path):
    import numpy as np
    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    path = tmp_path / "values.tif"
    scene[["B04"]].gs.to_cog(path)
    loaded = open_raster(path)
    changed = loaded.copy()
    changed["B04"] = loaded.B04.copy(data=loaded.B04.data * 3)
    assert "source" in changed.B04.encoding
    sampled = read_window(changed, changed.odc.geobox)
    np.testing.assert_allclose(sampled.B04, loaded.B04.compute() * 3)


@pytest.mark.parametrize("materialize", ["load", "persist"])
def test_loaded_file_values_do_not_need_the_original_file(scene, tmp_path, materialize):
    import numpy as np
    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    path = tmp_path / "values.tif"
    scene[["B04"]].gs.to_cog(path)
    loaded = getattr(open_raster(path), materialize)()
    loaded.close()
    path.unlink()
    sampled = read_window(loaded, loaded.odc.geobox)
    np.testing.assert_allclose(sampled.B04, loaded.B04)


def test_current_nodata_metadata_overrides_file_nodata(scene, tmp_path):
    import numpy as np
    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    data = scene[["B04"]].copy()
    data.B04.values[0, 0] = -999
    data = data.gs.write_nodata(-999)
    path = tmp_path / "values.tif"
    data.gs.to_cog(path)
    unmasked = open_raster(path).gs.write_nodata(None)
    sampled = read_window(unmasked, unmasked.odc.geobox)
    np.testing.assert_allclose(sampled.B04, unmasked.B04.compute())


def test_decoded_cog_uses_overviews_without_dask_execution(scene, tmp_path):
    import numpy as np
    from geosave.geodata.io import open_raster
    from geosave.geodata.viz.explorer.raster import read_window

    data = scene[["B04"]].copy()
    data.B04.values[0, 0] = -999
    data = data.gs.write_nodata(-999).gs.write_packing(0.5, 2)
    path = tmp_path / "packed.tif"
    data.gs.to_cog(path)
    decoded = open_raster(path, mask_and_scale=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        sampled = read_window(decoded, decoded.odc.geobox)
    np.testing.assert_allclose(sampled.B04.values[1:], data.B04.values[1:] * 0.5 + 2)
    np.testing.assert_allclose(sampled.B04, decoded.B04.compute())
    assert tasks == []


def test_spatial_axis_order_is_normalized_for_display(scene):
    import numpy as np
    from geosave.geodata.viz.explorer.raster import read_window

    transposed = scene[["B04"]].transpose("x", "y")
    sampled = read_window(transposed, transposed.odc.geobox)
    np.testing.assert_allclose(sampled.B04, scene.B04)


def test_composed_catalog_retains_ownership_and_independent_settings(scene, catalog):
    viewer = Explorer(catalog, basemap="None")
    viewer.table.selection = [0]
    opened = viewer.layers[-1]
    opened.alpha = 0.4
    composed = viewer * scene
    copied = composed.layers[1]
    assert copied is not opened and copied.alpha == 0.4
    composed.table.selection = []
    assert copied not in composed.layers
    assert opened in viewer.layers
    assert viewer.table.selection == [0]
    composed.close()
    viewer.close()


def test_removed_raster_does_not_execute_on_later_controls_or_view(scene, fields):
    layer = RasterLayer(scene.chunk({"y": 16, "x": 16}) * 2)
    viewer = Explorer(layer, fields, basemap="None")
    viewer.map.get_root()
    viewer.remove(layer)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        layer.display = "B03"
        viewer._ranges.event(x_range=(0, 1000), y_range=(0, 1000))
    assert tasks == []
    viewer.close()


def test_catalog_open_failure_preserves_selection_and_existing_layers(
    catalog, fields, monkeypatch
):
    viewer = Explorer(catalog, fields, basemap="None")

    def fail(*args, **kwargs):
        raise OSError("unavailable file")

    # The native row reader owns the error; the explorer keeps its last successful selection.
    monkeypatch.setattr(type(catalog.iloc[0].gs), "to_xarray", fail)
    viewer.table.selection = [0]
    assert len(viewer.layers) == 2
    assert viewer.table.selection == []
    assert "unavailable file" in viewer.status.object
    viewer.close()


def test_class_legend_without_colors_is_visible(scene):
    from geosave.geodata.attrs import Legend, stamp_attrs

    data = scene[["cover"]].copy()
    data["cover"] = stamp_attrs(data.cover, Legend(classes={0: "water", 1: "trees"}))
    layer = RasterLayer(data)
    assert set(layer.colors) == {0, 1}
    assert data.cover.gs.legend.color_map is None
    plot = layer.render(None, None).values()[0]
    assert plot.dimension_values(plot.vdims[-1]).max() == 1


def test_date_filters_keep_checked_row_identity_and_raster_time(monthly, catalog):
    from datetime import date

    viewer = Explorer(monthly, catalog, basemap="None")
    raster, vector = viewer.layers
    viewer.table.selection = [1]
    opened = viewer.layers[-1]
    vector.period = (date(2024, 4, 1), date(2024, 4, 30))
    assert vector.rows == [1]
    assert viewer.table.selection == [1]
    raster.time = monthly.time.values[-1]
    assert raster.select().B04.equals(monthly.B04.isel(time=-1))
    vector.period = (date(2024, 5, 1), date(2024, 5, 5))
    assert vector.rows == [2]
    assert opened in viewer.layers
    viewer.table.selection = []
    assert opened not in viewer.layers
    viewer.close()


def test_comparison_and_duplicate_names_use_unambiguous_choices(scene):
    layers = [
        RasterLayer(scene, name="raster"),
        RasterLayer(scene, name="raster (2)"),
        RasterLayer(scene, name="raster"),
    ]
    viewer = Explorer(*layers, basemap="None")
    assert len(viewer._menus["Rasters"].items) == 3
    viewer.map.get_root()
    viewer.compare = True
    viewer.side = "right"
    assert viewer.right == "raster (3)"
    viewer.remove(layers[-1])
    assert viewer.right in viewer.param.right.objects
    viewer.close()


def test_map_identification_reads_values_and_selects_vector(scene, fields):
    viewer = Explorer(scene, fields, basemap="None")
    vector = viewer.layers[-1]
    point = vector.frame.geometry.iloc[0].centroid
    viewer._identify(x=point.x, y=point.y)
    assert vector.selected == [0]
    assert "rice" in viewer._identified.object
    assert "B04" in viewer._identified.object
    viewer.close()
