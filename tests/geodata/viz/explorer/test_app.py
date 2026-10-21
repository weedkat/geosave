from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import panel as pn
import panel_material_ui as mui
import pytest
from bokeh.models import HoverTool
from dask.callbacks import Callback
from panel.util.parameters import edit_readonly

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


def test_raster_style_controls_reuse_computed_pixels(scene):
    data = scene.chunk({"y": 16, "x": 16}) * 2
    layer = RasterLayer(data, "B04")
    viewer = Explorer(layer, basemap="None")
    viewer.map.get_root()
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        layer.alpha = 0.4
        layer.cmap = "RdYlGn"
        layer.stretch = (5.0, 95.0)
        layer.clim = (0.2, 1.2)
    assert tasks == []
    assert isinstance(data.B04.data, da.Array)
    viewer.close()


@pytest.mark.parametrize("action", ["opacity", "pan", "uncached pan"])
def test_rgb_resends_pixels_only_for_uncached_views(scene, action):
    from bokeh.document import Document
    from bokeh.models import GlyphRenderer, ImageRGBA

    viewer = Explorer(scene, basemap="None")
    document = Document()
    root = viewer.map.get_root(document)
    document.add_root(root)
    images = {
        renderer.data_source
        for renderer in root.select({"type": GlyphRenderer})
        if isinstance(renderer.glyph, ImageRGBA)
    }
    updates = []

    def record(event):
        if event.model in images and getattr(event, "attr", None) == "data":
            updates.append(event)

    document.on_change(record)
    if action == "opacity":
        viewer.layers[0].alpha = 0.4
    else:
        left, right = viewer._ranges.x_range
        bottom, top = viewer._ranges.y_range
        distance = 0.25 if action == "uncached pan" else 0.01
        dx, dy = (right - left) * distance, (top - bottom) * distance
        viewer._ranges.event(
            x_range=(left + dx, right + dx), y_range=(bottom + dy, top + dy)
        )
    assert images
    assert bool(updates) == (action == "uncached pan")
    if action == "opacity":
        assert any(
            renderer.glyph.global_alpha == 0.4
            for renderer in root.select({"type": GlyphRenderer})
            if isinstance(renderer.glyph, ImageRGBA)
        )
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


def test_inline_page_does_not_echo_global_busy_state(scene):
    viewer = Explorer(scene, basemap="None")
    busy = pn.state.busy
    try:
        with edit_readonly(pn.state):
            pn.state.busy = True
        assert viewer.__panel__().busy is False
    finally:
        with edit_readonly(pn.state):
            pn.state.busy = busy
        viewer.close()


def test_sidebar_separates_catalogs_rasters_and_vectors(scene, catalog, fields):
    viewer = Explorer(catalog, scene, fields, basemap="None")
    viewer.__panel__().get_root()
    assert set(viewer._menus) == {"Catalogs", "Rasters", "Vectors"}
    assert all(isinstance(menu, mui.MenuList) for menu in viewer._menus.values())
    for group, layer in zip(("Catalogs", "Rasters", "Vectors"), viewer.layers):
        menu = viewer._menus[group]
        assert len(menu.items) == 1
        # The notebook bridge echoes only native serialized MenuList keys.
        menu.items = [
            {key: value for key, value in item.items() if key in menu._item_keys}
            for item in menu.items
        ]
        menu._handle_msg({"type": "click", "item": [0]})
        assert viewer.selected is layer
    viewer.close()


def test_display_is_one_selector_with_declared_rgb_and_bands(scene):
    viewer = Explorer(scene, basemap="None")
    layer = viewer.layers[0]
    assert layer.rgb == scene.gs.rgb
    assert layer.display == "RGB"
    selects = viewer._controls.select(mui.Select)
    display = next(widget for widget in selects if widget.label == "Display")
    assert list(display.options) == ["RGB", *layer.names]
    display.value = "cover"
    assert layer.display == "cover"
    assert not any(
        widget.label in {"Red", "Green", "Blue", "Band"} for widget in selects
    )
    viewer.close()


def test_class_controls_and_legend_match_the_selected_display(scene):
    viewer = Explorer(RasterLayer(scene, "cover"), basemap="None")
    assert "water" in viewer._legend.object and "trees" in viewer._legend.object
    assert not any(
        widget.visible for widget in viewer._controls.select(mui.RangeSlider)
    )
    assert not any(
        widget.visible and widget.label == "Colormap"
        for widget in viewer._controls.select(mui.Select)
    )
    viewer.layers[0].display = "B04"
    assert any(widget.visible for widget in viewer._controls.select(mui.RangeSlider))
    assert "water" not in viewer._legend.object
    viewer.close()


def test_new_renderers_never_inherit_hover_fields_they_do_not_have(scene, catalog):
    viewer = Explorer(scene, catalog, basemap="None")
    root = viewer.map.get_root()
    for selected in ([0], [], [0]):
        viewer.table.selection = selected
        for tool in root.select({"type": HoverTool}):
            fields = set()
            for _, expression in tool.tooltips:
                fields.update(re.findall(r"@\{([^}]+)\}", expression))
            for renderer in tool.renderers:
                assert fields <= set(renderer.data_source.data)
    viewer.close()


def test_hidden_raster_does_not_execute_dask_on_pan(scene, fields):
    raster = RasterLayer(scene.chunk({"x": 16, "y": 16}) * 2)
    raster.visible = False
    viewer = Explorer(fields, raster, basemap="None")
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        viewer.map.get_root()
        box = raster.bbox
        viewer.zoom(box)
    assert tasks == []
    viewer.close()


def test_switching_tables_keeps_their_models_in_the_notebook_document(catalog, fields):
    from panel.models.tabulator import DataTabulator

    viewer = Explorer(catalog, fields, basemap="None")
    root = viewer.__panel__().get_root()
    tables = set(root.select({"type": DataTabulator}))
    assert len(tables) == 2
    viewer.selected = viewer.layers[1]
    assert tables <= root.references()
    viewer.selected = viewer.layers[0]
    assert tables <= root.references()
    viewer.close()


def test_composition_retains_explicit_rgb_without_declared_rgb(scene):
    data = scene[["B04", "cover"]]
    assert data.gs.rgb is None
    viewer = Explorer(RasterLayer(data, rgb=("B04", "B04", "cover")), basemap="None")
    composed = viewer * scene
    assert composed.layers[0].is_rgb
    assert composed.layers[0].rgb == ("B04", "B04", "cover")
    assert composed.layers[0].select().data_vars.keys() == data.data_vars.keys()
    composed.close()
    viewer.close()


def test_two_raster_explorers_compose_without_catalog_or_table(scene):
    left = scene.gs.explore("B04", basemap="None")
    right = scene.gs.explore("cover", basemap="None")
    composed = left * right
    composed.__panel__().get_root()
    assert len(composed.layers) == 2
    assert len(composed._menus["Rasters"].items) == 2
    assert composed._menus["Catalogs"].items == []
    assert composed._menus["Vectors"].items == []
    assert composed.table is None
    assert not composed._catalog.visible
    assert [layer.display for layer in composed.layers] == ["B04", "cover"]
    composed.layers[1].display = "B03"
    assert right.layers[0].display == "cover"
    composed.close()
    left.close()
    right.close()


def test_right_explorer_retains_catalog_ownership(scene, catalog):
    left = scene.gs.explore(basemap="None")
    right = catalog.gs.explore(basemap="None")
    right.table.selection = [0]
    composed = left * right
    assert len(composed.layers) == 3
    assert len(composed._menus["Catalogs"].items) == 1
    assert len(composed._menus["Rasters"].items) == 2
    assert composed._catalog.visible
    composed.table.selection = []
    assert len(composed.layers) == 2
    assert len(right.layers) == 2
    assert right.table.selection == [0]
    composed.close()
    left.close()
    right.close()


def test_repeated_explorer_keeps_each_catalogs_opened_rasters(catalog):
    viewer = catalog.gs.explore(basemap="None")
    viewer.table.selection = [0]
    composed = viewer * viewer
    assert len(composed.layers) == 4
    composed.layers[0].table.selection = []
    assert len(composed.layers) == 3
    composed.layers[1].table.selection = []
    assert len(composed.layers) == 2
    assert len(viewer.layers) == 2
    composed.close()
    viewer.close()


def test_zoom_updates_the_rendered_map_ranges(scene, fields):
    from bokeh.models import Plot

    from geosave.geodata.viz.map import frame_bbox

    viewer = Explorer(scene, fields, basemap="None")
    root = viewer.map.get_root()
    figure = root.select_one({"type": Plot})
    extent = frame_bbox(viewer.layers[1].bbox)
    viewer._zoom_item({"label": viewer.layers[1].name})
    assert (figure.x_range.start, figure.x_range.end) == pytest.approx(
        (extent.left, extent.right), rel=0, abs=1e-6
    )
    assert (figure.y_range.start, figure.y_range.end) == pytest.approx(
        (extent.bottom, extent.top), rel=0, abs=1e-6
    )
    assert root.select_one({"type": Plot}) is figure
    viewer.close()


@pytest.mark.parametrize("display", ["RGB", "B04"])
def test_contrast_changes_reuse_visible_pixels(scene, display):
    import numpy as np
    from bokeh.models import GlyphRenderer

    viewer = Explorer(scene.chunk({"x": 16, "y": 16}) * 2, basemap="None")
    layer = viewer.layers[0]
    layer.display = display
    root = viewer.map.get_root()
    renderer = next(
        value
        for value in root.select({"type": GlyphRenderer})
        if "image" in value.data_source.data
    )
    pixels = renderer.data_source.data["image"][0].copy()
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        layer.stretch = (10, 90)
        layer.clim = (0.2, 0.8)
    assert tasks == []
    if display == "RGB":
        assert not np.array_equal(pixels, renderer.data_source.data["image"][0])
    else:
        assert renderer.glyph.color_mapper.low == 0.2
        assert renderer.glyph.color_mapper.high == 0.8
    viewer.close()


def test_zoom_preserves_map_aspect_and_later_style_changes(scene, fields):
    from bokeh.models import Plot

    viewer = Explorer(scene, fields, basemap="None")
    root = viewer.map.get_root()
    figure = root.select_one({"type": Plot})
    viewer._size.event(width=1000, height=500)
    viewer.zoom(viewer.layers[1].bbox)
    bounds = (
        figure.x_range.start,
        figure.y_range.start,
        figure.x_range.end,
        figure.y_range.end,
    )
    assert (bounds[2] - bounds[0]) / (bounds[3] - bounds[1]) == pytest.approx(2)
    viewer.layers[0].alpha = 0.4
    assert bounds == (
        figure.x_range.start,
        figure.y_range.start,
        figure.x_range.end,
        figure.y_range.end,
    )
    viewer.close()


def test_zoom_survives_a_deferred_server_refresh(scene, fields):
    import weakref

    from bokeh.document import Document
    from bokeh.models import Plot
    from panel.io.state import set_curdoc

    from geosave.geodata.viz.map import frame_bbox

    viewer = Explorer(scene, fields, basemap="None")
    document = Document()
    root = viewer.map.get_root(document)
    figure = root.select_one({"type": Plot})

    class Session:
        pass

    session = Session()
    document._session_context = weakref.ref(session)
    extent = frame_bbox(viewer.layers[1].bbox)
    viewer.zoom(viewer.layers[1].bbox)
    assert document.session_callbacks
    with set_curdoc(document):
        for callback in list(document.session_callbacks):
            document.remove_next_tick_callback(callback)
            callback.callback()
    assert (figure.x_range.start, figure.x_range.end) == pytest.approx(
        (extent.left, extent.right), rel=0, abs=1e-6
    )
    viewer.close()


def test_initial_map_size_preserves_spatial_aspect(scene):
    from bokeh.models import Plot

    viewer = Explorer(scene, basemap="None")
    root = viewer.map.get_root()
    figure = root.select_one({"type": Plot})
    viewer._size.event(width=1000, height=500)
    width = figure.x_range.end - figure.x_range.start
    height = figure.y_range.end - figure.y_range.start
    assert width / height == pytest.approx(2)
    bounds = (
        figure.x_range.start,
        figure.x_range.end,
        figure.y_range.start,
        figure.y_range.end,
    )
    viewer._size.event(width=500, height=500)
    viewer._size.event(width=1000, height=500)
    assert (
        figure.x_range.start,
        figure.x_range.end,
        figure.y_range.start,
        figure.y_range.end,
    ) == pytest.approx(bounds)
    viewer.close()


def test_repeated_plot_size_does_not_redraw_the_map(scene):
    viewer = Explorer(scene, basemap="None")
    viewer.map.get_root()
    viewer._size.event(width=1000, height=500)
    tick = viewer._refresh.tick
    viewer._size.event(width=1000, height=500)
    assert viewer._refresh.tick == tick
    viewer.close()


def test_catalog_hover_follows_cursor_and_omits_bookkeeping(catalog):
    catalog = catalog.assign(collection="Sentinel-2", split="train")
    viewer = Explorer(catalog, basemap="None")
    root = viewer.map.get_root()
    hover = next(tool for tool in root.select({"type": HoverTool}) if tool.tooltips)
    fields = dict(hover.tooltips)
    assert hover.point_policy == "follow_mouse"
    assert hover.line_policy == "none"
    assert hover.limit == 1
    assert not hover.show_arrow
    assert set(fields.values()) == {"@{collection}", "@{datetime}", "@{split}"}
    viewer.close()


@pytest.mark.parametrize("notebook", [False, True])
@pytest.mark.parametrize("threaded", [None, False, True])
def test_serve_chooses_threading_for_the_active_environment(
    scene, monkeypatch, notebook, threaded
):
    from IPython.core import getipython

    monkeypatch.setattr(
        getipython,
        "get_ipython",
        lambda: SimpleNamespace(kernel=object()) if notebook else None,
    )
    settings = {}
    handle = object()

    def serve(panel, **kwargs):
        settings.update(kwargs)
        return handle

    monkeypatch.setattr(pn, "serve", serve)
    viewer = Explorer(scene, basemap="None")
    try:
        assert viewer.serve(threaded=threaded) is handle
        assert settings["threaded"] is (notebook if threaded is None else threaded)
        assert settings["address"] == "localhost"
        assert settings["websocket_origin"] == ["localhost:5006", "127.0.0.1:5006"]
    finally:
        viewer.close()


def test_colab_serve_uses_the_proxy_origin_and_iframe(scene, monkeypatch):
    from IPython.core import getipython

    shell = type(
        "Shell", (), {"__module__": "google.colab._shell", "kernel": object()}
    )()
    monkeypatch.setattr(getipython, "get_ipython", lambda: shell)
    displayed = []
    output = SimpleNamespace(
        eval_js=lambda code: "https://session-5006.colab.googleusercontent.com/",
        serve_kernel_port_as_iframe=lambda port, **kwargs: displayed.append(port),
    )
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    settings = {}
    handle = object()

    def serve(panel, **kwargs):
        settings.update(kwargs)
        assert not displayed, "start the server before displaying its iframe"
        return handle

    monkeypatch.setattr(pn, "serve", serve)
    viewer = Explorer(scene, basemap="None")
    try:
        assert viewer.serve() is handle
        assert settings["threaded"] is True
        assert settings["show"] is False
        assert (
            "session-5006.colab.googleusercontent.com" in settings["websocket_origin"]
        )
        assert displayed == [5006]
    finally:
        viewer.close()


@pytest.mark.slow
def test_script_serve_accepts_both_local_browser_origins(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    root = Path(__file__).resolve().parents[4]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = tmp_path / "launch.py"
    app.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from scripts.smoke_explorer import create_explorer\n"
        "viewer = create_explorer()\n"
        f"viewer.serve(port={port}, open=False)\n"
        "viewer.close()\n"
        "print('SERVER_STOPPED', flush=True)\n"
    )
    log_path = tmp_path / "launch.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [sys.executable, str(app)], cwd=root, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert server.poll() is None, log_path.read_text()
                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.1)
            with playwright.sync_playwright() as runtime:
                browser = runtime.chromium.launch(
                    executable_path=os.environ.get("GEOSAVE_BROWSER"),
                    args=["--no-sandbox"],
                )
                for host in ["localhost", "127.0.0.1"]:
                    page = browser.new_page()
                    page.goto(f"http://{host}:{port}/")
                    page.wait_for_function(
                        "window.Bokeh && Bokeh.documents.length > 0 && "
                        "Bokeh.documents.every(doc => doc.is_idle)",
                        timeout=30000,
                    )
                    wait_for_raster(page)
                    page.close()
                browser.close()
            assert "SERVER_STOPPED" not in log_path.read_text()
            server.send_signal(signal.SIGINT)
            server.wait(timeout=15)
            assert server.returncode == 0, log_path.read_text()
            assert "SERVER_STOPPED" in log_path.read_text()
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=15)
    assert "Refusing websocket" not in log_path.read_text()


def wait_for_raster(page):
    """Wait for colored raster pixels, including canvases in shadow roots."""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if page.locator("canvas").evaluate_all("""canvases => canvases.some(canvas => {
            const context = canvas.getContext('2d');
            if (!context || canvas.width < 100 || canvas.height < 100) return false;
            const {data} = context.getImageData(0, 0, canvas.width, canvas.height);
            let colored = 0;
            for (let i = 0; i < data.length; i += 400) {
                if (data[i+3] && Math.max(data[i], data[i+1], data[i+2]) - Math.min(data[i], data[i+1], data[i+2]) > 30) colored++;
            }
            return colored > 100;
        })"""):
            return
        time.sleep(0.05)
    pytest.fail("the map has no colored raster pixels")


@pytest.mark.slow
def test_native_catalog_rgb_pan_and_style(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    root = Path(__file__).resolve().parents[4]
    catalog_app = tmp_path / "catalog_only.py"
    catalog_app.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from scripts.smoke_explorer import create_explorer\n"
        "from geosave.geodata.viz.explorer import Explorer\n"
        "proof = create_explorer()\n"
        "viewer = Explorer(proof.layers[0].data, basemap='None', height=430)\n"
        "viewer.servable()\n"
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (tmp_path / "server.log").open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "panel",
                "serve",
                "scripts/smoke_explorer.py",
                str(catalog_app),
                "--port",
                str(port),
                "--address",
                "127.0.0.1",
            ],
            cwd=root,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.1)
            with playwright.sync_playwright() as runtime:
                browser = runtime.chromium.launch(
                    executable_path=os.environ.get("GEOSAVE_BROWSER"),
                    args=["--no-sandbox"],
                )
                page = browser.new_page(viewport={"width": 1400, "height": 1100})
                errors = []
                page.on(
                    "pageerror",
                    lambda error: (
                        errors.append(str(error)),
                        print("BROWSER", str(error)),
                    ),
                )
                timings = {}
                page.goto(f"http://localhost:{port}/catalog_only")
                check = page.locator(".tabulator-row input[type=checkbox]").first
                check.wait_for(timeout=60000)
                canvas = page.locator("canvas").first
                canvas.wait_for()
                bounds = canvas.bounding_box()
                x = bounds["x"] + bounds["width"] / 2
                y = bounds["y"] + bounds["height"] / 2
                page.mouse.move(x, y)
                tooltip = page.locator(".bk-Tooltip:visible, .bk-tooltip:visible")
                tooltip.wait_for()
                before = tooltip.bounding_box()
                assert before["width"] < 350 and before["height"] < 160
                page.mouse.move(x + 30, y + 20)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    after = tooltip.bounding_box()
                    if after and abs(after["x"] - before["x"]) > 10:
                        break
                    page.wait_for_timeout(50)
                assert abs(after["x"] - before["x"]) > 10
                assert tooltip.count() == 1
                page.screenshot(path=str(tmp_path / "catalog-hover.png"))
                page.mouse.move(20, 20)
                for _ in range(2):
                    check.check()
                    page.wait_for_function("""() => Bokeh.documents.some(doc =>
                        [...doc._all_models.values()].some(model =>
                            model.type === 'GlyphRenderer' && model.visible
                            && model.glyph?.type === 'ImageRGBA'
                            && model.data_source.data.image?.[0]?.some(pixel => (pixel >>> 24) > 0)
                        ))""")
                    wait_for_raster(page)
                    check.uncheck()
                    page.wait_for_function("""() => Bokeh.documents.every(doc =>
                        [...doc._all_models.values()].every(model =>
                            model.type !== 'GlyphRenderer' || !model.visible
                            || model.glyph?.type !== 'ImageRGBA'
                        ))""")
                start = time.perf_counter()
                page.goto(f"http://localhost:{port}/smoke_explorer")
                check = page.locator(".tabulator-row input[type=checkbox]").first
                check.wait_for(timeout=60000)
                page.locator("canvas").first.wait_for()
                lazy = page.locator(".MuiListItemButton-root").filter(
                    has=page.get_by_text("Lazy xarray", exact=True)
                )
                playwright.expect(
                    lazy.get_by_role("button", name="Visibility", exact=True)
                ).to_be_visible()
                wait_for_raster(page)
                map_box = page.locator("canvas").first.bounding_box()
                assert 400 <= map_box["height"] <= 470
                assert (
                    page.locator(".tabulator:visible").bounding_box()["y"]
                    > map_box["y"] + map_box["height"]
                )
                timings["cold_seconds"] = time.perf_counter() - start
                basemap = page.get_by_role("combobox", name="Basemap", exact=True)
                assert basemap.bounding_box()["y"] < lazy.bounding_box()["y"]
                assert page.locator("header").bounding_box()["height"] <= 52

                def select(label, value):
                    widget = page.get_by_role("combobox", name=label, exact=True)
                    widget.click()
                    page.get_by_role("menuitem", name=value, exact=True).click()
                    playwright.expect(widget).to_contain_text(value)

                def choose(row):
                    title = row.locator(".MuiListItemText-primary")
                    playwright.expect(title).to_be_visible()
                    assert title.bounding_box()["width"] >= 100
                    title.click()
                    title = row.locator(".MuiListItemText-primary").inner_text()
                    page.get_by_role("heading", name=title, exact=True).wait_for()

                for cycle in range(3):
                    start = time.perf_counter()
                    check.check()
                    cog = page.get_by_role("button", name=re.compile("^COG ·"))
                    cog.wait_for(timeout=15000)
                    timings[f"open_{cycle}_seconds"] = time.perf_counter() - start
                    choose(cog)
                    display = page.get_by_role("combobox", name="Display", exact=True)
                    playwright.expect(display).to_contain_text("RGB")
                    select("Display", "green")
                    choose(lazy)
                    playwright.expect(display).to_contain_text("RGB")
                    choose(cog)
                    playwright.expect(display).to_contain_text("green")
                    select("Display", "RGB")
                    assert page.get_by_label("Red", exact=True).count() == 0
                    start = time.perf_counter()
                    check.uncheck()
                    cog.wait_for(state="detached")
                    timings[f"remove_{cycle}_seconds"] = time.perf_counter() - start

                vector = page.locator(".MuiListItemButton-root").filter(
                    has=page.get_by_text("vector", exact=True)
                )
                start = time.perf_counter()
                vector.get_by_role("button", name="Zoom to layer", exact=True).click()
                page.wait_for_function(
                    """() => Bokeh.documents.some(doc =>
                    [...doc._all_models.values()].some(model => {
                        if (!model.x_range || !model.y_range || !model.renderers) return false;
                        const x = model.x_range, y = model.y_range;
                        return Math.abs((x.start+x.end)/2 + 965000) < 5
                            && Math.abs((y.start+y.end)/2 - 5035000) < 5
                            && y.end-y.start < 80000
                            && Math.abs((x.end-x.start)/(y.end-y.start)
                                - model.inner_width/model.inner_height) < 0.02;
                    }))""",
                    timeout=15000,
                )
                timings["zoom_to_layer_seconds"] = time.perf_counter() - start
                page.evaluate("""() => {
                    window.rasterPatches = 0;
                    for (const doc of Bokeh.documents) {
                        doc.on_change(event => {
                            if (event.kind === 'ModelChanged' && event.attr === 'data' && event.model.data && 'image' in event.model.data) {
                                window.rasterPatches++;
                            }
                        });
                    }
                }""")
                start = time.perf_counter()
                canvas = page.locator("canvas").first
                rect = canvas.bounding_box()
                x, y = rect["x"] + 200, rect["y"] + 150
                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.move(x + 50, y + 30, steps=8)
                page.mouse.up()
                page.mouse.wheel(0, -300)
                page.wait_for_function("window.rasterPatches > 0", timeout=30000)
                timings["pan_zoom_raster_patch_seconds"] = time.perf_counter() - start
                choose(
                    page.locator(".MuiListItemButton-root").filter(
                        has=page.get_by_text("vector", exact=True)
                    )
                )
                start = time.perf_counter()
                select("Color by", "area")
                page.get_by_text("2 — 5", exact=True).wait_for()
                timings["vector_color_seconds"] = time.perf_counter() - start
                wait_for_raster(page)

                def check_theme(dark):
                    control = page.get_by_role("combobox", name="Color by", exact=True)
                    expected = "rgb(255, 255, 255)" if dark else "rgba(0, 0, 0, 0.87)"
                    playwright.expect(control).to_have_css("color", expected)
                    playwright.expect(page.locator("header .title")).to_have_css(
                        "color", expected
                    )
                    assert page.locator("header").evaluate(
                        "e => getComputedStyle(e).backgroundColor"
                    ) == page.locator(".sidebar").evaluate(
                        "e => getComputedStyle(e).backgroundColor"
                    )
                    playwright.expect(
                        page.locator(".tabulator-cell:visible").first
                    ).to_have_css("color", expected)
                    playwright.expect(
                        page.get_by_text("2 — 5", exact=True)
                    ).to_have_css("color", expected)
                    wait_for_raster(page)

                check_theme(False)
                page.screenshot(path=str(tmp_path / "explorer.png"))
                page.get_by_role("button", name="Toggle theme").click()
                check_theme(True)
                page.screenshot(path=str(tmp_path / "dark.png"))
                classified = page.locator(".MuiListItemButton-root").filter(
                    has=page.get_by_text("Land cover", exact=True)
                )
                choose(classified)
                classified.get_by_role("button", name="Visibility", exact=True).click()
                page.wait_for_function(
                    """() => Bokeh.documents.some(doc =>
                    [...doc._all_models.values()].some(model => {
                        if (model.type !== 'GlyphRenderer' || !model.visible || model.glyph?.type !== 'ImageRGBA') return false;
                        const image = model.data_source.data.image?.[0];
                        if (!image?.length) return false;
                        const colors = new Set();
                        for (let i=0; i<image.length; i+=400) {
                            colors.add(image[i]);
                            if (colors.size > 4) return false;
                        }
                        return colors.size >= 2;
                    }))""",
                    timeout=30000,
                )
                page.get_by_text("Water", exact=True).wait_for()
                page.get_by_text("Trees", exact=True).wait_for()
                page.get_by_text("Crops", exact=True).wait_for()
                assert (
                    page.get_by_role("combobox", name="Colormap", exact=True).count()
                    == 0
                )
                page.screenshot(path=str(tmp_path / "categorical-dark.png"))
                page.get_by_role("tab", name="Info", exact=True).click()
                page.get_by_text("EPSG:3857").wait_for()
                page.get_by_role("button", name="Toggle theme").click()
                page.screenshot(path=str(tmp_path / "info-light.png"))
                page.set_viewport_size({"width": 700, "height": 1100})
                page.wait_for_function("""() => Bokeh.documents.every(doc =>
                    doc.is_idle && [...doc._all_models.values()].every(model =>
                        !(model.properties.busy && model.busy)))
                    && Bokeh.documents.some(doc => [...doc._all_models.values()].some(model =>
                        model.type === 'Figure'
                        && Math.abs((model.x_range.end-model.x_range.start)
                            / (model.y_range.end-model.y_range.start)
                            - model.inner_width/model.inner_height) < 0.02
                    ))""")
                page.screenshot(path=str(tmp_path / "narrow.png"))
                (tmp_path / "timings.json").write_text(json.dumps(timings, indent=2))
                print("TIMINGS", json.dumps(timings), flush=True)
                assert errors == []
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=15)
    assert "Traceback" not in (tmp_path / "server.log").read_text()


@pytest.mark.slow
@pytest.mark.parametrize("static_first", [False, True])
def test_native_notebook_comm(static_first):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from jupyter_client import KernelManager

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("""
import sys
import os
os.environ.pop('VSCODE_CWD', None)
os.environ.pop('VSCODE_PID', None)
import geopandas as gpd
from shapely.geometry import Point
import geosave
import panel as pn
import bokeh
from urllib.request import urlopen
from urllib.error import URLError
import socket
from IPython.utils.capture import capture_output
from panel.io.notebook import ipywidget

frame = gpd.GeoDataFrame({'value':[1]}, geometry=[Point(0,0)], crs=4326)
pn.config.comms = 'default'
with capture_output() as setup:
    import geosave.geodata.viz.plot
    viewer = frame.gs.explore(basemap='None')
with capture_output() as captured:
    bundle = viewer._repr_mimebundle_()
core = f'https://cdn.bokeh.org/bokeh/release/bokeh-{bokeh.__version__}.min.js'
assert any(core in output.data.get('application/vnd.holoviews_load.v0+json', '')
           for output in setup.outputs)
assert not any(core in output.data.get('application/vnd.holoviews_load.v0+json', '')
               for output in captured.outputs)
with capture_output() as second_setup:
    second = frame.gs.explore(basemap='None')
assert not any(core in output.data.get('application/vnd.holoviews_load.v0+json', '')
               for output in second_setup.outputs)
with capture_output() as repeated:
    second_bundle = second._repr_mimebundle_()
assert not any(core in output.data.get('application/vnd.holoviews_load.v0+json', '')
               for output in repeated.outputs)
if isinstance(bundle, tuple):
    bundle = bundle[0]
if isinstance(second_bundle, tuple):
    second_bundle = second_bundle[0]
assert 'application/vnd.jupyter.widget-view+json' in bundle
mime = 'application/vnd.jupyter.widget-view+json'
assert bundle[mime]['model_id'] != second_bundle[mime]['model_id']
widget = ipywidget(viewer)
assert widget.comm.target_name == 'jupyter.widget'
assert 'application/vnd.jupyter.widget-view+json' in widget._repr_mimebundle_()
assert 'panel.io.ipywidget' not in sys.modules
assert 'lonboard' not in sys.modules
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
server = viewer.serve(port=port, open=False)
try:
    assert server.is_alive()
    from time import monotonic, sleep
    deadline = monotonic() + 10
    while monotonic() < deadline:
        try:
            with urlopen(f'http://localhost:{port}/', timeout=15) as response:
                assert response.status == 200
            break
        except URLError:
            sleep(0.05)
    else:
        raise AssertionError('notebook server did not accept connections')
finally:
    server.stop()
    server.join(timeout=10)
assert not server.is_alive()
viewer.close()
second.close()
""")
        ]
    )
    if not static_first:
        notebook.cells[0].source = notebook.cells[0].source.replace(
            "    import geosave.geodata.viz.plot", "    pass"
        )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    nbclient.NotebookClient(notebook, km=manager, timeout=90).execute()
