from __future__ import annotations

import re

import pytest
import xarray as xr
from bokeh.models import HoverTool
from dask.callbacks import Callback

from geosave.geodata.viz.explore import Explorer, RasterLayer


@pytest.fixture
def checked_catalog(scene, fields):
    """A catalog with an already opened raster, independent of file I/O."""
    from geosave.geodata.viz.explore import VectorLayer

    frame = fields.copy()
    frame["id"] = ["first", "second", "third"]
    frame["links"] = [[], [], []]
    frame["assets"] = [{}, {}, {}]
    frame["stac_extensions"] = [[], [], []]
    layer = VectorLayer(frame)
    layer.rasters = {0: RasterLayer(scene)}
    layer.raster_rows = [0]
    return layer


def test_copied_catalog_ticks_open_rasters_independently_of_footprints(checked_catalog):
    viewer = Explorer(checked_catalog, basemap="None")
    copied = viewer.copy()
    try:
        assert copied.table.value["raster"].tolist() == [True, False, False]
        copied.table.selection = []
        assert copied.layers[0].footprint_rows == []
        assert copied.layers[0].raster_rows == [0]
        assert len(copied.layers) == 2
    finally:
        copied.close()
        viewer.close()


def test_catalog_raster_control_preserves_a_source_column_named_raster(checked_catalog):
    from panel.models.tabulator import CellClickEvent
    from geosave.geodata.viz.explore import VectorLayer

    layer = VectorLayer(checked_catalog.data.assign(raster="source value"))
    layer.rasters = checked_catalog.rasters
    layer.raster_rows = [0]
    viewer = Explorer(layer, basemap="None")
    try:
        table = viewer.table
        assert table.value["raster"].tolist() == ["source value"] * 3
        column = next(iter(table.formatters))
        table._process_event(CellClickEvent(model=None, column=column, row=0))
        assert layer.raster_rows == []
        assert layer.footprint_rows == [0, 1, 2]
    finally:
        viewer.close()


@pytest.mark.parametrize("action", ["remove", "close"])
def test_detached_table_no_longer_changes_its_layer(fields, action):
    viewer = Explorer(fields, fields.copy(), basemap="None")
    layer = viewer.layers[0]
    table = viewer.table
    if action == "remove":
        viewer.remove(layer)
    else:
        viewer.close()
    table.selection = []
    assert layer.footprint_rows == [0, 1, 2]
    layer.footprint_rows = [1]
    assert table.selection == []
    viewer.close()
    assert viewer.table is None


def test_identify_uses_comparison_visibility_and_unique_labels(scene):
    first = RasterLayer(scene, name="scene")
    second = RasterLayer(scene * 2, name="scene")
    viewer = Explorer(first, second, basemap="None")
    x = (first.bbox.left + first.bbox.right) / 2
    y = (first.bbox.bottom + first.bbox.top) / 2
    try:
        viewer.map._identify(x=x, y=y)
        assert set(viewer.map.clicked_values) == {"scene", "scene (2)"}
        viewer.comparison.enabled = True
        viewer.map._identify(x=x, y=y)
        assert list(viewer.map.clicked_values) == ["scene"]
    finally:
        viewer.close()


@pytest.mark.parametrize("action", ["remove", "close"])
def test_detached_color_control_no_longer_changes_its_layer(fields, action):
    import panel as pn

    viewer = Explorer(fields, fields.copy(), basemap="None")
    layer = viewer.layers[0]
    page = viewer.__panel__()
    controls = page.select(pn.ui.Select)
    color = next(widget for widget in controls if widget.label == "Color by")
    if action == "remove":
        viewer.remove(layer)
    else:
        viewer.close()
    color.value = "area"
    assert layer.column is None
    viewer.close()


def test_identify_does_not_read_rasters_outside_the_date_period(monthly):
    import pandas as pd
    from dask.callbacks import Callback

    viewer = Explorer(
        monthly.isel(time=[0]).chunk(), monthly.isel(time=[1]), basemap="None"
    )
    first = viewer.layers[0]
    viewer.period = pd.Period("2024-04", freq="M")
    tasks = []
    try:
        with Callback(pretask=lambda key, *_: tasks.append(key)):
            viewer.map._identify(
                x=(first.bbox.left + first.bbox.right) / 2,
                y=(first.bbox.bottom + first.bbox.top) / 2,
            )
        assert tasks == []
        assert len(viewer.map.clicked_values) == 1
    finally:
        viewer.close()


@pytest.mark.parametrize("action", ["supersede", "remove", "close"])
@pytest.mark.parametrize("fails", [True, False])
def test_obsolete_catalog_result_does_not_change_selection(
    checked_catalog, monkeypatch, action, fails
):
    import asyncio
    import panel as pn

    viewer = Explorer(checked_catalog, basemap="None")
    layer = viewer.layers[0]
    queued = []
    started = asyncio.Event()
    finish = asyncio.Event()

    async def fail(*_):
        started.set()
        await finish.wait()
        if fails:
            raise OSError("obsolete failure")
        return {1: RasterLayer(checked_catalog.rasters[0].data)}

    monkeypatch.setattr(asyncio, "to_thread", fail)
    monkeypatch.setattr(pn.state, "execute", queued.append)

    async def exercise():
        layer.raster_rows = [0, 1]
        task = asyncio.create_task(queued.pop()())
        await asyncio.wait_for(started.wait(), 5)
        updates = []
        viewer.table.param.watch(lambda event: updates.append(event), "value")
        if action == "supersede":
            layer.raster_rows = [0, 2]
        elif action == "remove":
            viewer.remove(layer)
        else:
            viewer.close()
            assert updates == []
        expected = list(layer.raster_rows)
        finish.set()
        await task
        assert layer.raster_rows == expected
        assert layer.error == ""
        assert list(layer.rasters) == [0]
        if action == "remove":
            viewer.add(layer)
            assert layer.raster_rows == [0]

    try:
        asyncio.run(exercise())
    finally:
        finish.set()
        viewer.close()


def test_catalog_filter_sort_pagination_keep_row_identity(catalog):
    viewer = Explorer(catalog, basemap="None")
    table = viewer.table
    table.page_size = 1
    table.sorters = [{"field": "id", "dir": "desc"}]
    table.page = 2
    table.filters = [{"field": "id", "type": "=", "value": catalog.iloc[2]["id"]}]
    viewer.layers[0].raster_rows = [2]
    expected = catalog.iloc[2].gs.to_datatree().gs.to_dataset()
    xr.testing.assert_equal(viewer.layers[-1].data.B04, expected.B04)
    table.filters = []
    assert table.value["raster"].tolist() == [False, False, True]
    viewer.layers[0].raster_rows = []
    assert len(viewer.layers) == 1
    viewer.close()


def test_opacity_and_vector_style_do_not_compute_other_rasters(scene, fields):
    viewer = Explorer(scene.chunk({"y": 16, "x": 16}) * 2, fields, basemap="None")
    viewer.map.pane.get_root()
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        viewer.layers[1].column = "area"
        viewer.layers[0].visible = False
        viewer.layers[0].alpha = 0.2
    assert tasks == []
    viewer.close()


@pytest.mark.parametrize("action", ["opacity", "pan", "uncached pan"])
def test_rgb_resends_pixels_only_for_uncached_views(scene, action):
    from bokeh.document import Document
    from bokeh.models import GlyphRenderer, ImageRGBA

    viewer = Explorer(scene, basemap="None")
    document = Document()
    root = viewer.map.pane.get_root(document)
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
        left, right = viewer.map.ranges.x_range
        bottom, top = viewer.map.ranges.y_range
        # The cache spans half a view past each edge
        distance = 0.6 if action == "uncached pan" else 0.01
        dx, dy = (right - left) * distance, (top - bottom) * distance
        viewer.map.ranges.event(
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
        viewer.map.pane.get_root()
        figure = scene.gs.plot("B04")
        assert hv.render(figure, backend="matplotlib") is not None
        viewer.layers[0].band = "B04"
        viewer.map.pane.get_root()
        viewer.close()


def test_composed_catalog_retains_ownership_and_independent_settings(scene, catalog):
    viewer = Explorer(catalog, basemap="None")
    viewer.layers[0].raster_rows = [0]
    opened = viewer.layers[-1]
    opened.alpha = 0.4
    composed = viewer * scene
    copied = composed.layers[1]
    assert copied is not opened and copied.alpha == 0.4
    composed.layers[0].raster_rows = []
    assert copied not in composed.layers
    assert opened in viewer.layers
    assert viewer.layers[0].raster_rows == [0]
    composed.close()
    viewer.close()


def test_removed_raster_does_not_execute_on_later_controls_or_view(scene, fields):
    layer = RasterLayer(scene.chunk({"y": 16, "x": 16}) * 2)
    viewer = Explorer(layer, fields, basemap="None")
    viewer.map.pane.get_root()
    viewer.remove(layer)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        layer.band = "B03"
        viewer.map.ranges.event(x_range=(0, 1000), y_range=(0, 1000))
    assert tasks == []
    viewer.close()


def test_catalog_open_failure_preserves_selection_and_existing_layers(
    catalog, fields, monkeypatch
):
    viewer = Explorer(catalog, fields, basemap="None")

    def fail(*args, **kwargs):
        raise OSError("unavailable file")

    # The native row reader owns the error; the explorer keeps its last successful selection.
    monkeypatch.setattr(type(catalog.iloc[0].gs), "to_datatree", fail)
    viewer.layers[0].raster_rows = [0]
    assert len(viewer.layers) == 2
    assert viewer.layers[0].raster_rows == []
    assert "unavailable file" in viewer.status.object
    viewer.close()


def test_class_legend_without_colors_is_visible(scene):
    from geosave.geodata.attrs import Legend, stamp_attrs

    data = scene[["cover"]].copy()
    data["cover"] = stamp_attrs(data.cover, Legend(classes={0: "water", 1: "trees"}))
    layer = RasterLayer(data)
    assert set(layer.swatches) == {"water", "trees"}
    assert data.cover.gs.legend.color_map is None
    plot = layer.draw(None).values()[0]
    assert plot.dimension_values(plot.vdims[-1]).max() == 1


def test_date_filters_keep_checked_row_identity_and_raster_time(monthly, catalog):
    from datetime import date

    viewer = Explorer(monthly, catalog, basemap="None")
    raster, vector = viewer.layers
    vector.raster_rows = [1]
    opened = viewer.layers[-1]
    vector.period = (date(2024, 4, 1), date(2024, 4, 30))
    assert vector.shown_rows == [1]
    assert vector.raster_rows == [1]
    raster.time = monthly.time.values[-1]
    assert raster.select_shown_bands().B04.equals(monthly.B04.isel(time=-1))
    vector.period = (date(2024, 5, 1), date(2024, 5, 5))
    assert vector.shown_rows == [2]
    assert opened in viewer.layers
    vector.raster_rows = []
    assert opened not in viewer.layers
    viewer.close()


def test_map_identification_reads_values_and_selects_vector(scene, fields):
    viewer = Explorer(scene, fields, basemap="None")
    vector = viewer.layers[-1]
    point = vector.frame.geometry.iloc[0].centroid
    viewer.map._identify(x=point.x, y=point.y)
    assert vector.picked_rows == [0]
    assert "rice" in str(viewer.map.clicked_values)
    assert "B04" in str(viewer.map.clicked_values)
    viewer.close()


def test_new_renderers_never_inherit_hover_fields_they_do_not_have(scene, catalog):
    viewer = Explorer(scene, catalog, basemap="None")
    root = viewer.map.pane.get_root()
    for checked in ([0], [], [0]):
        viewer.layers[1].raster_rows = checked
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
        viewer.map.pane.get_root()
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
    viewer.active_layer = viewer.layers[1]
    assert tables <= root.references()
    viewer.active_layer = viewer.layers[0]
    assert tables <= root.references()
    viewer.close()


def test_composition_retains_explicit_rgb_without_declared_rgb(scene):
    data = scene[["B04", "cover"]]
    assert data.gs.rgb is None
    viewer = Explorer(RasterLayer(data, rgb=("B04", "B04", "cover")), basemap="None")
    composed = viewer * scene
    assert composed.layers[0].is_rgb
    assert composed.layers[0].rgb == ("B04", "B04", "cover")
    assert (
        composed.layers[0].select_shown_bands().data_vars.keys()
        == data.data_vars.keys()
    )
    composed.close()
    viewer.close()


def test_repeated_explorer_keeps_each_catalogs_opened_rasters(catalog):
    viewer = catalog.gs.explore(basemap="None")
    viewer.layers[0].raster_rows = [0]
    composed = viewer * viewer
    assert len(composed.layers) == 4
    composed.layers[0].raster_rows = []
    assert len(composed.layers) == 3
    composed.layers[1].raster_rows = []
    assert len(composed.layers) == 2
    assert len(viewer.layers) == 2
    composed.close()
    viewer.close()


def test_zoom_updates_the_rendered_map_ranges(scene, fields):
    from bokeh.models import Plot

    from geosave.geodata.viz.explore.mercator import frame_bbox

    viewer = Explorer(scene, fields, basemap="None")
    root = viewer.map.pane.get_root()
    figure = root.select_one({"type": Plot})
    extent = frame_bbox(viewer.layers[1].bbox)
    viewer.zoom(viewer.layers[1].bbox)
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
    layer.band = display
    root = viewer.map.pane.get_root()
    renderer = next(
        value
        for value in root.select({"type": GlyphRenderer})
        if "image" in value.data_source.data
    )
    pixels = renderer.data_source.data["image"][0].copy()
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        layer.stretch_percentiles = (10, 90)
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
    root = viewer.map.pane.get_root()
    figure = root.select_one({"type": Plot})
    viewer.map.size.event(width=1000, height=500)
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

    from geosave.geodata.viz.explore.mercator import frame_bbox

    viewer = Explorer(scene, fields, basemap="None")
    document = Document()
    root = viewer.map.pane.get_root(document)
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


def test_region_joins_drawn_boxes_and_polygons_in_wgs84(scene):
    from odc.geo.geom import box

    viewer = Explorer(scene, basemap="None")
    session = viewer._create_session()
    assert viewer.region is None

    session.map.pane.get_root()
    boxes, polygons = session.map.region.sources_per_figure[0]
    area = box(500_000, 9_000_000, 500_640, 9_000_640, "EPSG:32749")
    left, bottom, right, top = area.to_crs("EPSG:3857").boundingbox
    boxes.data = {"left": [left], "bottom": [bottom], "right": [right], "top": [top]}
    polygons.data = {"xs": [[left, right, right]], "ys": [[bottom, bottom, top]]}

    region = viewer.region
    assert region is not None
    assert region.crs == "EPSG:4326"
    assert region.to_crs("EPSG:32749").area == pytest.approx(area.area, rel=1e-3)
    assert scene.gs.crop(region, mask=False).sizes == scene.sizes
    viewer.close()


def test_comparison_shows_one_chosen_raster(scene, monthly):
    viewer = Explorer(scene, monthly, basemap="None")
    first, second = viewer.layers
    comparison = viewer.comparison

    assert list(comparison.rasters.values()) == [first, second]
    assert (comparison.left, comparison.right) == (first, second)

    comparison.enabled = True
    comparison.side = "right"
    assert comparison.shown_raster is second

    viewer.remove(second)
    assert comparison.enabled is False
    assert comparison.left is first
    viewer.close()


def date_scene(scene, *dates):
    """The scene on each date, as one layer."""
    import numpy as np

    times = np.array(dates, dtype="datetime64[ns]")
    return RasterLayer(scene.expand_dims(time=times), "B04")


def test_periods_are_the_months_of_switched_on_layers(scene):
    import pandas as pd

    viewer = Explorer(
        date_scene(scene, "2024-03-05"),
        date_scene(scene, "2024-04-05"),
        date_scene(scene, "2024-05-05"),
        basemap="None",
    )

    assert viewer.param.period.objects == [
        pd.Period("2024-03", "M"),
        pd.Period("2024-04", "M"),
        pd.Period("2024-05", "M"),
    ]
    viewer.close()


def test_day_step_separates_dates_one_day_apart(scene):
    viewer = Explorer(
        date_scene(scene, "2024-01-05"), date_scene(scene, "2024-01-06"), basemap="None"
    )
    assert len(viewer.param.period.objects) == 1

    viewer.freq = "D"

    assert len(viewer.param.period.objects) == 2
    viewer.close()


def test_each_period_draws_only_layers_with_a_date_in_it(scene):
    import pandas as pd

    march = date_scene(scene, "2024-03-05")
    april = date_scene(scene, "2024-04-05")
    timeless = RasterLayer(scene, "B04")
    viewer = Explorer(march, april, timeless, basemap="None")

    viewer.period = pd.Period("2024-04", "M")

    assert viewer.select_drawn_rasters() == [april, timeless]
    viewer.close()


def test_hiding_a_layer_drops_its_periods(scene):
    march = date_scene(scene, "2024-03-05")
    viewer = Explorer(march, date_scene(scene, "2024-04-05"), basemap="None")

    march.visible = False

    assert len(viewer.param.period.objects) == 1
    viewer.close()


def test_select_time_takes_the_last_date_in_the_period(scene):
    import numpy as np
    import pandas as pd

    layer = date_scene(scene, "2024-01-05", "2024-01-20", "2024-02-03")

    assert layer.select_time(pd.Period("2024-01", "M")) == np.datetime64("2024-01-20")
    assert layer.select_time(pd.Period("2024-03", "M")) is None


def test_date_slider_survives_its_last_dated_raster_leaving(scene):
    import panel as pn

    march = date_scene(scene, "2024-03-05")
    viewer = Explorer(march, RasterLayer(scene, "B04"), basemap="None")
    slider = pn.ui.DiscreteSlider.from_param(viewer.param.period)
    slider.get_root()

    march.visible = False

    assert viewer.period_count == 0
    assert viewer.select_drawn_rasters() == [viewer.layers[1]]
    viewer.close()


def test_details_reuses_controls_and_detaches_selection(fields):
    viewer = Explorer(fields, fields.copy(), basemap="None")
    details = viewer.details
    first, second = viewer.layers
    try:
        page = viewer.__panel__()
        initial = details.view_style()
        viewer.active_layer = second
        assert details.layer is second
        viewer.active_layer = first
        assert details.view_style() is initial
        assert viewer.__panel__() is page
        details.release()
        viewer.active_layer = second
        assert details.layer is None
    finally:
        viewer.close()


def test_initial_period_synchronizes_a_preconfigured_raster(monthly):
    layer = RasterLayer(monthly)
    layer.time = layer.times[-1]
    viewer = Explorer(layer, basemap="None")
    try:
        assert layer.time == layer.select_time(viewer.period)
    finally:
        viewer.close()
