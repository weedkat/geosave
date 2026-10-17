"""Product regressions for the Material explorer sidebar and map tools."""

import re

import panel_material_ui as mui
import pytest
from bokeh.models import HoverTool
from dask.callbacks import Callback

from geosave.geodata.viz.explorer import Explorer, RasterLayer


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


def test_hover_preserves_attribute_names_and_datetime_offsets(fields):
    import pandas as pd

    frame = fields.rename(columns={"crop": "land cover", "area": "mean-value"})
    frame["A.B"] = "text"
    frame["observed"] = pd.Timestamp("2024-01-01 12:00:00+07:00")
    viewer = Explorer(frame, basemap="None")
    root = viewer.map.get_root()
    for tool in root.select({"type": HoverTool}):
        assert tool.renderers
        for renderer in tool.renderers:
            assert (
                renderer.data_source.data["observed"][0] == "2024-01-01 12:00:00 +0700"
            )
    assert frame.observed.iloc[0].utcoffset().total_seconds() == 7 * 3600
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


@pytest.mark.parametrize("kind,count", [("point", 50_001), ("polygon", 5_001)])
@pytest.mark.parametrize("column", ["value", "class"])
def test_dense_vectors_use_bounded_datashader_images(kind, count, column):
    import geopandas as gpd
    import geoviews as gv
    import numpy as np
    from shapely import box
    from geosave.geodata.viz.explorer import VectorLayer

    x = np.arange(count) % 100
    y = np.arange(count) // 100
    geometry = (
        gpd.points_from_xy(x, y) if kind == "point" else box(x, y, x + 0.8, y + 0.8)
    )
    frame = gpd.GeoDataFrame(
        {"value": np.full(count, 3.0), "class": np.where(x % 2, "rice", "palm")},
        geometry=geometry,
        crs=3857,
    )
    layer = VectorLayer(frame, column=column)
    image = layer.render(
        column, layer.color, layer.alpha, True, layer.rows, []
    ).values()[0]
    assert isinstance(image, gv.Image if column == "value" else gv.RGB)
    assert image.dimension_values(image.kdims[0], expanded=False).size <= 900
    assert image.dimension_values(image.kdims[1], expanded=False).size <= 600


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
    from bokeh.models import GlyphRenderer
    import numpy as np

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
