import pytest
from bokeh.models import HoverTool

from geosave.geodata.viz.explore import Explorer, VectorLayer


def test_hover_preserves_attribute_names_and_datetime_offsets(fields):
    import pandas as pd

    frame = fields.rename(columns={"crop": "land cover", "area": "mean-value"})
    frame["A.B"] = "text"
    frame["observed"] = pd.Timestamp("2024-01-01 12:00:00+07:00")
    viewer = Explorer(frame, basemap="None")
    root = viewer.map.pane.get_root()
    for tool in root.select({"type": HoverTool}):
        assert tool.renderers
        for renderer in tool.renderers:
            assert (
                renderer.data_source.data["observed"][0] == "2024-01-01 12:00:00 +0700"
            )
    assert frame.observed.iloc[0].utcoffset().total_seconds() == 7 * 3600
    viewer.close()


@pytest.mark.parametrize("kind,count", [("point", 50_001), ("polygon", 5_001)])
@pytest.mark.parametrize("column", ["value", "class"])
def test_dense_vectors_use_bounded_datashader_images(kind, count, column):
    import geopandas as gpd
    import geoviews as gv
    import numpy as np
    from shapely import box

    from geosave.geodata.viz.explore import VectorLayer

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
    image = layer.render().values()[0]
    assert isinstance(image, gv.Image if column == "value" else gv.RGB)
    assert image.dimension_values(image.kdims[0], expanded=False).size <= 900
    assert image.dimension_values(image.kdims[1], expanded=False).size <= 600


def test_glyph_layers_do_not_redraw_on_pan(fields, monkeypatch):
    from geosave.geodata.viz.explore import VectorLayer

    renders = []
    render = VectorLayer.render

    def count(self, *args, **kwargs):
        renders.append(1)
        return render(self, *args, **kwargs)

    monkeypatch.setattr(VectorLayer, "render", count)
    viewer = Explorer(fields, basemap="None")
    viewer.map.pane.object[()]
    drawn = len(renders)
    viewer.map.ranges.event(x_range=(0.0, 1000.0), y_range=(0.0, 1000.0))
    viewer.map.pane.object[()]

    assert len(renders) == drawn
    viewer.close()


def test_numeric_color_limits_skip_missing_and_infinite_values(fields):
    import numpy as np

    from geosave.geodata.viz.explore import VectorLayer

    frame = fields.assign(area=[1.0, np.inf, np.nan], empty=np.nan)
    layer = VectorLayer(frame)

    assert layer.get_limits("area") == (1.0, 1.0)
    assert layer.get_limits("empty") is None
    layer.column = "empty"
    assert layer.render()


def test_an_empty_pyarrow_date_column_does_not_stop_exploring():
    import geopandas as gpd
    import pandas as pd
    import pyarrow as pa
    from shapely.geometry import box

    # Item tables read from GeoParquet hold pyarrow dates, empty where no Item has one.
    dates = pd.array([None, None], dtype=pd.ArrowDtype(pa.timestamp("us", tz="UTC")))
    fields = gpd.GeoDataFrame(
        {"end_datetime": dates}, geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)], crs=4326
    )

    viewer = fields.gs.explore(basemap="None")
    try:
        viewer.map.pane.get_root()
    finally:
        viewer.close()


def test_categories_take_sorted_palette_colours_shared_by_map_and_legend(fields):
    from geosave.geodata.viz.explore import VectorLayer

    frame = fields.assign(
        crop=["rice", "palm"] * (len(fields) // 2) + ["rice"] * (len(fields) % 2)
    )
    layer = VectorLayer(frame, column="crop")

    assert list(layer.category_colors) == ["palm", "rice"]
    assert layer.swatches == layer.category_colors
    layer.column = "area"
    assert layer.category_colors is None


def count_drawn_features(layer) -> int:
    drawn = 0
    for element in layer.render().values():
        drawn += len(element.data)
    return drawn


def test_unchecked_footprint_is_not_drawn(fields):
    layer = VectorLayer(fields[fields.geom_type == "Polygon"])
    assert layer.footprint_rows == [0, 1]

    layer.footprint_rows = [0]

    assert count_drawn_features(layer) == 1


def test_frame_without_assets_offers_no_rasters(fields):
    layer = VectorLayer(fields)

    assert layer.has_rasters is False


def test_checking_a_raster_row_adds_its_layer(catalog):
    import time

    from geosave.geodata.viz.explore import RasterLayer

    layer = VectorLayer(catalog)
    assert layer.has_rasters is True

    layer.raster_rows = [0]
    # Rows open off the page's thread when a page runs; here they open in place.
    deadline = time.monotonic() + 10
    while not layer.rasters and time.monotonic() < deadline:
        time.sleep(0.05)

    assert list(layer.rasters) == [0]
    raster = layer.rasters[0]
    assert isinstance(raster, RasterLayer)
    assert raster.row.equals(catalog.iloc[0])


def test_map_click_picks_without_opening_rasters(catalog):
    layer = VectorLayer(catalog)
    layer.shown_rows = list(range(len(catalog)))
    center = layer.frame.geometry.iloc[0].centroid

    layer.identify(center.x, center.y, 1.0)

    assert 0 in layer.picked_rows
    assert layer.raster_rows == []


def test_catalog_status_does_not_replace_drawn_geometry(fields):
    from geosave.geodata.viz.explore import VectorLayer

    layer = VectorLayer(fields)
    original = layer.draw(None)
    layer.error = "Unable to open raster"
    assert layer.draw(None) is original
    layer.color = "#ff0000"
    assert layer.draw(None) is not original
