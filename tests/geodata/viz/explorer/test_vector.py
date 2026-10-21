import pytest
from bokeh.models import HoverTool

from geosave.geodata.viz.explorer import Explorer


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
