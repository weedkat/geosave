from __future__ import annotations

from geosave.geodata.viz.explore import Explorer, RasterLayer


def test_header_export_is_generated_on_click_with_current_legends(scene):
    import base64
    from html import unescape

    import panel as pn
    from dask.callbacks import Callback

    layer = RasterLayer(scene.chunk(), band="cover", name="Land <cover>")
    hidden = RasterLayer(scene, band="B04", name="Hidden export layer")
    hidden.visible = False
    viewer = Explorer(layers=[layer, hidden], basemap="None")
    tasks = []
    try:
        with Callback(posttask=lambda key, *args: tasks.append(key)):
            page = viewer.__panel__()
            download = next(
                widget
                for widget in page.header
                if isinstance(widget, pn.ui.FileDownload)
            )
        assert not tasks
        assert not download.data
        download.transfer()
        content = unescape(base64.b64decode(download.data.split(",", 1)[1]).decode())
        assert "Land &lt;cover&gt;" in content
        assert "water" in content and "trees" in content
        assert "Hidden export layer" not in content
        assert "do not load new detail" in content

        layer.band = "B04"
        download.transfer()
        content = unescape(base64.b64decode(download.data.split(",", 1)[1]).decode())
        assert "Land &lt;cover&gt;" in content
        assert "<b>B04</b>" in content
        assert "<span>water</span>" not in content
    finally:
        viewer.close()
