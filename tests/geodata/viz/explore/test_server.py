"""Opt-in Explorer serving, headless browser, and notebook smoke tests."""

from __future__ import annotations

from geosave.geodata.viz.explore import create_layer

import os
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import panel as pn
import pytest

from geosave.geodata.viz.explore import Explorer
from geosave.geodata.viz.explore.server import servers

pytestmark = pytest.mark.slow


@pytest.fixture
def port():
    """Return a port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


def test_colab_serve_uses_the_proxy_origin_and_iframe(scene, monkeypatch, port):
    from IPython.core import getipython

    shell = type(
        "Shell", (), {"__module__": "google.colab._shell", "kernel": object()}
    )()
    monkeypatch.setattr(getipython, "get_ipython", lambda: shell)
    displayed = []
    output = SimpleNamespace(
        eval_js=lambda code: f"https://session-{port}.colab.googleusercontent.com/",
        serve_kernel_port_as_iframe=lambda port, **kwargs: displayed.append(port),
    )
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    settings = {}
    handle = SimpleNamespace(stop=lambda: None)

    def serve(panel, **kwargs):
        settings.update(kwargs)
        assert not displayed, "start the server before displaying its iframe"
        return handle

    monkeypatch.setattr(pn, "serve", serve)
    viewer = Explorer(layers=[create_layer(scene)], basemap="None")
    try:
        assert viewer.serve(port=port) is handle
        assert settings["threaded"] is True
        assert settings["show"] is False
        assert (
            f"session-{port}.colab.googleusercontent.com"
            in settings["websocket_origin"]
        )
        assert displayed == [port]
    finally:
        viewer.close()


def _wait_for(port, *, server=None, log_path=None):
    """Return once a server answers on `port`."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server is not None:
            assert server.poll() is None, log_path.read_text()
        with socket.socket() as sock:
            if sock.connect_ex(("localhost", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"nothing answers on port {port}")


def test_serving_again_on_a_port_replaces_the_earlier_explorer(scene, port):
    first = Explorer(layers=[create_layer(scene)], basemap="None")
    second = Explorer(layers=[create_layer(scene)], basemap="None")
    try:
        first.serve(port=port, threaded=True, open=False, verbose=False)
        _wait_for(port)

        second.serve(port=port, threaded=True, open=False, verbose=False)
        _wait_for(port)

        assert first.server is None
        assert servers[port] is second
    finally:
        first.close()
        second.close()


def test_stop_releases_the_port(scene, port):
    viewer = Explorer(layers=[create_layer(scene)], basemap="None")
    try:
        viewer.serve(port=port, threaded=True, open=False, verbose=False)
        _wait_for(port)

        viewer.stop()

        assert viewer.server is None
        assert port not in servers
        with socket.socket() as sock:
            assert sock.connect_ex(("localhost", port)) != 0
    finally:
        viewer.close()


def test_a_port_another_program_holds_is_refused_in_the_calling_thread(scene):
    viewer = Explorer(layers=[create_layer(scene)], basemap="None")
    with socket.socket() as blocker:
        blocker.bind(("localhost", 0))
        blocker.listen()
        port = blocker.getsockname()[1]
        try:
            with pytest.raises(OSError, match=f"port {port} is in use"):
                viewer.serve(port=port, threaded=True, open=False, verbose=False)
            assert viewer.server is None
        finally:
            viewer.close()


def test_script_serve_accepts_both_local_browser_origins(tmp_path, rgb_catalog):
    playwright = pytest.importorskip("playwright.sync_api")
    root = Path(__file__).resolve().parents[4]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = tmp_path / "launch.py"
    app.write_text(
        "import geosave as gs\n"
        f"viewer = gs.open_vector({str(rgb_catalog)!r}).gs.explore(basemap='None')\n"
        "viewer.table.raster_rows = [0]\n"
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


def test_native_catalog_selection_renders_and_removes_raster(
    tmp_path, rgb_catalog, record_property
):
    playwright = pytest.importorskip("playwright.sync_api")
    root = Path(__file__).resolve().parents[4]
    app = tmp_path / "catalog_only.py"
    app.write_text(
        "import geosave as gs\n"
        f"viewer = gs.open_vector({str(rgb_catalog)!r}).gs.explore(basemap='None')\n"
        "viewer.servable()\n"
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "panel",
                "serve",
                str(app),
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
            _wait_for(port, server=server, log_path=log_path)
            with playwright.sync_playwright() as runtime:
                browser = runtime.chromium.launch(
                    executable_path=os.environ.get("GEOSAVE_BROWSER"),
                    args=["--no-sandbox"],
                )
                page = browser.new_page()
                frame_sizes = []

                def record_frame(payload):
                    frame_sizes.append(
                        len(
                            payload.encode("utf-8")
                            if isinstance(payload, str)
                            else payload
                        )
                    )

                page.on(
                    "websocket", lambda socket: socket.on("framereceived", record_frame)
                )
                timings = []
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://localhost:{port}/catalog_only")
                # The first row's raster column, a tick toggled by a click
                check = page.locator(
                    '.tabulator-row .tabulator-cell[tabulator-field="raster"]'
                ).first
                check.wait_for(timeout=60000)
                for _ in range(2):
                    frame_sizes.clear()
                    started = time.perf_counter()
                    check.click()
                    page.wait_for_function("""() => Bokeh.documents.some(doc =>
                        [...doc._all_models.values()].some(model =>
                            model.type === 'GlyphRenderer' && model.visible
                            && model.glyph?.type === 'ImageRGBA'
                            && model.data_source.data.image?.[0]?.some(pixel => (pixel >>> 24) > 0)
                        ))""")
                    wait_for_raster(page)
                    timings.append(
                        {
                            "click_to_pixels_ms": (time.perf_counter() - started)
                            * 1000,
                            "websocket_payload_bytes": sum(frame_sizes),
                        }
                    )
                    # Footprints and raster loading are independent browser controls.
                    footprint = page.locator(
                        '.tabulator-row input[type="checkbox"]'
                    ).first
                    footprint.uncheck()
                    page.wait_for_function("""() => Bokeh.documents.every(doc =>
                        [...doc._all_models.values()].every(model =>
                            model.type !== 'GlyphRenderer' || !model.visible
                            || !['MultiPolygons', 'Patches'].includes(model.glyph?.type)
                            || !model.data_source.data.xs?.length
                        ))""")
                    wait_for_raster(page)
                    footprint.check()
                    check.click()
                    page.wait_for_function("""() => Bokeh.documents.every(doc =>
                        [...doc._all_models.values()].every(model =>
                            model.type !== 'GlyphRenderer' || !model.visible
                            || model.glyph?.type !== 'ImageRGBA'
                        ))""")
                record_property("catalog_browser_measurements", json.dumps(timings))
                assert errors == []
                assert server.poll() is None
                assert "Traceback" not in log_path.read_text()
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=15)


def test_native_notebook_comm():
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
second = frame.gs.explore(basemap='None')
second_bundle = second._repr_mimebundle_()
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
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    nbclient.NotebookClient(notebook, km=manager, timeout=90).execute()


def test_serving_the_same_explorer_again_keeps_its_layers(scene, port):
    viewer = Explorer(layers=[create_layer(scene)], basemap="None")
    try:
        viewer.serve(port=port, threaded=True, open=False, verbose=False)
        _wait_for(port)
        viewer.serve(port=port, threaded=True, open=False, verbose=False)
        _wait_for(port)

        assert viewer.layers
        assert servers[port] is viewer
    finally:
        viewer.close()


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
