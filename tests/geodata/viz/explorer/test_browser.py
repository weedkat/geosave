"""Actual served-browser regression; run with pytest -m slow."""

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


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


def test_native_catalog_rgb_pan_and_style(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    root = Path(__file__).resolve().parents[4]
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
                        !(model.properties.busy && model.busy)))""")
                page.screenshot(path=str(tmp_path / "narrow.png"))
                (tmp_path / "timings.json").write_text(json.dumps(timings, indent=2))
                print("TIMINGS", json.dumps(timings), flush=True)
                assert errors == []
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=15)
    assert "Traceback" not in (tmp_path / "server.log").read_text()
