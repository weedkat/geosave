from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import numpy as np
import pystac
import pytest
import rasterio
from prefect.settings import temporary_settings
from prefect.testing.utilities import prefect_test_harness
from pystac.extensions.projection import ProjectionExtension

from geosave.geodata.core.anchor import GeoAnchor


@pytest.fixture(scope="module")
def prefect_server():
    with (
        temporary_settings({"server.analytics_enabled": False}),
        prefect_test_harness(),
    ):
        yield


@dataclass
class LocalCatalog:
    """STAC documents served by the local endpoint fixture."""

    metadata: pystac.Collection
    items: list[pystac.Item]


@pytest.fixture
def local_stac(tmp_path):
    anchor = GeoAnchor.from_coordinates(
        45.0,
        12.0,
        shape=4,
        crs="EPSG:32633",
        resolution=10,
        timespan="2025-01",
    )
    box = anchor.geobox
    bounds = list(box.geographic_extent.boundingbox)
    timestamp = datetime(2025, 1, 15, 12, tzinfo=timezone.utc)
    collection = pystac.Collection(
        id="optical",
        description="Local raster smoke test",
        license="CC-BY-4.0",
        extent=pystac.Extent(
            pystac.SpatialExtent([bounds]),
            pystac.TemporalExtent([[timestamp, timestamp]]),
        ),
    )
    item = pystac.Item(
        id="optical-scene",
        geometry=box.geographic_extent.json,
        bbox=bounds,
        datetime=timestamp,
        properties={"platform": "local-sample"},
        collection="optical",
    )
    ProjectionExtension.ext(item, add_if_missing=True).apply(
        code="EPSG:32633",
        shape=list(box.shape),
        transform=list(box.transform),
    )
    for band, value in (("red", 2000), ("nir", 6000), ("unused", 1)):
        path = tmp_path / f"optical-{band}.tif"
        pixels = np.full(tuple(box.shape), value, dtype="uint16")
        pixels[0, 0] = 0
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=box.width,
            height=box.height,
            count=1,
            dtype="uint16",
            crs="EPSG:32633",
            transform=box.transform,
            nodata=0,
        ) as destination:
            destination.write(pixels, 1)
        item.add_asset(
            band,
            pystac.Asset(
                href=str(path),
                media_type=pystac.MediaType.GEOTIFF,
                roles=["data"],
                extra_fields={
                    "raster:bands": [
                        {
                            "data_type": "uint16",
                            "nodata": 0,
                            "scale": 0.0001,
                            "offset": 0,
                            "unit": "1",
                        }
                    ]
                },
            ),
        )
    return {"optical": LocalCatalog(collection, [item])}, anchor


@pytest.fixture
def stac_server(local_stac):
    """Serve fixture STAC documents while retaining real local raster assets."""
    catalogues, anchor = local_stac
    queries = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def reply(self, data):
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def search(self, params):
            requested = params.get("collections", list(catalogues))
            if isinstance(requested, str):
                requested = requested.split(",")
            queries.append(requested)
            self.reply(
                {
                    "type": "FeatureCollection",
                    "features": [
                        item.to_dict()
                        for name in requested
                        for item in catalogues[name].items
                    ],
                    "links": [],
                }
            )

        def do_POST(self):
            self.search(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/search":
                self.search(
                    {key: values[0] for key, values in parse_qs(parsed.query).items()}
                )
            elif parsed.path.startswith("/collections/"):
                self.reply(catalogues[parsed.path.rsplit("/", 1)[1]].metadata.to_dict())
            elif parsed.path == "/collections":
                self.reply(
                    {
                        "collections": [
                            entry.metadata.to_dict() for entry in catalogues.values()
                        ],
                        "links": [],
                    }
                )
            else:
                server = cast(ThreadingHTTPServer, self.server)
                root = f"http://127.0.0.1:{server.server_port}"
                self.reply(
                    {
                        "type": "Catalog",
                        "stac_version": "1.0.0",
                        "id": "local",
                        "description": "Local test catalog",
                        "conformsTo": [
                            f"https://api.stacspec.org/v1.0.0/{kind}"
                            for kind in ("core", "item-search", "collections")
                        ],
                        "links": [
                            {"rel": "self", "href": root},
                            {"rel": "root", "href": root},
                            {
                                "rel": "search",
                                "href": root + "/search",
                                "method": "POST",
                                "type": "application/geo+json",
                            },
                            {"rel": "data", "href": root + "/collections"},
                        ],
                    }
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", queries, anchor
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
