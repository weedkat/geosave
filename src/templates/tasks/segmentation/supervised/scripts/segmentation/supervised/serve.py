"""Serve a release with LitServe: a raster store in, a product store out.

    python scripts/segmentation/supervised/serve.py release/model --port 8000

    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \\
        -d '{"rasters": "s3://bucket/scene.zarr", "destination": "s3://bucket/product.zarr"}'

The server runs the release's whole pipeline, so a request names stores
instead of carrying pixels:

    rasters store ──open──▶ pipeline.predict_to_zarr ──▶ destination store

`rasters` is a store as `pipeline.ingestor.ingest_to_zarr` writes it: one
group per raw raster. The release must carry a pipeline
(`ChainModel.from_pipeline`). Its sub-processors selected by class path are
this workspace's own, so build the server image from the workspace.

Requires `pip install litserve`.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import litserve as ls

from geosave.geodata.io import open_raster_tree
from geosave.ml.release import load_pipeline

# Sub-processors selected by class path live in this workspace's `modules`.
# Set at import, so LitServe's worker processes, which import this file, find them too.
WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))


class RasterAPI(ls.LitAPI):
    """Predict the raster store a request names into the store it asks for.

    Args:
        release: Hub repository id or local folder of a release with a pipeline.
    """

    def __init__(self, release: str) -> None:
        super().__init__()
        self.release = release

    def setup(self, device: str) -> None:
        """Load the release's pipeline and move its model to the device."""
        self.pipeline = load_pipeline(self.release)
        self.pipeline.model.to(device)

    def decode_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Take the request as sent: `rasters`, `destination` and optional `chunks`."""
        return request

    def predict(self, request: dict[str, Any]) -> str:
        """Write the product of the named rasters and return its store's path."""
        rasters = open_raster_tree(request["rasters"])
        store = self.pipeline.predict_to_zarr(
            rasters, request["destination"], chunks=request.get("chunks", 2048)
        )
        return str(store)

    def encode_response(self, output: str) -> dict[str, str]:
        """Name the written store."""
        return {"product": output}


def main() -> None:
    """Start the server for one release."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("release", help="Hub repository id or local release folder.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args()

    server = ls.LitServer(RasterAPI(args.release), accelerator=args.accelerator)
    server.run(port=args.port)


if __name__ == "__main__":
    main()
