"""A monitored processor returns the same products and records its steps in Prefect."""

import pytest
import torch
import xarray as xr
from prefect import get_client
from prefect.testing.utilities import prefect_test_harness

import geosave  # noqa: F401  registers the .gs accessors
from geosave.geodata import GeoRegion
from geosave.geodata.io import write_cog
from geosave.ml.processing.predictors import ChipPredictor
from geosave.ml.processing.processor import Processor
from geosave_flows.monitor import monitor
from tests.geodata.sampling.conftest import build_scene


class PixelModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Conv2d(2, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.score(image)


class TwoScenes:
    def ingest(self, region):
        return [build_scene(times=None), build_scene(times=None)]


class Bands:
    def preprocess(self, raster):
        return raster[["red", "nir"]].astype("float32")

    def normalize(self, image):
        return image


class Strongest:
    def postprocess(self, logits):
        return logits.argmax("band").astype("uint8").to_dataset(name="class_id")


@pytest.fixture(scope="module", autouse=True)
def prefect_server():
    with prefect_test_harness():
        yield


def build_processor(model: PixelModel) -> Processor:
    return Processor(
        model, TwoScenes(), Bands(), ChipPredictor(size=4, stride=3), Strongest()
    )


def test_a_monitored_region_returns_the_same_products_and_records_each_step():
    model = PixelModel().eval()
    region = GeoRegion.from_bbox((106.8, -6.25, 106.85, -6.2), timespan="2024-07-01")
    expected = build_processor(model).predict_region(region)

    products = monitor(build_processor(model)).predict_region(region)

    assert len(products) == 2
    xr.testing.assert_equal(products[0], expected[0])
    with get_client(sync_client=True) as client:
        flows = sorted(run.name for run in client.read_flows())
        tasks = sorted(run.name.rsplit("-", 1)[0] for run in client.read_task_runs())
    assert flows == ["predict_raster", "predict_region"]
    # Prefect also records each raster's subflow as a task run of its parent.
    assert tasks == [
        "ingest",
        "postprocess",
        "postprocess",
        "predict",
        "predict",
        "predict_raster",
        "predict_raster",
    ]


def test_a_monitored_processor_predicts_a_file(tmp_path):
    scene = build_scene(times=None)
    write_cog(scene, tmp_path / "scene.tif")
    model = PixelModel().eval()
    expected = build_processor(model).predict_raster(scene)

    product = monitor(build_processor(model)).predict_path(tmp_path / "scene.tif")

    assert product["class_id"].values.tolist() == expected["class_id"].values.tolist()
