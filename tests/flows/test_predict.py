"""The prediction flow writes one store per acquisition and records its tasks in Prefect."""

import numpy as np
import pytest
import xarray as xr
from prefect import get_client
from prefect.testing.utilities import prefect_test_harness

import geosave  # noqa: F401  registers the .gs accessors
from geosave.geodata.io import open_raster
from geosave.ml.processing.raster.ingestion.base import RasterIngestor
from geosave.ml.release import ChainModel
from geosave_flows.predict import predict_region
from tests.geodata.sampling.conftest import build_scene
from tests.ml.processing.raster.test_pipeline import build_library_pipeline

DAYS = ["2024-07-01", "2024-07-06"]


class TwoDays(RasterIngestor):
    def __call__(self, region):
        acquired = []
        for day in DAYS:
            scene = build_scene(times=None).assign_coords(time=np.datetime64(day))
            acquired.append(xr.DataTree.from_dict({"image": scene}))
        return acquired


@pytest.fixture(scope="module", autouse=True)
def prefect_server():
    with prefect_test_harness():
        yield


def test_a_region_is_predicted_into_one_store_per_acquisition(tmp_path):
    pipeline = build_library_pipeline()
    pipeline.ingestor = TwoDays()
    ChainModel.from_pipeline(pipeline).save_pretrained(tmp_path / "release")
    region = {
        "kind": "coordinates",
        "latitude": -6.2,
        "longitude": 106.8,
        "shape": 8,
        "resolution": 10,
        "timespan": "2024-07",
    }

    stores = predict_region(
        repo=str(tmp_path / "release"), region=region, root=str(tmp_path / "out")
    )

    assert [store.name for store in stores] == [
        "20240701T000000.zarr",
        "20240706T000000.zarr",
    ]
    expected = pipeline.predict(pipeline.ingestor(None)[0])
    stored = open_raster(stores[0])
    assert stored["class_id"].values.tolist() == expected["class_id"].values.tolist()
    with get_client(sync_client=True) as client:
        tasks = sorted(run.name.rsplit("-", 1)[0] for run in client.read_task_runs())
    assert tasks == ["ingest", "predict_to_zarr", "predict_to_zarr"]
