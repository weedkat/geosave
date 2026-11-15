"""A processor is built from specs, predicts a raster by chips, and reloads from a release."""

import json

import numpy as np
import pytest
import torch
import xarray as xr

import geosave  # noqa: F401  registers the .gs accessors
from geosave.geodata.io import write_cog
from geosave.ml import build_model
from geosave.ml.processing.predictors import ChipPredictor
from geosave.ml.processing.roles import INGESTORS, POSTPROCESSORS, PREPROCESSORS
from geosave.ml.processing.processor import ProcessorSpec, build_processor
from geosave.ml.release import ChainModel, load_processor
from tests.geodata.sampling.conftest import build_scene

BANDS = ["red", "nir"]


class PixelModel(torch.nn.Module):
    """Scores each pixel alone, so merged chips must equal the whole raster."""

    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Conv2d(2, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.score(image)


@pytest.fixture(autouse=True)
def sub_processors(monkeypatch: pytest.MonkeyPatch) -> None:
    for registry in (INGESTORS, PREPROCESSORS, POSTPROCESSORS):
        monkeypatch.setattr(registry, "_factories", dict(registry))

    @INGESTORS.register("two_scenes")
    class TwoScenes:
        def ingest(self, region):
            return [build_scene(times=None), build_scene(times=None)]

    @PREPROCESSORS.register("halve")
    class Halve:
        def __init__(self, bands):
            self.bands = bands

        def preprocess(self, raster):
            return raster[self.bands].astype("float32")

        def normalize(self, image):
            return image / 2

    @POSTPROCESSORS.register("strongest")
    class Strongest:
        def postprocess(self, logits):
            return logits.argmax("band").astype("uint8").to_dataset(name="class_id")


@pytest.fixture
def spec() -> ProcessorSpec:
    return ProcessorSpec(
        ingestor={"name": "two_scenes"},
        preprocessor={"name": "halve", "init_args": {"bands": BANDS}},
        predictor={"name": "chips", "init_args": {"size": 4, "stride": 3}},
        postprocessor={"name": "strongest"},
    )


def test_a_processor_records_its_sub_processors_and_their_defaults(spec):
    processor = build_processor(spec, PixelModel())

    assert isinstance(processor.predictor, ChipPredictor)
    assert processor.spec is not None
    assert processor.spec.model_dump(exclude_none=True) == {
        "ingestor": {"name": "two_scenes", "init_args": {}},
        "preprocessor": {"name": "halve", "init_args": {"bands": BANDS}},
        "predictor": {
            "name": "chips",
            "init_args": {
                "size": 4,
                "stride": 3,
                "mode": "reflect",
                "window": "hann",
                "batch_size": 8,
            },
        },
        "postprocessor": {"name": "strongest", "init_args": {}},
    }
    json.dumps(processor.spec.model_dump())


def test_merged_chips_predict_what_the_whole_raster_would(spec):
    model = PixelModel().eval()
    processor = build_processor(spec, model)
    raster = build_scene(times=None)

    product = processor.predict_raster(raster)

    pixels = np.stack([raster["red"].values, raster["nir"].values])
    with torch.inference_mode():
        whole = model(image=torch.from_numpy(pixels)[None] / 2)[0]
    assert product["class_id"].values.tolist() == whole.argmax(0).tolist()
    assert product.gs.geobox == raster.gs.geobox


def test_a_product_keeps_its_rasters_acquisition_time(spec):
    processor = build_processor(spec, PixelModel().eval())
    raster = build_scene().isel(time=2)

    product = processor.predict_raster(raster)

    assert product["time"].values == raster["time"].values


def test_a_region_yields_one_product_per_ingested_raster(spec):
    processor = build_processor(spec, PixelModel().eval())

    products = processor.predict_region(region=None)

    assert len(products) == 2
    xr.testing.assert_equal(products[0], products[1])


def _segmentation_chain():
    return build_model(
        {
            "encoder": {
                "name": "small_cnn",
                "init_args": {"in_channels": 2, "input_size": 16},
            },
            "decoder": {"name": "unet"},
            "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
        }
    ).eval()


def test_a_released_processor_reloads_and_predicts_a_file(tmp_path):
    library_spec = ProcessorSpec(
        ingestor={"name": "sentinel2_l2a", "init_args": {"bands": ["B04", "B08"]}},
        preprocessor={"name": "sentinel2_reflectance", "init_args": {"bands": BANDS}},
        predictor={"name": "chips", "init_args": {"size": 16, "stride": 12}},
        postprocessor={
            "name": "class_map",
            "init_args": {
                "classes": ["a", "b", "c"],
                "colors": ["#000000", "#111111", "#222222"],
            },
        },
    )
    processor = build_processor(library_spec, _segmentation_chain())
    scene = build_scene(times=None)
    write_cog(scene, tmp_path / "scene.tif")

    ChainModel.from_processor(processor).save_pretrained(tmp_path / "release")
    reloaded = load_processor(str(tmp_path / "release"))

    assert reloaded.spec == processor.spec
    expected = processor.predict_raster(scene)
    product = reloaded.predict_path(tmp_path / "scene.tif")
    assert product["class_id"].values.tolist() == expected["class_id"].values.tolist()


def test_a_project_sub_processor_cannot_be_released(spec):
    processor = build_processor(spec, _segmentation_chain())

    with pytest.raises(ValueError, match="'two_scenes' is defined in tests"):
        ChainModel.from_processor(processor)


def test_a_release_without_a_processor_refuses_to_load_one(tmp_path):
    chain = build_model(
        {
            "head": {
                "name": "segmentation",
                "init_args": {"feature_channels": 2, "num_classes": 3},
            }
        }
    )
    ChainModel.from_chain(chain).save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="carries no processor"):
        load_processor(str(tmp_path))
