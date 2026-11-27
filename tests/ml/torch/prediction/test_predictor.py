import dask
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.geodata import GeoRegion, RasterChipper
from geosave.ml.pipeline import Ingestor, Pipeline, PreprocessOutput, Preprocessor
from geosave.ml.pipeline.postprocessing import SegmentationPostprocessor
from geosave.ml.torch import ModelChain
from geosave.ml.torch.prediction import TorchPredictor
from tests.geodata.conftest import _refuse

GRID = GeoBox.from_bbox(
    (500_000.0, 9_000_000.0, 500_480.0, 9_000_480.0), crs="EPSG:32749", shape=(48, 48)
)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """An untrained chain that reads a two-band `image`, saved and loaded back as a release."""
    chain = ModelChain.from_config(
        {
            "encoder": {
                "name": "small_cnn",
                "init_args": {"in_channels": 2, "input_size": 32},
            },
            "decoder": {"name": "unet"},
            "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
        }
    )
    folder = tmp_path_factory.mktemp("release")
    chain.save_pretrained(folder)
    return ModelChain.from_pretrained(folder)


REGION = GeoRegion.from_geometry(GRID.extent)


class Prepared(Ingestor):
    """A collection that is already the model input."""

    def __call__(self, region: GeoRegion) -> xr.DataTree:
        return prepared()


def prepared(name: str = "image") -> xr.DataTree:
    """One lazy model input: two float64 bands under `name` on a 48 x 48 grid."""
    rng = np.random.default_rng(0)
    coords = dict(xr_coords(GRID, always_yx=True))
    bands = {}
    for band in ("red", "nir"):
        bands[band] = (("y", "x"), rng.random(GRID.shape))
    return xr.DataTree.from_dict({name: xr.Dataset(bands, coords=coords).chunk()})


def test_each_chip_yields_its_location_and_output(model):
    predictor = TorchPredictor(model, chip_size=32, chip_overlap=0.5, batch_size=3)

    predicted = predictor(PreprocessOutput((prepared(), prepared())))
    results = list(predicted.outputs)

    chips = RasterChipper([prepared(), prepared()], size=32, overlap=0.5)
    assert predicted.chips.sources == chips.sources
    assert [location for location, _ in results] == [location for location, _ in chips]
    assert len(results) == 8
    # Each chip's source is its window.
    assert [location.source.index for location, _ in results] == [0] * 4 + [1] * 4
    for _, output in results:
        assert output.shape == (3, 32, 32)
        assert output.dtype == np.float32


def test_cutting_chips_reads_no_pixels_before_the_first_output(model):
    predictor = TorchPredictor(model, chip_size=32)

    with dask.config.set(scheduler=_refuse):
        predicted = predictor(PreprocessOutput((prepared(),)))

    next(predicted.outputs)


def test_a_pipeline_runs_a_released_model_into_a_class_map(model):
    pipeline = Pipeline(
        ingestors=[Prepared()],
        preprocessor=Preprocessor(steps=[]),
        predictor=TorchPredictor(model, chip_size=32, chip_overlap=0.5),
        postprocessor=SegmentationPostprocessor(window="hann"),
    )

    product = pipeline(REGION).product

    assert product.class_id.shape == (48, 48)
    assert set(np.unique(product.class_id.values)) <= {0, 1, 2}
    assert product.gs.geobox == GRID


def test_a_released_pipeline_predicts_what_the_original_predicts(model, tmp_path):
    pipeline = Pipeline(
        ingestors=[Prepared()],
        preprocessor=Preprocessor(steps=[]),
        predictor=TorchPredictor(model, chip_size=32, chip_overlap=0.5),
        postprocessor=SegmentationPostprocessor(window="hann"),
    )

    pipeline.save_pretrained(tmp_path)
    loaded = Pipeline.from_pretrained(tmp_path)

    assert isinstance(loaded.predictor.model, ModelChain)
    # The same repository still loads as the plain model.
    assert isinstance(ModelChain.from_pretrained(tmp_path), ModelChain)
    xr.testing.assert_identical(loaded(REGION).product, pipeline(REGION).product)
