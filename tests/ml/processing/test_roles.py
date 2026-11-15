"""Each processing role has its own registry, checked against the role's protocol."""

import pytest

from geosave.ml.factory import BuildSpec
from geosave.ml.processing.predictors import ChipPredictor
from geosave.ml.processing.roles import PREDICTORS, PREPROCESSORS


@pytest.fixture(autouse=True)
def restore_preprocessors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PREPROCESSORS, "_factories", dict(PREPROCESSORS))


def test_a_registered_preprocessor_rebuilds_from_its_spec():
    @PREPROCESSORS.register("scaler")
    class Scaler:
        def __init__(self, bands, scale=1.0):
            self.bands = bands
            self.scale = scale

        def preprocess(self, raster):
            return raster[self.bands]

        def normalize(self, image):
            return image * self.scale

    spec = PREPROCESSORS.get_spec(
        BuildSpec(name="Scaler", init_args={"bands": ["red"]})
    )
    scaler = PREPROCESSORS.build(BuildSpec.model_validate(spec.model_dump()))

    assert spec.init_args == {"bands": ["red"], "scale": 1.0}
    assert isinstance(scaler, Scaler)


def test_a_class_missing_a_role_method_is_refused():
    with pytest.raises(TypeError, match="Half is not a Preprocessor"):

        @PREPROCESSORS.register("half")
        class Half:
            def preprocess(self, raster):
                return raster


def test_the_library_predictor_is_found_without_importing_its_module():
    assert PREDICTORS.resolve(BuildSpec(name="chips")) is ChipPredictor
