import numpy as np
import pytest

from geosave.geodata.sampling import RasterChipper
from geosave.ml.pipeline import PredictOutput
from geosave.ml.pipeline.postprocessing import RegressionPostprocessor
from tests.ml.pipeline.postprocessing.test_segmentation import GRID, build_window


def chip_values(bands: int, size: int = 4, overlap: int = 2) -> PredictOutput:
    """Band `b` of every chip predicts `b + 1`."""
    chips = RasterChipper(build_window(), size=size, overlap=overlap)

    def predict():
        for location, _ in chips:
            values = np.ones((bands, size, size), dtype="float32")
            for band in range(bands):
                values[band] *= band + 1
            yield location, values

    return PredictOutput(chips, predict())


def test_each_output_band_becomes_a_named_variable():
    product = RegressionPostprocessor(["biomass", "height"])(chip_values(2)).product

    assert list(product.data_vars) == ["biomass", "height"]
    np.testing.assert_allclose(product.biomass.values, 1.0)
    np.testing.assert_allclose(product.height.values, 2.0)
    assert product.biomass.dtype == "float32"
    assert product.gs.geobox == GRID


def test_another_number_of_bands_than_variables_is_refused():
    with pytest.raises(ValueError, match="2 bands for 1 variables"):
        RegressionPostprocessor(["biomass"])(chip_values(2, overlap=0))
