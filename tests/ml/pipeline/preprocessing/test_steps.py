import xarray as xr

from geosave.ml.pipeline.preprocessing import Composite, MaskAndScale
from tests.ml.pipeline.preprocessing.test_preprocessor import sources


def test_mask_and_scale_runs_the_tree_method():
    xr.testing.assert_identical(
        MaskAndScale()(sources()), sources().gs.mask_and_scale()
    )


def test_composite_passes_every_argument_to_the_tree_method():
    decoded = sources().gs.mask_and_scale()
    step = Composite("MS", "mean", closed="right", label="right", offset="1D")

    expected = decoded.gs.composite(
        "MS", "mean", closed="right", label="right", offset="1D"
    )
    xr.testing.assert_identical(step(decoded), expected)
