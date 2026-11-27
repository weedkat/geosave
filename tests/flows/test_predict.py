"""The deployment entry point waits for a portable processor and product writer."""

import pytest

from geosave_flows.configs import CoordinateRegionConfig
from geosave_flows.predict import predict_region


def test_region_prediction_is_explicitly_unsettled(tmp_path):
    region = CoordinateRegionConfig(
        kind="coordinates", latitude=-6.2, longitude=106.8, shape=8, resolution=10
    )
    with pytest.raises(NotImplementedError, match="product writer"):
        predict_region.fn("local-release", region, str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()
