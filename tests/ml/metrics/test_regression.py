from __future__ import annotations

import pytest
import torch

from geosave_engine.ml.metrics import RegressionMetrics


def test_regression_metrics_score_flat_values() -> None:
    metrics = RegressionMetrics()
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])

    metrics.update(target + 1.0, target)
    scores = metrics.compute()

    assert scores["mae"].item() == pytest.approx(1.0)
    assert scores["rmse"].item() == pytest.approx(1.0)
    assert scores["r2"].item() == pytest.approx(1 - 4 / 5)


def test_regression_metrics_refuse_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown metrics"):
        RegressionMetrics(["mape"])
