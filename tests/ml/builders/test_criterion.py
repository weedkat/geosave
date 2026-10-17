from typing import Any

import pytest
import torch

from geosave.ml.builders.criterion import build_criterion
from geosave.ml.criterion import ProbOhemCrossEntropy2d


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "CrOsS_EnTrOpY", "init_args": {"ignore_index": 255}},
        {
            "class_path": "torch.nn.CrossEntropyLoss",
            "init_args": {"ignore_index": 255},
        },
    ],
)
def test_build_criterion_supports_registered_names_and_class_paths(
    spec: dict[str, Any],
) -> None:
    criterion = build_criterion(spec)
    logits = torch.zeros(2, 3)
    target = torch.tensor([1, 255])

    assert criterion(logits, target).item() == pytest.approx(
        torch.log(torch.tensor(3.0)).item()
    )


def test_build_criterion_exposes_ohem() -> None:
    criterion = build_criterion({"name": "ohem", "init_args": {"ignore_index": 255}})

    assert isinstance(criterion, ProbOhemCrossEntropy2d)


def test_build_criterion_rejects_optimizer_classes() -> None:
    with pytest.raises(TypeError, match="Module subclass"):
        build_criterion({"class_path": "torch.optim.AdamW"})
