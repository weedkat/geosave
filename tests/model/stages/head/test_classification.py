from __future__ import annotations

import torch

from geosave.model.chain import ModelChain
from geosave.model.stages.head.classification import ClassificationHead
from geosave.model.registry import list_models


def _pyramid() -> list[torch.Tensor]:
    return [torch.rand(2, 8, 14, 14), torch.rand(2, 16, 7, 7)]


def test_the_head_ends_a_chain_under_its_registered_name() -> None:
    assert ("head", "classification") in list_models().index

    chain = ModelChain(head=ClassificationHead(num_classes=2, pyramid_channels=[8, 16]))

    assert chain(pyramid=_pyramid()).shape == (2, 2)
