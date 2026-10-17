from __future__ import annotations

import json

import pytest
import torch

from geosave.model.chain import ModelChain, Published, chain_step
from geosave.model.stages.head.segmentation import SegmentationHead
from geosave.model.registry import build_model, list_models


def test_the_class_count_fixes_the_channel_count() -> None:
    head = SegmentationHead(num_classes=3, feature_channels=4)

    logits = head.forward_logits(torch.rand(2, 4, 8, 8))

    assert head.num_classes == 3
    assert logits.shape == (2, 3, 8, 8)


def test_logits_are_resized_to_the_input_size() -> None:
    head = SegmentationHead(num_classes=2, feature_channels=4, input_size=16)

    assert head.forward_logits(torch.rand(1, 4, 14, 14)).shape == (1, 2, 16, 16)


def test_no_classes_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one class"):
        SegmentationHead(num_classes=0, feature_channels=4)


def test_the_head_ends_a_chain() -> None:
    chain = ModelChain(head=SegmentationHead(num_classes=2, feature_channels=4))

    assert chain(feature_map=torch.rand(1, 4, 8, 8)).shape == (1, 2, 8, 8)


def test_the_registered_name_builds_a_head_whose_class_count_survives_json() -> None:
    assert ("head", "segmentation") in list_models().index

    model = build_model(
        {
            "head": {
                "name": "segmentation",
                "init_args": {
                    "num_classes": 2,
                    "feature_channels": 4,
                },
            }
        }
    )

    saved = json.loads(json.dumps(model.stage_specs))
    assert saved["head"]["init_args"]["num_classes"] == 2
    assert model.head.num_classes == 2


class StubEncoder(torch.nn.Module):
    input_size: Published[int | tuple[int, int]]

    def __init__(self) -> None:
        super().__init__()
        self.input_size = 16

    @chain_step(outputs=("pyramid",))
    def forward_pyramid(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [torch.rand(image.shape[0], 4, 14, 14)]


class StubDecoder(torch.nn.Module):
    feature_channels: Published[int]

    def __init__(self) -> None:
        super().__init__()
        self.feature_channels = 4

    @chain_step(outputs=("feature_map",))
    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        return pyramid[0]


def test_earlier_stages_supply_the_feature_width_and_input_size() -> None:
    model = build_model(
        {
            "encoder": {"class_path": f"{__name__}.StubEncoder"},
            "decoder": {"class_path": f"{__name__}.StubDecoder"},
            "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
        }
    )

    assert model(image=torch.rand(2, 3, 16, 16)).shape == (2, 3, 16, 16)
