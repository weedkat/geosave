from __future__ import annotations


import torch

from geosave.ml.torch.typing import FeatureMap
from geosave.ml.torch.chain import ModelChain
from geosave.ml.torch.chain.published import Published


class StubEncoder(torch.nn.Module):
    returns = ("pyramid",)

    input_size: Published[int | tuple[int, int]]

    def __init__(self) -> None:
        super().__init__()
        self.input_size = 16

    def forward(self, image: torch.Tensor) -> list[FeatureMap]:
        pyramid = [torch.rand(image.shape[0], 4, 14, 14)]
        return pyramid


class StubDecoder(torch.nn.Module):
    returns = ("feature_map",)

    feature_channels: Published[int]

    def __init__(self) -> None:
        super().__init__()
        self.feature_channels = 4

    def forward(self, pyramid: list[FeatureMap]) -> FeatureMap:
        feature_map = pyramid[0]
        return feature_map


def test_earlier_modules_supply_the_feature_width_and_input_size() -> None:
    model = ModelChain.from_config(
        {
            "encoder": {"class_path": f"{__name__}.StubEncoder"},
            "decoder": {"class_path": f"{__name__}.StubDecoder"},
            "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
        }
    )

    assert model(image=torch.rand(2, 3, 16, 16)).shape == (2, 3, 16, 16)
