from __future__ import annotations


import torch

from geosave.model.chain import Published, chain_step
from geosave.model.registry import build_model


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
