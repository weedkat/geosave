from __future__ import annotations

from typing import Annotated

import pytest
import torch
from torch import nn

from geosave.ml.torch.chain.catalog import list_models
from geosave.ml.torch.chain.published import Published
from geosave.ml.torch.chain.registry import Registry

Features = Annotated[torch.Tensor, "features"]
Tokens = Annotated[torch.Tensor, "tokens"]
Decoded = Annotated[torch.Tensor, "decoded"]


class Encoder(nn.Module):
    returns = ("features", "tokens")

    feature_channels: Published[int]

    def __init__(self, feature_channels: int = 2) -> None:
        super().__init__()
        self.feature_channels = feature_channels

    def forward(self, image: torch.Tensor) -> tuple[Features, Tokens]:
        return image * 2, image


class Decoder(nn.Module):
    returns = ("decoded",)

    def __init__(self, feature_channels: int, depth: int) -> None:
        super().__init__()

    def forward(self, features: Features, tokens: Tokens) -> Decoded:
        return features * 2


class Head(nn.Module):
    returns = ("logits",)

    def __init__(self, feature_channels: int) -> None:
        super().__init__()

    def forward(self, decoded: Decoded) -> torch.Tensor:
        return decoded.sum()


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Registry:
    registry = Registry()
    registry.register("tiny")(Encoder)
    registry.register("decode")(Decoder)
    registry.register("logits")(Head)
    monkeypatch.setattr("geosave.ml.torch.chain.catalog.MODULES", registry)
    return registry


def test_list_models_shows_each_models_forward_publishing_and_needs(
    registry: Registry,
) -> None:
    table = list_models()

    # Classes registered outside the library's module packages are custom.
    assert list(table.index) == [
        ("custom", "tiny"),
        ("custom", "decode"),
        ("custom", "logits"),
    ]
    assert list(table.columns) == ["class", "forward", "publishes", "requires"]

    models = table.loc["custom"]
    assert models.loc["tiny", "class"] == "Encoder"
    assert models.loc["tiny", "forward"] == "forward(image) -> features, tokens"
    assert models.loc["tiny", "publishes"] == "feature_channels"
    # feature_channels comes from the encoder; depth only the caller can give.
    assert models.loc["decode", "requires"] == "depth"
    assert models.loc["logits", "forward"] == "forward(decoded) -> logits"
    assert models.loc["logits", "requires"] == ""


def test_list_models_groups_library_models_by_family_in_chain_order() -> None:
    table = list_models()

    families = list(table.index.get_level_values("family").unique())
    assert families == ["encoder", "decoder", "head", "monolith"]
    assert list(table.loc["decoder"].index) == ["dpt", "unet"]
    assert table.loc[("head", "segmentation"), "class"] == "SegmentationHead"


def test_list_models_of_one_model_shows_what_runs_before_and_after_it(
    registry: Registry,
) -> None:
    connections = list_models("decode")

    assert connections.to_dict(orient="index") == {
        ("before", "tiny"): {"values": "features, tokens"},
        ("after", "logits"): {"values": "decoded"},
    }


def test_a_model_connecting_to_nothing_lists_no_rows(registry: Registry) -> None:
    class Alone(nn.Module):
        returns = ("logits",)

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return pixels.sum()

    registry.register("alone")(Alone)

    assert list_models("alone").empty
    assert (
        list_models().loc[("custom", "alone"), "forward"] == "forward(pixels) -> logits"
    )


def test_a_model_missing_a_value_another_model_returns_cannot_run_before(
    registry: Registry,
) -> None:
    class FeaturesOnly(nn.Module):
        returns = ("features",)

        def forward(self, image: torch.Tensor) -> Features:
            return image

    registry.register("features_only")(FeaturesOnly)

    # The decoder also reads tokens, which only the tiny encoder returns.
    assert list(list_models("decode").index) == [
        ("before", "tiny"),
        ("after", "logits"),
    ]
