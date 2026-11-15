from __future__ import annotations

import pytest
import torch
from torch import nn

from geosave.ml.chain import chain_step
from geosave.ml.components.catalog import list_models
from geosave.ml.components.published import Published
from geosave.ml.factory import Registry


class Encoder(nn.Module):
    feature_channels: Published[int]

    def __init__(self, feature_channels: int = 2) -> None:
        super().__init__()
        self.feature_channels = feature_channels

    @chain_step()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        features = image * 2
        return features


class Head(nn.Module):
    def __init__(self, feature_channels: int) -> None:
        super().__init__()

    @chain_step(head=True)
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features


class Decoder(nn.Module):
    def __init__(self, feature_channels: int, depth: int) -> None:
        super().__init__()

    @chain_step()
    def decode(self, features: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        decoded = features * scale
        return decoded


@pytest.fixture
def listed_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, Registry[nn.Module]]:
    components = {
        "encoder": Registry(nn.Module),
        "decoder": Registry(nn.Module),
        "head": Registry(nn.Module),
    }
    components["encoder"].register("tiny")(Encoder)
    components["decoder"].register("decode")(Decoder)
    components["head"].register("logits")(Head)
    monkeypatch.setattr("geosave.ml.components.catalog.COMPONENTS", components)
    return components


def test_list_models_shows_each_models_forward_publishing_and_needs(
    listed_stages: dict[str, Registry[nn.Module]],
) -> None:
    table = list_models()

    # Stages read in chain order, the order of their registries.
    assert list(table.index) == [
        ("encoder", "tiny"),
        ("decoder", "decode"),
        ("head", "logits"),
    ]
    assert list(table.columns) == ["forward", "publishes", "requires"]
    assert table.loc[("encoder", "tiny"), "forward"] == "encode(image)"
    assert table.loc[("encoder", "tiny"), "publishes"] == "feature_channels"
    # feature_channels comes from the encoder; depth only the caller can give.
    assert table.loc[("decoder", "decode"), "forward"] == "decode(features, scale=1.0)"
    assert table.loc[("decoder", "decode"), "requires"] == "depth"
    assert table.loc[("head", "logits"), "requires"] == ""


def test_list_models_of_one_model_shows_what_connects_before_and_after(
    listed_stages: dict[str, Registry[nn.Module]],
) -> None:
    connections = list_models("tiny")

    assert connections.to_dict(orient="index") == {
        "decoder": {"before": "", "after": "decode"},
        "head": {"before": "", "after": "logits"},
    }


def test_a_stage_may_take_what_a_stage_further_back_publishes(
    listed_stages: dict[str, Registry[nn.Module]],
) -> None:
    class DeepHead(nn.Module):
        def __init__(self, feature_channels: int) -> None:
            super().__init__()

        @chain_step(head=True)
        def logits(self, decoded: torch.Tensor) -> torch.Tensor:
            return decoded

    listed_stages["head"].register("deep")(DeepHead)

    # The decoder publishes nothing; the encoder before it supplies the width.
    assert list_models("decode").loc["head", "after"] == "deep"
