from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy

import pytest
import torch
from torch import nn

from geosave.ml.chain import chain_step
from geosave.ml.components.published import Published
from geosave.ml.components.registry import COMPONENTS
from geosave.ml.components.build import build_model
from geosave.ml.factory import Registry


class Encoder(nn.Module):
    feature_channels: Published[int]

    def __init__(self, feature_channels: int = 2) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.factor = nn.Parameter(torch.tensor(2.0))

    @chain_step()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        features = image * self.factor
        return features


class Head(nn.Module):
    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels

    @chain_step(head=True)
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.feature_channels


def make_head(feature_channels: int) -> Head:
    return Head(feature_channels)


def make_registry(name: str, factory: Callable[..., nn.Module]) -> Registry[nn.Module]:
    registry = Registry(nn.Module)
    registry.register(name)(factory)
    return registry


@pytest.fixture
def model_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(COMPONENTS, "encoder", make_registry("test", Encoder))
    monkeypatch.setitem(COMPONENTS, "head", make_registry("test", make_head))


@pytest.mark.parametrize("use_path", [False, True])
def test_build_model_builds_serializable_stages_and_wires_published_values(
    use_path: bool, model_factories: None
) -> None:
    stages: dict[str, dict[str, str]] = {
        "encoder": {"name": "test"},
        "head": {"name": "test"},
    }
    if use_path:
        stages = {
            "encoder": {"class_path": f"{__name__}.Encoder"},
            "head": {"class_path": f"{__name__}.Head"},
        }
    original = deepcopy(stages)
    model = build_model(json.loads(json.dumps(stages)))

    result = model(torch.tensor(3.0))
    assert isinstance(result, torch.Tensor)
    assert result.item() == 8.0
    result.backward()
    encoder = model.get_submodule("encoder")
    assert isinstance(encoder, Encoder)
    gradient = encoder.factor.grad
    assert gradient is not None
    assert gradient.item() == 3.0
    assert stages == original


def test_explicit_arguments_override_published_values(model_factories: None) -> None:
    model = build_model(
        {
            "encoder": {"name": "test"},
            "head": {"name": "test", "init_args": {"feature_channels": 7}},
        }
    )
    head = model.get_submodule("head")
    assert isinstance(head, Head)
    assert head.feature_channels == 7


def test_unknown_constructor_arguments_are_not_dropped(model_factories: None) -> None:
    with pytest.raises(TypeError, match="featre_channels") as caught:
        build_model({"encoder": {"name": "test", "init_args": {"featre_channels": 7}}})
    assert caught.value.__notes__ == ["While building stage 'encoder'"]


def test_build_model_saves_initial_arguments_before_factory_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(channels: list[int], pretrained: bool = True) -> Encoder:
        channels.append(8)
        encoder = Encoder(feature_channels=sum(channels))
        if pretrained:
            nn.init.constant_(encoder.factor, 3.0)
        return encoder

    monkeypatch.setitem(COMPONENTS, "encoder", make_registry("test", factory))
    model = build_model(
        {"encoder": {"name": "test", "init_args": {"channels": [2, 4]}}}
    )

    encoder = model.get_submodule("encoder")
    assert isinstance(encoder, Encoder)
    assert encoder.feature_channels == 14
    assert encoder.factor.item() == 3.0
    assert model.stage_specs == {
        "encoder": {
            "name": "test",
            "init_args": {"channels": [2, 4], "pretrained": False},
        }
    }


def test_build_model_accepts_classes_and_wires_published_values() -> None:
    model = build_model({"encoder": Encoder, "head": Head})

    assert model(torch.tensor(3.0)).item() == 8.0
    assert model.stage_specs == {
        "encoder": {
            "class_path": f"{__name__}.Encoder",
            "init_args": {"feature_channels": 2},
        },
        "head": {
            "class_path": f"{__name__}.Head",
            "init_args": {"feature_channels": 2},
        },
    }


def test_build_model_takes_a_built_stage_and_wires_what_it_publishes() -> None:
    encoder = Encoder(feature_channels=5)

    model = build_model({"encoder": encoder, "head": Head})

    assert model.get_submodule("encoder") is encoder
    assert model(torch.tensor(3.0)).item() == 11.0


def test_a_chain_holding_a_built_stage_has_no_recipe_to_export() -> None:
    model = build_model({"encoder": Encoder(), "head": Head})

    with pytest.raises(ValueError, match="'encoder'"):
        model.stage_specs
