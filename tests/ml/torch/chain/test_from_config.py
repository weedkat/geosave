from __future__ import annotations

import json
from copy import deepcopy
from typing import Annotated
from uuid import uuid4

import pytest
import torch
from huggingface_hub import HfApi
from torch import nn

from geosave.ml.torch.chain import ModelChain
from geosave.ml.torch.chain.published import Published
from geosave.ml.torch.chain.registry import Registry

Features = Annotated[torch.Tensor, "features"]
Decoded = Annotated[torch.Tensor, "decoded"]


class Encoder(nn.Module):
    returns = ("features",)

    feature_channels: Published[int]
    input_size: Published[int]

    def __init__(
        self, feature_channels: int = 2, input_size: int = 8, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.input_size = input_size
        self.factor = nn.Parameter(torch.tensor(2.0))

    def forward(self, image: torch.Tensor) -> Features:
        return image * self.factor


class Decoder(nn.Module):
    returns = ("decoded",)

    feature_channels: Published[int]

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels * 3

    def forward(self, features: Features) -> Decoded:
        return features + 1


class Head(nn.Module):
    returns = ("logits",)

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels

    def forward(self, features: Features) -> torch.Tensor:
        return features + self.feature_channels


class DecodedHead(nn.Module):
    returns = ("logits",)

    def __init__(self, feature_channels: int, input_size: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.input_size = input_size

    def forward(self, decoded: Decoded) -> torch.Tensor:
        return decoded + self.feature_channels


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> Registry:
    registry = Registry()
    registry.register("encoder")(Encoder)
    registry.register("decoder")(Decoder)
    registry.register("head")(Head)
    registry.register("decoded_head")(DecodedHead)
    monkeypatch.setattr("geosave.ml.torch.chain.registry.MODULES", registry)
    return registry


@pytest.mark.parametrize("use_path", [False, True])
def test_from_config_wires_published_values(
    use_path: bool, registered: Registry
) -> None:
    modules: dict[str, dict[str, str]] = {
        "encoder": {"name": "encoder"},
        "head": {"name": "head"},
    }
    if use_path:
        modules = {
            "encoder": {"class_path": f"{__name__}.Encoder"},
            "head": {"class_path": f"{__name__}.Head"},
        }
    config = json.loads(json.dumps(modules))
    original = deepcopy(config)

    model = ModelChain.from_config(config)
    result = model(image=torch.tensor(3.0))

    assert isinstance(result, torch.Tensor)
    assert result.item() == 8.0
    result.backward()
    encoder = model.get_submodule("encoder")
    assert isinstance(encoder, Encoder)
    assert encoder.factor.grad is not None
    assert encoder.factor.grad.item() == 3.0
    assert config == original


def test_the_nearest_earlier_publisher_supplies_an_argument(
    registered: Registry,
) -> None:
    model = ModelChain.from_config(
        {
            "encoder": {"name": "encoder"},
            "decoder": {"name": "decoder"},
            "head": {"name": "decoded_head"},
        }
    )

    head = model.get_submodule("head")
    assert isinstance(head, DecodedHead)
    # The decoder republishes the width; only the encoder publishes the size.
    assert head.feature_channels == 6
    assert head.input_size == 8


def test_explicit_arguments_override_published_values(registered: Registry) -> None:
    model = ModelChain.from_config(
        {
            "encoder": {"name": "encoder"},
            "head": {"name": "head", "init_args": {"feature_channels": 7}},
        }
    )

    head = model.get_submodule("head")
    assert isinstance(head, Head)
    assert head.feature_channels == 7


def test_unknown_constructor_arguments_are_not_dropped(registered: Registry) -> None:
    with pytest.raises(TypeError, match="featre_channels") as caught:
        ModelChain.from_config(
            {"encoder": {"name": "encoder", "init_args": {"featre_channels": 7}}}
        )

    assert caught.value.__notes__ == ["While building module 'encoder'"]


def test_stages_that_do_not_connect_are_refused_before_any_is_built(
    registered: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = []
    monkeypatch.setattr(Encoder, "__init__", lambda self, **_: built.append(self))

    with pytest.raises(
        TypeError,
        match="'head' reads decoded, which no input or earlier module provides",
    ):
        ModelChain.from_config(
            {
                "encoder": {"name": "encoder"},
                "head": {"name": "decoded_head"},
                "decoder": {"name": "decoder"},
            }
        )

    assert built == []


def test_any_stage_name_selects_by_registered_name(registered: Registry) -> None:
    model = ModelChain.from_config({"backbone": {"name": "encoder"}})

    assert isinstance(model.get_submodule("backbone"), Encoder)


def test_the_release_configuration_writes_out_every_argument(
    registered: Registry,
) -> None:
    model = ModelChain.from_config(
        {
            "encoder": {"name": "encoder"},
            "head": {
                "class_path": f"{__name__}.Head",
                "init_args": {"feature_channels": 1},
            },
        }
    )

    assert model.config == {
        "encoder": {
            "name": "encoder",
            "init_args": {"feature_channels": 2, "input_size": 8, "pretrained": False},
        },
        "head": {
            "class_path": f"{__name__}.Head",
            "init_args": {"feature_channels": 1},
        },
    }


def test_direct_chain_uses_the_given_instances_and_has_no_release_recipe() -> None:
    encoder = Encoder(feature_channels=5)
    head = Head(feature_channels=encoder.feature_channels)

    model = ModelChain(encoder=encoder, head=head)

    assert model.get_submodule("encoder") is encoder
    assert model.get_submodule("head") is head
    assert model(image=torch.tensor(3.0)).item() == 11.0
    with pytest.raises(ValueError, match="Use ModelChain.from_config"):
        model.config


def test_from_config_builds_on_the_requested_chain_class(registered: Registry) -> None:
    class CustomChain(ModelChain):
        pass

    config = json.loads('{"encoder": {"name": "encoder"}, "head": {"name": "head"}}')
    model = CustomChain.from_config(config)

    assert type(model) is CustomChain
    assert [name for name, _ in model.named_children()] == ["encoder", "head"]
    assert model(image=torch.tensor(3.0)).item() == 8.0


def test_a_saved_chain_loads_back_with_its_weights(
    tmp_path, registered: Registry
) -> None:
    chain = ModelChain.from_config(
        {
            "encoder": {"name": "encoder"},
            "decoder": {"name": "decoder"},
            "head": {"name": "decoded_head"},
        }
    )
    encoder = chain.get_submodule("encoder")
    assert isinstance(encoder, Encoder)
    with torch.no_grad():
        encoder.factor.fill_(5.0)

    chain.save_pretrained(tmp_path)
    loaded = ModelChain.from_pretrained(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    assert json.loads((tmp_path / "config.json").read_text()) == chain.config
    assert "library_name: geosave" in (tmp_path / "README.md").read_text()
    # features = 5 * 3; decoded = 16; the head adds the decoder's width, 6.
    assert loaded(image=torch.tensor(3.0)).item() == 22.0
    assert chain(image=torch.tensor(3.0)).item() == 22.0


LIBRARY_CHAINS = {
    "small_cnn": {
        "encoder": {"name": "small_cnn", "init_args": {"input_size": 64}},
        "decoder": {"name": "unet"},
        "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
    },
    "dinov3": {
        "encoder": {
            "name": "dinov3",
            "init_args": {
                "model_name": "vit_small_patch16_dinov3.lvd1689m",
                "pretrained": False,
                "input_size": 64,
            },
        },
        "decoder": {"name": "dpt"},
        "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
    },
}


@pytest.mark.parametrize("chain_name", list(LIBRARY_CHAINS))
def test_a_saved_library_chain_predicts_the_same_after_loading(
    tmp_path, chain_name: str
) -> None:
    chain = ModelChain.from_config(LIBRARY_CHAINS[chain_name]).eval()
    image = torch.rand(1, 3, 64, 64)

    chain.save_pretrained(tmp_path)
    loaded = ModelChain.from_pretrained(tmp_path).eval()

    assert loaded.config == chain.config
    torch.testing.assert_close(loaded(image=image), chain(image=image))


@pytest.mark.integration
def test_a_pushed_chain_loads_back_from_the_hub() -> None:
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        pytest.skip("Not logged in to the Hugging Face Hub")

    repo_id = f"{user}/geosave-chain-test-{uuid4().hex[:8]}"
    chain = ModelChain.from_config(LIBRARY_CHAINS["small_cnn"]).eval()
    image = torch.rand(1, 3, 64, 64)

    try:
        chain.push_to_hub(repo_id, private=True)
        info = api.model_info(repo_id)
        loaded = ModelChain.from_pretrained(repo_id, revision=info.sha).eval()

        assert info.private
        assert loaded.config == chain.config
        torch.testing.assert_close(loaded(image=image), chain(image=image))
    finally:
        api.delete_repo(repo_id, missing_ok=True)
