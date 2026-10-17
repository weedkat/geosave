from __future__ import annotations

import ast
import json
import subprocess
import sys
from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn

from geosave.model.chain import Published, chain_step
from geosave.model.factory import BuildSpec
from geosave.model.registry import (
    MODEL_REGISTRY,
    build_model,
    list_models,
    register_model,
)


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


@pytest.fixture
def model_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    list_models()
    monkeypatch.setitem(MODEL_REGISTRY, "encoder", {"TEST": Encoder})
    monkeypatch.setitem(MODEL_REGISTRY, "head", {"TEST": make_head})


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


@pytest.mark.parametrize("error_type", [TypeError, ValueError, RuntimeError])
def test_build_model_preserves_constructor_errors(monkeypatch, error_type) -> None:
    failure = error_type("invalid layout")

    def failing_encoder() -> nn.Module:
        raise failure

    list_models()
    monkeypatch.setitem(MODEL_REGISTRY, "encoder", {"FAIL": failing_encoder})

    with pytest.raises(error_type) as caught:
        build_model({"encoder": {"name": "fail"}})

    assert caught.value is failure
    assert failure.__notes__ == ["While building stage 'encoder'"]


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_model({})


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"name": "test", "class_path": f"{__name__}.Head"},
        {"name": "missing"},
        {"class_path": "Head"},
        {"name": "test", "unknown": True},
        {"name": "test", "init_args": []},
    ],
)
def test_model_chain_rejects_malformed_stage_selectors(
    spec: dict[str, object], model_factories: None
) -> None:
    with pytest.raises((TypeError, ValueError, KeyError)):
        build_model({"encoder": spec})


def test_build_model_records_an_independent_resolved_recipe(
    model_factories: None,
) -> None:
    specs: dict[str, dict[str, str]] = {
        "encoder": {"name": "test"},
        "head": {"name": "test"},
    }

    model = build_model(specs)
    specs.clear()
    assert list(model.stage_specs) == ["encoder", "head"]


def test_build_model_accepts_validated_build_specs(model_factories: None) -> None:
    model = build_model(
        {
            "encoder": BuildSpec(name="test"),
            "head": BuildSpec(name="test"),
        }
    )

    torch.testing.assert_close(model(torch.tensor(3.0)), torch.tensor(8.0))
    copied = model.stage_specs
    copied.clear()
    assert list(model.stage_specs) == ["encoder", "head"]


def test_register_model_accepts_factory_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(MODEL_REGISTRY, "test", {})
    decorated = register_model("test", "factory")(make_head)
    assert decorated is make_head
    assert MODEL_REGISTRY["test"]["FACTORY"] is make_head


def test_build_model_captures_defaults_and_preserves_factory_selector(
    model_factories: None,
) -> None:
    model = build_model({"encoder": {"name": "test"}, "head": {"name": "test"}})

    assert model.stage_specs == {
        "encoder": {"name": "test", "init_args": {"feature_channels": 2}},
        "head": {"name": "test", "init_args": {"feature_channels": 2}},
    }
    assert isinstance(model.get_submodule("encoder"), Encoder)
    assert isinstance(model.get_submodule("head"), Head)


def test_build_model_saves_initial_arguments_before_factory_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(channels: list[int], pretrained: bool = True) -> Encoder:
        channels.append(8)
        encoder = Encoder(feature_channels=sum(channels))
        if pretrained:
            nn.init.constant_(encoder.factor, 3.0)
        return encoder

    monkeypatch.setitem(MODEL_REGISTRY, "encoder", {"TEST": factory})
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


@pytest.mark.parametrize(
    ("encoder", "inputs"),
    [
        (
            {"name": "dinov3", "init_args": {"pretrained": False, "in_channels": 4}},
            ["image"],
        ),
        (
            {
                "name": "prithvi_tl",
                "init_args": {
                    "model_name": "prithvi_eo_v2_tiny_tl",
                    "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
                },
            },
            ["image", "temporal_coords", "location_coords"],
        ),
        (
            {
                "name": "clay",
                "init_args": {
                    "model_name": "clay_v15_tiny",
                    "in_channels": 4,
                    "waves": [0.49, 0.56, 0.665, 0.842],
                    "gsd": 10.0,
                },
            },
            ["image"],
        ),
    ],
)
def test_registered_encoders_chain_into_a_decoder_and_head(
    encoder: dict[str, Any], inputs: list[str]
) -> None:
    model = build_model(
        {
            "encoder": deepcopy(encoder),
            "decoder": {"name": "dpt"},
            "head": {
                "name": "segmentation",
                "init_args": {"num_classes": 2},
            },
        }
    )
    assert list(model.inputs) == inputs


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


def test_build_model_builds_a_class_no_import_path_reaches() -> None:
    class Local(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image + 1

    assert build_model({"head": Local})(torch.tensor(1.0)).item() == 2.0


def test_build_model_mixes_classes_with_stage_specifications(
    model_factories: None,
) -> None:
    model = build_model({"encoder": Encoder, "head": {"name": "test"}})

    assert model(torch.tensor(3.0)).item() == 8.0


def test_build_model_refuses_a_class_that_is_not_a_module() -> None:
    with pytest.raises(TypeError, match="nn.Module"):
        build_model({"encoder": dict})


def test_build_model_takes_a_built_stage_and_wires_what_it_publishes() -> None:
    encoder = Encoder(feature_channels=5)

    model = build_model({"encoder": encoder, "head": Head})

    assert model.get_submodule("encoder") is encoder
    assert model(torch.tensor(3.0)).item() == 11.0


def test_a_chain_holding_a_built_stage_has_no_recipe_to_export() -> None:
    model = build_model({"encoder": Encoder(), "head": Head})

    with pytest.raises(ValueError, match="'encoder'"):
        model.stage_specs


class Decoder(nn.Module):
    def __init__(self, feature_channels: int, depth: int) -> None:
        super().__init__()

    @chain_step()
    def decode(self, features: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        decoded = features * scale
        return decoded


@pytest.fixture
def listed_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    list_models()
    monkeypatch.setattr("geosave.model.registry.MODEL_REGISTRY", {})
    from geosave.model.registry import MODEL_REGISTRY as empty

    empty["head"] = {"LOGITS": Head}
    empty["encoder"] = {"TINY": Encoder}
    empty["decoder"] = {"DECODE": Decoder}


def test_list_models_shows_each_models_forward_publishing_and_needs(
    listed_stages: None,
) -> None:
    table = list_models()

    # Stages read in chain order, whatever order they registered in.
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
    listed_stages: None,
) -> None:
    connections = list_models("tiny")

    assert connections.to_dict(orient="index") == {
        "decoder": {"before": "", "after": "decode"},
        "head": {"before": "", "after": "logits"},
    }


def test_a_stage_may_take_what_a_stage_further_back_publishes(
    listed_stages: None,
) -> None:
    from geosave.model.registry import MODEL_REGISTRY as registry

    class DeepHead(nn.Module):
        def __init__(self, feature_channels: int) -> None:
            super().__init__()

        @chain_step(head=True)
        def logits(self, decoded: torch.Tensor) -> torch.Tensor:
            return decoded

    registry["head"]["DEEP"] = DeepHead

    # The decoder publishes nothing; the encoder before it supplies the width.
    assert list_models("decode").loc["head", "after"] == "deep"


def test_list_models_finds_built_in_connections() -> None:
    connections = list_models("unet")

    assert connections.loc["encoder", "before"] == (
        "clay, dinov3, prithvi, prithvi_tl, small_cnn"
    )
    # Other test modules may register heads of their own.
    assert {"regression", "segmentation"} <= set(
        connections.loc["head", "after"].split(", ")
    )


def test_list_models_refuses_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="no model named 'nope'"):
        list_models("nope")


def test_redefining_a_registered_class_replaces_it(monkeypatch) -> None:
    # A notebook cell run twice defines a new class of the same name and module.
    monkeypatch.setitem(MODEL_REGISTRY, "rerun", {})

    def define() -> type[nn.Module]:
        class Encoder(nn.Module):
            pass

        return register_model("rerun", "tiny")(Encoder)

    define()
    second = define()

    assert MODEL_REGISTRY["rerun"]["TINY"] is second


def test_another_class_under_a_taken_name_is_refused(monkeypatch) -> None:
    monkeypatch.setitem(MODEL_REGISTRY, "rerun", {})
    register_model("rerun", "tiny")(Encoder)

    with pytest.raises(ValueError, match="already identifies"):
        register_model("rerun", "tiny")(Head)


@pytest.mark.slow
def test_every_stage_package_registers_its_models():
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            "from geosave.model.registry import list_models; print(list_models().index.tolist())",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert set(ast.literal_eval(result.stdout.strip().splitlines()[-1])) == {
        ("decoder", "dpt"),
        ("decoder", "unet"),
        ("encoder", "clay"),
        ("encoder", "dinov3"),
        ("encoder", "prithvi"),
        ("encoder", "prithvi_tl"),
        ("head", "classification"),
        ("head", "regression"),
        ("head", "segmentation"),
        ("model", "ibm_granite_biomass"),
    }
