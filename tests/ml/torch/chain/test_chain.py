"""ModelChain runs modules in order over one record of named values."""

from __future__ import annotations

import copy
import pickle
from typing import Annotated

import pytest
import torch
import torch.nn as nn

from geosave.ml.torch.chain import ModelChain

Features = Annotated[torch.Tensor, "features"]
Decoded = Annotated[torch.Tensor, "decoded"]


class Encode(nn.Module):
    returns = ("features",)

    def forward(self, image: torch.Tensor) -> Features:
        return image * 2


class Decode(nn.Module):
    returns = ("decoded",)

    def forward(self, features: Features) -> Decoded:
        return features + 1


class Head(nn.Module):
    returns = ("logits",)

    def forward(self, decoded: Decoded) -> torch.Tensor:
        return decoded.sum(0, keepdim=True)


class Split(nn.Module):
    returns = ("low", "high")

    def forward(self, image: torch.Tensor) -> tuple[Features, Features]:
        return image - 1, image + 1


class ReadHigh(nn.Module):
    returns = ("logits",)

    def forward(self, high: Features) -> torch.Tensor:
        return high * 10


@pytest.fixture
def x() -> torch.Tensor:
    return torch.arange(4, dtype=torch.float32)


def read_inputs(chain: ModelChain) -> list[tuple[str, object]]:
    inputs = []
    for parameter in chain.inputs:
        inputs.append((parameter.name, parameter.annotation))
    return inputs


def test_a_chain_calls_its_modules_in_order(x):
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    assert read_inputs(chain) == [("image", torch.Tensor)]
    assert chain(image=x).tolist() == [16.0]


def test_a_module_reads_returned_values_by_name(x):
    chain = ModelChain(encoder=Split(), head=ReadHigh())

    torch.testing.assert_close(chain(image=x), (x + 1) * 10)


def test_two_values_of_one_type_are_told_apart_by_name(x):
    class Difference(nn.Module):
        returns = ("change",)

        def forward(self, low: Features, high: Features) -> torch.Tensor:
            return high - low

    chain = ModelChain(encoder=Split(), head=Difference())

    torch.testing.assert_close(chain(image=x), torch.full_like(x, 2.0))


def test_a_later_module_reads_an_earlier_modules_value_past_the_previous(x):
    class SkipHead(nn.Module):
        returns = ("logits",)

        def forward(self, decoded: Decoded, features: Features) -> torch.Tensor:
            return decoded + features

    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=SkipHead())

    torch.testing.assert_close(chain(image=x), x * 4 + 1)


def test_a_later_module_reads_the_chains_inputs(x):
    class AddImage(nn.Module):
        returns = ("logits",)

        def forward(self, decoded: Decoded, image: torch.Tensor) -> torch.Tensor:
            return decoded + image

    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=AddImage())

    assert read_inputs(chain) == [("image", torch.Tensor)]
    torch.testing.assert_close(chain(image=x), x * 3 + 1)


def test_a_module_returning_a_name_replaces_its_value(x):
    class Normalize(nn.Module):
        returns = ("image",)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image - 1

    chain = ModelChain(
        normalize=Normalize(), encoder=Encode(), decoder=Decode(), head=Head()
    )

    assert chain(image=x).tolist() == [8.0]


def test_a_lone_model_runs_its_forward():
    class EndToEnd(nn.Module):
        returns = ("logits",)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image * 2

    chain = ModelChain(model=EndToEnd())
    image = torch.ones(1, 2)

    assert read_inputs(chain) == [("image", torch.Tensor)]
    torch.testing.assert_close(chain(image=image), image * 2)


def test_the_first_module_reads_any_values_as_the_chains_inputs(x):
    chain = ModelChain(head=Head())

    assert read_inputs(chain) == [("decoded", Decoded)]
    assert chain(decoded=x).tolist() == [6.0]


def test_reading_a_name_nothing_provides_is_refused():
    with pytest.raises(
        TypeError,
        match="'head' reads decoded, which no input or earlier module provides",
    ):
        ModelChain(encoder=Encode(), head=Head())


def test_modules_written_out_of_order_are_refused():
    with pytest.raises(
        TypeError,
        match="'encoder' reads image, which no input or earlier module provides",
    ):
        ModelChain(decoder=Decode(), encoder=Encode(), head=Head())


def test_a_value_read_as_another_type_is_refused():
    class ReadList(nn.Module):
        returns = ("logits",)

        def forward(self, features: list[Features]) -> torch.Tensor:
            return features[0]

    with pytest.raises(
        TypeError,
        match=r"'head' reads features as list\[features\]; the record holds features",
    ):
        ModelChain(encoder=Encode(), head=ReadList())


def test_a_module_without_returns_is_refused():
    class Unnamed(nn.Module):
        def forward(self, features: Features) -> Decoded:
            return features

    with pytest.raises(TypeError, match="Unnamed declares no `returns`"):
        ModelChain(encoder=Encode(), decoder=Unnamed())


def test_returns_naming_another_number_of_values_is_refused():
    class Miscounted(nn.Module):
        returns = ("low", "high")

        def forward(self, image: torch.Tensor) -> Features:
            return image

    with pytest.raises(
        TypeError, match="Miscounted.returns names 2 values; its forward returns 1"
    ):
        ModelChain(encoder=Miscounted())


def test_a_forward_parameter_with_a_default_is_refused():
    class Scaled(nn.Module):
        returns = ("logits",)

        def forward(self, decoded: Decoded, scale: float = 1.0) -> torch.Tensor:
            return decoded * scale

    with pytest.raises(TypeError, match="'scale' has a default"):
        ModelChain(encoder=Encode(), decoder=Decode(), head=Scaled())


def test_hooks_on_every_module_fire(x):
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())
    fired = []
    for name, module in chain.named_children():
        module.register_forward_hook(lambda *_, name=name: fired.append(name))

    chain(image=x)

    assert fired == ["encoder", "decoder", "head"]


def test_assigning_a_module_replaces_it_in_the_chain(x):
    class Mean(nn.Module):
        returns = ("logits",)

        def forward(self, decoded: Decoded) -> torch.Tensor:
            return decoded.mean(0, keepdim=True)

    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())
    chain.head = Mean()

    assert chain(image=x).tolist() == [4.0]


class Detector(nn.Module):
    returns = ("objects",)

    def forward(self, decoded: Decoded) -> list[dict[str, torch.Tensor]]:
        return [{"boxes": decoded.reshape(1, 4), "scores": decoded[:1]}]


def test_the_last_modules_result_is_returned_unchanged(x):
    detector = Detector()
    captured = []
    detector.register_forward_hook(lambda module, args, output: captured.append(output))

    result = ModelChain(encoder=Encode(), decoder=Decode(), objects=detector)(image=x)

    assert result is captured[0]
    torch.testing.assert_close(result[0]["boxes"], (x * 2 + 1).reshape(1, 4))


def test_gradients_reach_the_inputs(x):
    x.requires_grad_()
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    chain(image=x).sum().backward()

    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))


def test_repr_and_mermaid_show_the_data_flow():
    class SkipHead(nn.Module):
        returns = ("logits",)

        def forward(self, high: Features, image: torch.Tensor) -> torch.Tensor:
            return high + image

    chain = ModelChain(encoder=Split(), head=SkipHead())

    assert "inputs: image: Tensor" in repr(chain)
    assert "  encoder: Split.forward(image) -> low, high" in repr(chain)
    assert "  head: SkipHead.forward(high, image) -> logits" in repr(chain)

    mermaid = chain.to_mermaid()
    assert 'input_image -->|"image: Tensor"| encoder' in mermaid
    assert 'encoder -->|"high: features"| head' in mermaid
    assert 'input_image -->|"image: Tensor"| head' in mermaid
    assert 'head --> result(["result"])' in mermaid


def test_inputs_may_be_given_by_position(x):
    chain = ModelChain(encoder=Encode(), decoder=Decode(), head=Head())

    assert chain(x).tolist() == chain(image=x).tolist()


class LinearEncode(nn.Module):
    returns = ("features",)

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.norm = nn.BatchNorm1d(3)

    def forward(self, image: torch.Tensor) -> Features:
        return self.norm(self.linear(image))


class LinearHead(nn.Module):
    returns = ("logits",)

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, features: Features) -> torch.Tensor:
        return self.linear(features)


@pytest.fixture
def model() -> ModelChain:
    torch.manual_seed(0)
    return ModelChain(encoder=LinearEncode(), head=LinearHead()).eval()


@pytest.fixture
def batch() -> torch.Tensor:
    return torch.rand(5, 4)


def test_a_state_dict_loads_into_a_fresh_chain(model, batch):
    fresh = ModelChain(encoder=LinearEncode(), head=LinearHead()).eval()

    fresh.load_state_dict(model.state_dict())

    assert {
        "encoder.linear.weight",
        "encoder.norm.running_mean",
        "head.linear.bias",
    } <= set(model.state_dict())
    torch.testing.assert_close(fresh(image=batch), model(image=batch))


def test_a_deep_copy_is_an_independent_chain(model, batch):
    copied = copy.deepcopy(model)
    with torch.no_grad():
        copied.get_submodule("head.linear").bias.add_(1.0)

    torch.testing.assert_close(copied(image=batch), model(image=batch) + 1.0)


def test_a_chain_survives_pickling(model, batch):
    restored = pickle.loads(pickle.dumps(model))

    torch.testing.assert_close(restored(image=batch), model(image=batch))


def test_dtype_and_mode_reach_every_module(model, batch):
    model.double().train()

    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    assert all(module.training for module in model.modules())
    assert model(image=batch.double()).dtype == torch.float64

    model.eval()
    assert not any(module.training for module in model.modules())


def test_functional_call_runs_with_other_parameters(model, batch):
    zeros = {}
    for name, parameter in model.named_parameters():
        zeros[name] = torch.zeros_like(parameter)

    result = torch.func.functional_call(model, zeros, args=(), kwargs={"image": batch})

    assert torch.count_nonzero(result) == 0


def test_an_exported_chain_matches_eager(model, batch):
    exported = torch.export.export(model, args=(), kwargs={"image": batch})

    torch.testing.assert_close(exported.module()(image=batch), model(image=batch))


def test_a_chain_compiles_as_one_graph(model, batch):
    compiled = torch.compile(model, fullgraph=True, backend="eager")

    assert torch._dynamo.explain(model)(image=batch).graph_break_count == 0
    torch.testing.assert_close(compiled(image=batch), model(image=batch))
