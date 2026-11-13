"""@chain_step derives a step's keys from the method's own signature and return."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from geosave.model.chain import chain_step
from geosave.model.chain.step import Step


def contract(method) -> Step:
    declaration = getattr(method, "_chain_step")
    assert isinstance(declaration, Step)
    return declaration


def invoke(module, method, context):
    return contract(getattr(module, method)).invoke(module, context)


class Encode(nn.Module):
    @chain_step()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        feature_map = image * 2
        return feature_map


class Optional(nn.Module):
    @chain_step()
    def run(
        self, image: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        masked = image if mask is None else image * mask
        return masked


@pytest.fixture
def x() -> torch.Tensor:
    return torch.ones(3)


def test_an_absent_optional_falls_back_to_the_method_default(x):
    result = invoke(Optional(), "run", {"image": x})

    assert isinstance(result, dict)
    assert isinstance(result["masked"], torch.Tensor)
    assert result["masked"].tolist() == [1.0, 1.0, 1.0]


def test_a_supplied_optional_is_forwarded(x):
    result = invoke(Optional(), "run", {"image": x, "mask": torch.zeros(3)})

    assert isinstance(result, dict)
    assert isinstance(result["masked"], torch.Tensor)
    assert result["masked"].tolist() == [0.0, 0.0, 0.0]


def test_a_context_key_of_the_wrong_type_is_refused():
    with pytest.raises(TypeError, match="expected Tensor, got list"):
        invoke(Encode(), "encode", {"image": [1, 2, 3]})


def test_returning_an_expression_instead_of_a_name_is_refused():
    with pytest.raises(TypeError, match="must be `return name`"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> torch.Tensor:
                return image * 2


def test_a_declared_output_type_is_checked_against_the_real_value(x):
    class Sneaky(nn.Module):
        @chain_step()
        def run(self, image: torch.Tensor) -> torch.Tensor:
            wrong = [image]
            return wrong  # type: ignore[return-value]

    with pytest.raises(TypeError, match="output 'wrong' expected Tensor"):
        invoke(Sneaky(), "run", {"image": x})


class ExplicitOutputs(nn.Module):
    @chain_step(outputs=("features", "tokens"))
    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, list]:
        if image.sum() > 0:
            return image * 2, [image]
        return image - 1, []


def test_explicit_outputs_name_expressions_and_branches(x):
    method = ExplicitOutputs().encode
    assert contract(method).outputs == {"features": torch.Tensor, "tokens": list}
    positive = contract(method).invoke(ExplicitOutputs(), {"image": x})
    negative = contract(method).invoke(ExplicitOutputs(), {"image": -x})
    assert isinstance(positive, dict)
    assert isinstance(negative, dict)
    torch.testing.assert_close(positive["features"], x * 2)
    torch.testing.assert_close(negative["features"], -x - 1)
    assert isinstance(positive["tokens"], list)
    assert len(positive["tokens"]) == 1
    assert negative["tokens"] == []


def test_decorated_methods_keep_their_ordinary_signature(x):
    torch.testing.assert_close(Encode().encode(x), x * 2)
    torch.testing.assert_close(
        Optional().run(x, mask=torch.zeros_like(x)), torch.zeros_like(x)
    )
