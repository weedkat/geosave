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


def test_a_none_context_value_counts_as_missing():
    with pytest.raises(KeyError, match="missing input 'image'"):
        invoke(Encode(), "encode", {"image": None})


@pytest.mark.parametrize("empty", [False, True])
def test_terminal_native_list(x, empty):
    class Detector(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
            return [] if empty else [{"boxes": image}, {"boxes": image + 1}]

    result = invoke(Detector(), "forward", {"image": x})
    assert len(result) == (0 if empty else 2)
    if not empty:
        assert result[0]["boxes"] is x
        torch.testing.assert_close(result[1]["boxes"], x + 1)


def test_terminal_native_dict(x):
    class Classifier(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"scores": image}

    result = invoke(Classifier(), "forward", {"image": x})
    assert result["scores"] is x


def test_terminal_tuple_is_not_unpacked(x):
    class Reconstruction(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return image, image + 1

    result = invoke(Reconstruction(), "forward", {"image": x})
    assert isinstance(result, tuple)
    assert result[0] is x


def test_terminal_outer_type_mismatch(x):
    class Wrong(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
            return image  # type: ignore[return-value]

    with pytest.raises(TypeError, match="result expected list"):
        invoke(Wrong(), "forward", {"image": x})


@pytest.mark.parametrize("as_list", [False, True])
def test_terminal_union_validates_outer_types(x, as_list):
    class Choice(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
            return [image] if as_list else image

    result = invoke(Choice(), "forward", {"image": x})
    assert (result[0] if as_list else result) is x


def test_two_return_statements_are_refused():
    with pytest.raises(TypeError, match="exactly one return statement"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> torch.Tensor:
                if image.sum() > 0:
                    early = image
                    return early
                late = image
                return late


def test_returning_an_expression_instead_of_a_name_is_refused():
    with pytest.raises(TypeError, match="must be `return name`"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> torch.Tensor:
                return image * 2


def test_consuming_and_producing_one_key_reads_as_a_self_cycle():
    with pytest.raises(TypeError, match="self-cycle"):

        class Bad(nn.Module):
            @chain_step()
            def run(self, image: torch.Tensor) -> torch.Tensor:
                image = image * 2
                return image


def test_a_declared_output_type_is_checked_against_the_real_value(x):
    class Sneaky(nn.Module):
        @chain_step()
        def run(self, image: torch.Tensor) -> torch.Tensor:
            wrong = [image]
            return wrong  # type: ignore[return-value]

    with pytest.raises(TypeError, match="output 'wrong' expected Tensor"):
        invoke(Sneaky(), "run", {"image": x})


def test_a_class_with_no_readable_source_is_refused_with_a_clear_message():
    generated = (
        "import torch\n"
        "import torch.nn as nn\n"
        "from geosave.model.chain import chain_step\n"
        "class Generated(nn.Module):\n"
        "    @chain_step()\n"
        "    def run(self, image: torch.Tensor) -> torch.Tensor:\n"
        "        out = image\n"
        "        return out\n"
    )

    with pytest.raises(TypeError, match="no source is available"):
        exec(compile(generated, "<generated>", "exec"), {})


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


def test_explicit_outputs_do_not_require_source(x):
    namespace = {}
    exec(
        "import torch\n"
        "from geosave.model.chain import chain_step\n"
        "@chain_step(outputs=('encoded',))\n"
        "def encode(self, image: torch.Tensor) -> torch.Tensor:\n"
        "    return image * 3\n",
        namespace,
    )
    result = namespace["encode"](None, x)
    torch.testing.assert_close(result, x * 3)


def test_decorated_methods_keep_their_ordinary_signature(x):
    torch.testing.assert_close(Encode().encode(x), x * 2)
    torch.testing.assert_close(
        Optional().run(x, mask=torch.zeros_like(x)), torch.zeros_like(x)
    )


@pytest.mark.parametrize("outputs", [("a", "a")])
def test_invalid_explicit_output_names_are_rejected(outputs):
    with pytest.raises(TypeError, match="non-empty tuple of unique names"):

        @chain_step(outputs=outputs)
        def encode(self, image: torch.Tensor) -> torch.Tensor:
            return image
