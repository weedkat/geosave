from typing import Any

import pytest
from pydantic import ValidationError
from torch import nn

from geosave.model.factory import BuildSpec


def test_imported_factory_preserves_missing_attribute_error() -> None:
    spec = BuildSpec(class_path="torch.nn.MissingLayer")

    with pytest.raises(AttributeError, match="MissingLayer"):
        spec.resolve({}, nn.Module)


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"name": "cross_entropy", "class_path": "torch.nn.CrossEntropyLoss"},
        {"class_path": "CrossEntropyLoss"},
        {"class_path": "torch.nn.CrossEntropyLoss", "unknown": True},
        {"class_path": "torch.nn.CrossEntropyLoss", "init_args": []},
    ],
)
def test_resolve_rejects_invalid_specs(spec: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BuildSpec.model_validate(spec)


def test_build_spec_resolves_registered_names() -> None:
    spec = BuildSpec(name="CrOsS_EnTrOpY")

    assert (
        spec.resolve({"CROSS_ENTROPY": nn.CrossEntropyLoss}, nn.Module)
        is nn.CrossEntropyLoss
    )


def test_build_spec_rejects_unknown_registered_names() -> None:
    spec = BuildSpec(name="missing")

    with pytest.raises(KeyError, match="Unknown name"):
        spec.resolve({"CROSS_ENTROPY": nn.CrossEntropyLoss}, nn.Module)
