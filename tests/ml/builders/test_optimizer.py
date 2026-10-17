import pytest
from torch import nn

from geosave.ml.builders.optimizer import build_optimizer


def test_optimizer_groups_exact_children_and_remaining_parameters() -> None:
    model = nn.ModuleDict({"encoder": nn.Linear(2, 2), "head": nn.Linear(2, 1)})
    model["encoder"].bias.requires_grad_(False)

    optimizer = build_optimizer(
        {
            "name": "adamw",
            "init_args": {"lr": 1e-3, "weight_decay": 1e-2},
            "groups": {"head": {"lr": 1e-4, "weight_decay": 0.0}},
        },
        model,
    )

    head, remaining = optimizer.param_groups
    assert head["params"] == [model["head"].weight, model["head"].bias]
    assert remaining["params"] == [model["encoder"].weight]
    assert (head["lr"], head["weight_decay"]) == (1e-4, 0.0)


def test_optimizer_rejects_unknown_and_overlapping_groups() -> None:
    first = nn.Linear(1, 1)
    second = nn.Linear(1, 1)
    second.weight = first.weight
    model = nn.ModuleDict({"first": first, "second": second})

    with pytest.raises(ValueError, match="share parameters"):
        build_optimizer(
            {"name": "adamw", "groups": {"first": {}, "second": {}}},
            model,
        )
    with pytest.raises(KeyError, match="Unknown model groups"):
        build_optimizer({"name": "adamw", "groups": {"missing": {}}}, model)


def test_optimizer_rejects_a_model_without_trainable_parameters() -> None:
    model = nn.Linear(1, 1).requires_grad_(False)

    with pytest.raises(ValueError, match="trainable parameters"):
        build_optimizer({"name": "adamw"}, model)
