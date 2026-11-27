from functools import partial

import pytest
import torch
from torch import nn

from geosave.ml.torch.training.optim import configure_optimizers


class Chain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)


def test_named_children_get_their_own_options_and_the_rest_one_group():
    model = Chain()

    optimizer = configure_optimizers(
        model, partial(torch.optim.SGD, lr=0.1), {"encoder": {"lr": 0.01}}, None, None
    )

    encoder, rest = optimizer.param_groups
    assert encoder["params"] == list(model.encoder.parameters())
    assert encoder["lr"] == 0.01
    assert rest["params"] == list(model.head.parameters())
    assert rest["lr"] == 0.1


def test_scheduler_comes_with_lightning_options():
    configured = configure_optimizers(
        Chain(),
        partial(torch.optim.SGD, lr=0.1),
        None,
        partial(torch.optim.lr_scheduler.StepLR, step_size=2),
        {"interval": "step", "frequency": 1},
    )

    schedule = configured["lr_scheduler"]
    assert isinstance(schedule["scheduler"], torch.optim.lr_scheduler.StepLR)
    assert schedule["scheduler"].optimizer is configured["optimizer"]
    assert (schedule["interval"], schedule["frequency"]) == ("step", 1)


def test_a_group_may_not_set_params():
    with pytest.raises(ValueError, match="overrides or shares"):
        configure_optimizers(
            Chain(), torch.optim.SGD, {"encoder": {"params": []}}, None, None
        )
