import torch
from torch import nn

from geosave.ml.builders.scheduler import build_scheduler


def test_scheduler_returns_lightning_metadata() -> None:
    optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1)

    configured = build_scheduler(
        {
            "name": "reduce_on_plateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
            "interval": "epoch",
            "strict": False,
            "scheduler_name": "validation",
        },
        optimizer,
    )

    assert isinstance(
        configured["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    assert configured["scheduler"].optimizer is optimizer
    assert {
        key: configured[key] for key in ("monitor", "interval", "strict", "name")
    } == {
        "monitor": "val_loss",
        "interval": "epoch",
        "strict": False,
        "name": "validation",
    }
