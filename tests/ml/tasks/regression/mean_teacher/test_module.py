from __future__ import annotations

import math

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset

from geosave.ml.tasks.regression import mean_teacher
from tests.ml.tasks.regression.supervised.test_module import STAGES


def _module(**options) -> mean_teacher.Module:
    module = mean_teacher.Module(model_chain=STAGES, **options)
    module.configure_model()
    return module


def _batch(weak: torch.Tensor, strong: torch.Tensor):
    """Return a training batch whose labeled chip is predicted exactly."""
    image = torch.ones(2, 2, 2, 2)
    valid = torch.ones(2, 2, 2, dtype=torch.bool)
    return {
        "labeled": {"inputs": {"image": image}, "target": image[:, :1].clone(), "valid": valid},
        "unlabeled": {"weak": {"image": weak}, "strong": {"image": strong}, "valid": valid},
    }


def test_the_consistency_weight_ramps_up_then_holds() -> None:
    module = _module(unlabeled_weight=2.0, rampup_steps=10)

    assert module.consistency_weight() == pytest.approx(2.0 * math.exp(-5.0))


def test_views_the_teacher_agrees_with_add_no_loss() -> None:
    module = _module(rampup_steps=0)
    weak = torch.rand(2, 2, 2, 2)

    loss = module.training_step(_batch(weak, weak.clone()), 0)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_strong_views_are_pulled_toward_the_teacher() -> None:
    module = _module(rampup_steps=0)
    weak = torch.ones(2, 2, 2, 2)

    loss = module.training_step(_batch(weak, weak * 3.0), 0)

    # Predicted 3 against the teacher's 1, squared.
    assert loss.item() == pytest.approx(4.0)


def test_the_teacher_moves_toward_the_student() -> None:
    module = _module(ema=0.5)
    module.model.get_submodule("model").factor.data.fill_(3.0)

    module.on_train_batch_end(None, None, 0)

    # Half of the teacher's 1 is kept, half moves to the student's 3.
    assert module.teacher.get_submodule("model").factor.item() == pytest.approx(2.0)


class _Batches(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.rand(2, 4, 4)
        valid = torch.ones(4, 4, dtype=torch.bool)
        return {
            "labeled": {"inputs": {"image": image}, "target": image[:1] * 2, "valid": valid},
            "unlabeled": {
                "weak": {"image": image},
                "strong": {"image": image + 0.1},
                "valid": valid,
            },
        }


def test_lightning_fits_a_few_steps(tmp_path) -> None:
    module = mean_teacher.Module(
        model_chain=STAGES, optimizer={"name": "sgd", "init_args": {"lr": 0.1}}
    )
    trainer = Trainer(
        accelerator="cpu",
        max_steps=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, train_dataloaders=DataLoader(_Batches(), batch_size=2))

    assert trainer.global_step == 3
    assert {"train_loss", "train_unlabeled_loss"} <= set(trainer.callback_metrics)
