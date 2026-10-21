from __future__ import annotations

import math

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset

from modules.regression import mean_teacher
from tests.templates.tasks.regression.mean_teacher.test_data import (
    Scenes as UnlabeledValues,
)
from tests.templates.tasks.regression.supervised.test_data import Chips as Values
from tests.templates.tasks.regression.supervised.test_module import STAGES
from tests.templates.tasks.segmentation.supervised.test_data import (
    FLIP,
    _spec,
)


def _module(**options) -> mean_teacher.Module:
    module = mean_teacher.Module(model_chain=STAGES, **options)
    module.configure_model()
    return module


def _batch(weak: torch.Tensor, strong: torch.Tensor):
    """Return a training batch whose labeled chip is predicted exactly."""
    image = torch.ones(2, 2, 2, 2)
    return {
        "labeled": {"inputs": {"image": image}, "target": image[:, :1].clone()},
        "unlabeled": {"weak": {"image": weak}, "strong": {"image": strong}},
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

    module.optimizer_step(0, 0, torch.optim.SGD(module.model.parameters(), lr=0.0))

    # Half of the teacher's 1 is kept, half moves to the student's 3.
    assert module.teacher.get_submodule("model").factor.item() == pytest.approx(2.0)


class _Batches(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.rand(2, 4, 4)
        return {
            "labeled": {"inputs": {"image": image}, "target": image[:1] * 2},
            "unlabeled": {
                "weak": {"image": image},
                "strong": {"image": image + 0.1},
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


def test_teacher_ema_runs_once_per_optimizer_step_with_accumulation(tmp_path):
    class RecordingTeacher(mean_teacher.Module):
        def __init__(self):
            super().__init__(
                model_chain=STAGES,
                ema=0.5,
                optimizer={"name": "sgd", "init_args": {"lr": 0.1}},
            )
            self.student_steps = []

        def optimizer_step(self, *args, **kwargs):
            super().optimizer_step(*args, **kwargs)
            self.student_steps.append(self.model.get_submodule("model").factor.item())

    module = RecordingTeacher()
    trainer = Trainer(
        accelerator="cpu",
        max_steps=2,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, train_dataloaders=DataLoader(_Batches(), batch_size=1))

    first, second = module.student_steps
    expected = 0.5 * (0.5 * 1.0 + 0.5 * first) + 0.5 * second
    assert module.teacher.get_submodule("model").factor.item() == pytest.approx(
        expected
    )


def test_trainer_consumes_model_ready_samples(tmp_path):
    data = mean_teacher.DataModule(
        _spec(),
        Values(),
        Values(),
        augmentations=FLIP,
        batch_size=2,
        num_workers=0,
        unlabeled=UnlabeledValues(),
    )
    module = mean_teacher.Module(model_chain=STAGES)
    trainer = Trainer(
        accelerator="cpu",
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, datamodule=data)
    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    assert set(Values()[0]) == {"inputs", "target"}
