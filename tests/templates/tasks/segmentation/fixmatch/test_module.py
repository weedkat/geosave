from __future__ import annotations

import math

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset

from modules.segmentation import fixmatch
from tests.templates.tasks.segmentation.fixmatch.test_data import (
    Scenes as UnlabeledLabels,
)
from tests.templates.tasks.segmentation.supervised.test_data import (
    FLIP,
    _spec,
)
from tests.templates.tasks.segmentation.supervised.test_data import (
    Chips as Labels,
)

STAGES = {
    "model": {
        "class_path": "tests.templates.tasks.segmentation.supervised.test_module.SegmentationModel",
        "init_args": {"num_classes": 2},
    }
}


def _batch(weak: torch.Tensor, strong: torch.Tensor):
    """Return a training batch with one labeled chip of class 0."""
    image = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    return {
        "labeled": {
            "inputs": {"image": image},
            "target": target,
        },
        "unlabeled": {"weak": {"image": weak}, "strong": {"image": strong}},
    }


def _module(**options) -> fixmatch.Module:
    module = fixmatch.Module(model_chain=STAGES, **options)
    module.configure_model()
    return module


def test_unconfident_pseudo_labels_add_no_loss() -> None:
    module = _module()
    weak = torch.zeros(1, 2, 4, 4)
    loss = module.training_step(_batch(weak, torch.randn(1, 2, 4, 4)), 0)

    assert loss.item() == pytest.approx(math.log(2))


def test_confident_pseudo_labels_are_averaged_over_all_pixels() -> None:
    module = _module(unlabeled_weight=2.0)
    weak = torch.zeros(1, 2, 4, 4)
    weak[:, 1, 2:] = 10.0

    loss = module.training_step(_batch(weak, torch.zeros(1, 2, 4, 4)), 0)

    # Labeled: log 2. Eight of sixteen pixels are confident, weighted by 2.
    assert loss.item() == pytest.approx(2 * math.log(2))
    loss.backward()
    assert module.model.get_submodule("model").factor.grad is not None


def test_all_confident_pixels_carry_pseudo_labels() -> None:
    module = _module()
    weak = torch.zeros(1, 2, 4, 4)
    weak[:, 1] = 10.0
    loss = module.training_step(_batch(weak, torch.zeros(1, 2, 4, 4)), 0)

    assert loss.item() == pytest.approx(2 * math.log(2))


class _Batches(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.randn(2, 4, 4)
        return {
            "labeled": {
                "inputs": {"image": image},
                "target": torch.zeros(4, 4, dtype=torch.long),
            },
            "unlabeled": {
                "weak": {"image": image},
                "strong": {"image": image + 0.1},
            },
        }


def test_lightning_fits_a_few_steps(tmp_path) -> None:
    module = fixmatch.Module(model_chain=STAGES, confidence=0.5)
    trainer = Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, train_dataloaders=DataLoader(_Batches(), batch_size=2))

    assert trainer.global_step == 2
    assert {"train_loss", "train_unlabeled_loss", "train_confident"} <= set(
        trainer.callback_metrics
    )


def test_trainer_consumes_model_ready_samples(tmp_path):
    data = fixmatch.DataModule(
        _spec(),
        Labels(),
        Labels(),
        augmentations=FLIP,
        batch_size=2,
        num_workers=0,
        unlabeled=UnlabeledLabels(),
    )
    module = fixmatch.Module(model_chain=STAGES)
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
    assert set(Labels()[0]) == {"inputs", "target"}
