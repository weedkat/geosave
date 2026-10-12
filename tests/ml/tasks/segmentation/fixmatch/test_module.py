from __future__ import annotations

import math

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset

from geosave_engine.ml.tasks.segmentation import fixmatch

STAGES = {
    "model": {
        "class_path": "tests.ml.tasks.segmentation.supervised.test_module.SegmentationModel",
        "init_args": {"num_classes": 2},
    }
}


def _batch(weak: torch.Tensor, strong: torch.Tensor, valid: torch.Tensor):
    """Return a training batch with one labeled chip of class 0."""
    image = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    return {
        "labeled": {
            "inputs": {"image": image},
            "target": target,
            "valid": torch.ones(1, 4, 4, dtype=torch.bool),
        },
        "unlabeled": {"weak": {"image": weak}, "strong": {"image": strong}, "valid": valid},
    }


def _module(**options) -> fixmatch.Module:
    module = fixmatch.Module(model_chain=STAGES, **options)
    module.configure_model()
    return module


def test_unconfident_pseudo_labels_add_no_loss() -> None:
    module = _module()
    weak = torch.zeros(1, 2, 4, 4)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)

    loss = module.training_step(_batch(weak, torch.randn(1, 2, 4, 4), valid), 0)

    assert loss.item() == pytest.approx(math.log(2))


def test_confident_pseudo_labels_train_the_strong_view_over_valid_pixels() -> None:
    module = _module(unlabeled_weight=2.0)
    weak = torch.zeros(1, 2, 4, 4)
    weak[:, 1] = 10.0
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[:, :2] = False

    loss = module.training_step(_batch(weak, torch.zeros(1, 2, 4, 4), valid), 0)

    # Labeled: log 2. Unlabeled: log 2 on the 8 valid pixels of 8, weighted 2.
    assert loss.item() == pytest.approx(3 * math.log(2))
    loss.backward()
    assert module.model.get_submodule("model").factor.grad is not None


def test_a_pixel_the_weak_view_lacks_carries_no_pseudo_label() -> None:
    module = _module()
    weak = torch.zeros(1, 2, 4, 4)
    weak[:, 1] = 10.0
    valid = torch.zeros(1, 4, 4, dtype=torch.bool)

    loss = module.training_step(_batch(weak, torch.zeros(1, 2, 4, 4), valid), 0)

    assert loss.item() == pytest.approx(math.log(2))


class _Batches(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.randn(2, 4, 4)
        return {
            "labeled": {
                "inputs": {"image": image},
                "target": torch.zeros(4, 4, dtype=torch.long),
                "valid": torch.ones(4, 4, dtype=torch.bool),
            },
            "unlabeled": {
                "weak": {"image": image},
                "strong": {"image": image + 0.1},
                "valid": torch.ones(4, 4, dtype=torch.bool),
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
