from __future__ import annotations

import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave_engine.ml.tasks.classification import supervised
from geosave_engine.model.chain import chain_step


class Classifier(nn.Module):
    """Average each band over the chip and project the averages to classes."""

    def __init__(self, num_classes: int, bands: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.projection = nn.Linear(bands, num_classes)

    @chain_step(head=True)
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(image.mean(dim=(-2, -1)))


STAGES = {
    "model": {"class_path": f"{__name__}.Classifier", "init_args": {"num_classes": 3}}
}


def test_defaults_and_class_count() -> None:
    module = supervised.Module(model_chain=STAGES)
    module.configure_model()

    assert module.num_classes == 3
    assert isinstance(module.criterion, nn.CrossEntropyLoss)
    assert isinstance(module.configure_optimizers(), torch.optim.AdamW)


def test_a_last_stage_without_a_class_count_is_refused() -> None:
    module = supervised.Module(
        model_chain={"model": {"class_path": "torch.nn.Identity"}}
    )

    with pytest.raises(TypeError):
        module.configure_model()


class _Chips(Dataset):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int):
        image = torch.full((2, 4, 4), float(index % 3))
        return {
            "inputs": {"image": image},
            "target": torch.tensor(index % 3),
            "valid": torch.ones(4, 4, dtype=torch.bool),
        }


def test_lightning_fits_validates_and_tests(tmp_path) -> None:
    module = supervised.Module(model_chain=STAGES, metrics=["accuracy.macro"])
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    loader = DataLoader(_Chips(), batch_size=3)

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    fitted = set(trainer.callback_metrics)
    result = trainer.test(module, dataloaders=loader, verbose=False)

    assert {"train_loss", "val_loss", "val_accuracy_macro"} <= fitted
    assert 0.0 <= result[0]["test_accuracy_macro"] <= 1.0

