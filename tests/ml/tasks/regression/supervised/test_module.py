from __future__ import annotations

import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave_engine.ml.tasks.regression import supervised
from geosave_engine.model.chain import chain_step


class Scale(nn.Module):
    """Predict the first band times a learnt factor."""

    def __init__(self) -> None:
        super().__init__()
        self.variables = ["biomass"]
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def values(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :1] * self.factor


class Unnamed(nn.Module):
    """Predict without naming its variables."""

    @chain_step(head=True)
    def values(self, image: torch.Tensor) -> torch.Tensor:
        return image


STAGES = {"model": {"class_path": f"{__name__}.Scale"}}


def _module() -> supervised.Module:
    module = supervised.Module(model_chain=STAGES)
    module.configure_model()
    return module


def test_only_known_pixels_are_scored() -> None:
    module = _module()
    target = torch.tensor([[[[3.0, torch.nan], [1.0, 1.0]]]])
    batch: supervised.Sample = {
        "inputs": {"image": torch.ones(1, 2, 2, 2)},
        "target": target,
        "valid": torch.ones(1, 2, 2, dtype=torch.bool),
    }

    loss = module.training_step(batch, 0)

    # Errors 2, 0, 0 over the three known pixels.
    assert loss.item() == pytest.approx(4 / 3)


def test_the_last_stage_names_its_variables() -> None:
    module = supervised.Module(
        model_chain={"model": {"class_path": f"{__name__}.Unnamed"}}
    )

    with pytest.raises(TypeError, match="names no variables"):
        module.configure_model()


def test_evaluation_scores_known_pixels_with_valid_inputs() -> None:
    module = _module()
    batch: supervised.Sample = {
        "inputs": {"image": torch.ones(1, 2, 2, 2)},
        "target": torch.tensor([[[[3.0, torch.nan], [5.0, 1.0]]]]),
        "valid": torch.tensor([[[True, True], [False, True]]]),
    }

    predictions, target = module._evaluate(batch)

    assert target.tolist() == [3.0, 1.0]
    assert predictions.tolist() == [1.0, 1.0]


class _Chips(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.rand(2, 4, 4)
        return {
            "inputs": {"image": image},
            "target": image[:1] * 2,
            "valid": torch.ones(4, 4, dtype=torch.bool),
        }


def test_lightning_fits(tmp_path) -> None:
    module = supervised.Module(
        model_chain=STAGES, optimizer={"name": "sgd", "init_args": {"lr": 0.5}}
    )
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=5,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    loader = DataLoader(_Chips(), batch_size=2)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert {"train_loss", "train_mae", "val_loss", "val_rmse"} <= set(
        trainer.callback_metrics
    )
    assert module.model.get_submodule("model").factor.item() > 1.0


class EndToEnd(nn.Module):
    """An end-to-end regressor: forward only, no chain_step."""

    def __init__(self) -> None:
        super().__init__()
        self.variables = ["biomass"]
        self.layer = nn.Conv2d(2, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image)


def test_lightning_fits_an_end_to_end_model(tmp_path) -> None:
    module = supervised.Module(
        model_chain={"model": {"class_path": f"{__name__}.EndToEnd"}}
    )
    trainer = Trainer(
        accelerator="cpu",
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, train_dataloaders=DataLoader(_Chips(), batch_size=2))

    assert module.variables == ["biomass"]
    assert "train_loss" in trainer.callback_metrics
