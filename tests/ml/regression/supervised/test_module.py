from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave_engine.geodata import raster, stack
from geosave_engine.ml.regression import supervised
from geosave_engine.model.chain import chain_step
from geosave_engine.model.spec import ChipsSpec
from tests.ml.test_inputs import _raster, _samples


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
    batch = (
        {"image": torch.ones(1, 2, 2, 2)},
        target,
        [0],
        torch.ones(1, 2, 2, dtype=torch.bool),
    )

    loss = module.training_step(batch, 0)

    # Errors 2, 0, 0 over the three known pixels.
    assert loss.item() == pytest.approx(4 / 3)


def test_the_last_stage_names_its_variables() -> None:
    module = supervised.Module(
        model_chain={"model": {"class_path": f"{__name__}.Unnamed"}}
    )

    with pytest.raises(TypeError, match="names no variables"):
        module.configure_model()


def _merge_dataset(parents, tmp_path):
    """Cut 4x4 chips with 2 pixels of overlap, as a Dataset exposes them."""
    samples = _samples(parents, (4, 4), tmp_path, overlap=2)
    return SimpleNamespace(
        frames=samples.frames,
        windows=samples.windows,
        target="target",
        spec=SimpleNamespace(chips=ChipsSpec(size=4, overlap=2, window="hann")),
    )


def test_completed_raster_reads_original_values_instead_of_blending_chips(
    tmp_path,
) -> None:
    image = _raster(6, 8)
    values = np.arange(48, dtype="float32").reshape(6, 8)
    values[0, 0] = np.nan
    target = raster({"target": (("y", "x"), values)}, image.gs.geobox, nodata=np.nan)
    dataset = _merge_dataset({"a": stack({"image": image, "target": target})}, tmp_path)
    module = _module()
    module._start_evaluation()
    ids = list(reversed(range(len(dataset.windows))))
    predictions = torch.full((len(ids), 1, 4, 4), 2.0)
    valid = torch.ones(len(ids), 4, 4, dtype=torch.bool)

    completed = module._merge(predictions, ids, valid, dataset)

    assert len(completed) == 1
    merged, truth = completed[0]
    torch.testing.assert_close(merged, torch.full((1, 1, 6, 8), 2.0))
    assert truth[0, 0, 0, 0].isnan()
    assert truth[0, 0, 5, 7] == 47.0
    assert module._raster_count == 1


def test_a_chip_added_twice_is_refused(tmp_path) -> None:
    image = _raster(6, 8)
    target = raster(
        {"target": (("y", "x"), np.zeros((6, 8), "float32"))}, image.gs.geobox
    )
    dataset = _merge_dataset({"a": stack({"image": image, "target": target})}, tmp_path)
    module = _module()
    module._start_evaluation()
    predictions = torch.zeros(1, 1, 4, 4)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    module._merge(predictions, [0], valid, dataset)

    with pytest.raises(ValueError, match="already accumulated"):
        module._merge(predictions, [0], valid, dataset)


class _Chips(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.rand(2, 4, 4)
        return (
            {"image": image},
            image[:1] * 2,
            index,
            torch.ones(4, 4, dtype=torch.bool),
        )


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

    trainer.fit(module, train_dataloaders=DataLoader(_Chips(), batch_size=2))

    assert {"train_loss", "train_mae", "train_rmse"} <= set(trainer.callback_metrics)
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
