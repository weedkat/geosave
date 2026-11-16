from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave.ml.cli import GeosaveCLI
from modules.segmentation import supervised
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave.geodata.sampling import ChipLayout
from geosave.ml.chain import chain_step
from tests.templates.tasks.segmentation.supervised.test_data import (
    CRS,
    FLIP,
    TRANSFORM,
    _preprocessor,
)
from tests.templates.tasks.segmentation.supervised.test_data import (
    Chips as Labels,
)


class SegmentationModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def logits(
        self, image: torch.Tensor, offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = image * self.factor
        return logits if offset is None else logits + offset


class MappedSegmentationModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(outputs=("logits",))
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.factor


class Samples(Dataset):
    def __init__(self, samples: list[supervised.Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> supervised.Sample:
        return self.samples[index]


@pytest.fixture
def stages() -> dict[str, dict[str, object]]:
    return {
        "model": {
            "class_path": f"{__name__}.SegmentationModel",
            "init_args": {"num_classes": 2},
        }
    }


def build_module(model_chain, **changes) -> supervised.Module:
    """Return a Module with every argument stated, as the YAML config states it."""
    arguments = {
        "model_chain": model_chain,
        "ignore_index": 255,
        "criterion": {"name": "cross_entropy", "init_args": {"ignore_index": 255}},
        "optimizer": {"name": "adamw", "init_args": {"lr": 1e-3}},
        "lr_scheduler": None,
        "metrics": ["iou.macro"],
        "predictor": {"name": "dense", "init_args": {"size": 2}},
    }
    arguments.update(changes)
    return supervised.Module(**arguments)


def test_ignored_targets_produce_zero_loss_and_no_metric_update(stages):
    task = build_module(stages)
    task.configure_model()
    batch = {
        "inputs": {"image": torch.ones(1, 2, 2, 2)},
        "target": torch.full((1, 2, 2), 255, dtype=torch.long),
        "transform": TRANSFORM[None],
        "crs": [CRS],
    }

    loss = task.training_step(batch, 0)
    loss.backward()

    assert loss.item() == 0.0
    assert task.model.get_submodule("model").factor.grad.item() == 0.0
    assert all(metric.update_count == 0 for metric in task.train_metrics.values())


class UncountedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.factor


def test_the_last_stage_states_the_class_count() -> None:
    task = build_module(
        {
            "model": {
                "class_path": f"{__name__}.SegmentationModel",
                "init_args": {"num_classes": 3},
            }
        },
    )

    task.configure_model()

    assert task.num_classes == 3
    assert len(task.val_metrics) > 0


def test_optimizer_keeps_direct_chain_parameters_in_default_group(stages):
    task = build_module(
        stages,
        optimizer={
            "class_path": "torch.optim.SGD",
            "init_args": {"lr": 0.1},
            "groups": {"model": {"lr": 0.01}},
        },
    )
    task.configure_model()
    bias = nn.Parameter(torch.tensor(1.0))
    task.model.register_parameter("bias", bias)
    task.model.register_parameter(
        "frozen", nn.Parameter(torch.tensor(1.0), requires_grad=False)
    )

    optimizer = task.configure_optimizers()

    assert len(optimizer.param_groups) == 2
    stage, remaining = optimizer.param_groups
    assert stage["params"] == [task.model.get_submodule("model").factor]
    assert len(remaining["params"]) == 1
    assert remaining["params"][0] is bias
    assert stage["lr"] == 0.01
    assert remaining["lr"] == 0.1
    bias.grad = torch.tensor(2.0)
    optimizer.step()
    torch.testing.assert_close(bias, torch.tensor(0.8))


def test_forward_preserves_prepared_inputs_and_routes_context(
    stages: dict[str, dict[str, object]],
) -> None:
    task = build_module(stages)
    task.configure_model()
    image = torch.tensor([[[[0.0, 1000.0], [2000.0, 3000.0]]] * 2], dtype=torch.float64)
    original = image.clone()
    offset = torch.tensor(4.0)

    actual = task(image=image, offset=offset)

    torch.testing.assert_close(actual, image + offset)
    torch.testing.assert_close(image, original)
    assert actual.dtype == torch.float64


def test_steps_accept_sample_batches(
    stages: dict[str, dict[str, object]],
) -> None:
    task = build_module(stages)
    task.configure_model()
    image = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]])
    target = torch.tensor([[[0, 1], [1, 0]]])

    loss = task.training_step(
        {
            "inputs": {"image": image},
            "target": target,
            "transform": TRANSFORM[None],
            "crs": [CRS],
        },
        0,
    )

    expected = torch.nn.functional.cross_entropy(image, target)
    torch.testing.assert_close(loss, expected)


def test_training_and_checkpoint_reload_preserve_construction(
    stages: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    task = build_module(
        stages,
        ignore_index=-1,
        optimizer={"class_path": "torch.optim.SGD", "init_args": {"lr": 0.1}},
        criterion={
            "class_path": "torch.nn.CrossEntropyLoss",
            "init_args": {"ignore_index": -1},
        },
        lr_scheduler={
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {"T_max": 2},
            "interval": "step",
        },
    )
    image = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    loader = DataLoader(
        Samples(
            [
                {
                    "inputs": {"image": image, "offset": torch.tensor(0.0)},
                    "target": torch.tensor([[0, 1], [1, 0]]),
                    "transform": TRANSFORM,
                    "crs": CRS,
                }
            ]
        ),
        batch_size=1,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(task, train_dataloaders=loader)
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = supervised.Module.load_from_checkpoint(checkpoint, weights_only=False)

    assert trainer.global_step == 1
    model = task.model.get_submodule("model")
    assert isinstance(model, SegmentationModel)
    assert model.factor.item() > 1.0
    torch.testing.assert_close(
        restored(image=image.unsqueeze(0), offset=torch.tensor(0.0)),
        task(image=image.unsqueeze(0), offset=torch.tensor(0.0)),
    )
    assert restored.hparams["model_chain"] == stages
    assert restored.hparams == task.hparams
    configured = restored.configure_optimizers()
    assert isinstance(configured["optimizer"], torch.optim.SGD)
    assert configured["lr_scheduler"]["interval"] == "step"
    assert restored.criterion.ignore_index == -1


def test_lightning_cli_parses_nested_construction_specs(
    stages: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    cli = GeosaveCLI(
        supervised.Module,
        run=False,
        save_config_callback=None,
        seed_everything_default=False,
        auto_configure_optimizers=False,
        args={
            "model": {
                "model_chain": stages,
                "ignore_index": -1,
                "metrics": ["iou.macro"],
                "predictor": {"name": "dense", "init_args": {"size": 2}},
                "criterion": {
                    "class_path": "torch.nn.CrossEntropyLoss",
                    "init_args": {"ignore_index": -1},
                },
                "optimizer": {
                    "class_path": "torch.optim.SGD",
                    "init_args": {"lr": 0.1},
                    "groups": {"model": {"lr": 0.01}},
                },
                "lr_scheduler": {
                    "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "logger": False,
                "enable_progress_bar": False,
                "default_root_dir": str(tmp_path),
            },
        },
    )
    task = cli.model
    assert isinstance(task, supervised.Module)
    task.configure_model()
    configured = task.configure_optimizers()
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert task.criterion.ignore_index == -1
    assert configured["lr_scheduler"]["monitor"] == "val_loss"
    config = yaml.safe_load(cli.parser.dump(cli.config))
    assert isinstance(config, dict)
    assert config["model"]["optimizer"]["init_args"] == {"lr": 0.1}


def _evaluation(stages, tmp_path):
    """Return a module, two chips whose second band leads by 1.0, and a trainer."""
    task = build_module(stages)
    image = torch.stack([torch.zeros(2, 2), torch.ones(2, 2)])
    loader = DataLoader(
        Samples(
            [
                {
                    "inputs": {"image": image},
                    "target": torch.tensor([[label, label], [label, 255]]),
                    "transform": TRANSFORM,
                    "crs": CRS,
                }
                for label in (0, 1)
            ]
        ),
        batch_size=2,
    )
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    return task, loader, trainer


def test_validation_scores_chips_over_their_known_labels(stages, tmp_path):
    task, loader, trainer = _evaluation(stages, tmp_path)

    validation = trainer.validate(task, dataloaders=loader, verbose=False)

    # Class 1 is predicted everywhere: wrong on the chip labelled 0, right on
    # the one labelled 1; the ignored label of each is left out.
    expected = (
        torch.log1p(torch.tensor(1.0).exp()) + torch.log1p(torch.tensor(-1.0).exp())
    ) / 2
    assert validation[0]["val_loss"] == pytest.approx(expected.item(), abs=1e-4)
    assert validation[0]["val_iou_macro"] == pytest.approx(0.25)


@pytest.mark.parametrize("mode", ["reflect", "drop"])
def test_validation_chips_and_merges_scenes_larger_than_a_chip(stages, tmp_path, mode):
    # The image is the one-hot label, so the identity model predicts it exactly.
    labels = torch.tensor([[0, 0, 1, 1, 1]] * 5)
    image = torch.nn.functional.one_hot(labels, 2).permute(2, 0, 1).float() * 4
    scene = {
        "inputs": {"image": image},
        "target": labels,
        "transform": TRANSFORM,
        "crs": CRS,
    }
    task = build_module(
        stages,
        predictor={"name": "dense", "init_args": {"size": 2, "mode": mode}},
    )
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    loader = DataLoader(Samples([scene]), batch_size=1)
    validation = trainer.validate(task, dataloaders=loader, verbose=False)

    # "drop" leaves the last row and column unpredicted; they are not scored.
    assert validation[0]["val_iou_macro"] == pytest.approx(1.0)
    assert validation[0]["val_loss"] < 0.05


class GridModel(nn.Module):
    """A stage that declares the grid entries, as a location-aware encoder does."""

    grids: list[GeoBox] = []

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def logits(
        self, image: torch.Tensor, transform: torch.Tensor, crs: list[str]
    ) -> torch.Tensor:
        for coefficients, code in zip(transform.tolist(), crs, strict=True):
            GridModel.grids.append(
                GeoBox(image.shape[-2:], Affine(*coefficients), code)
            )
        return image * self.factor


def test_a_stage_reading_the_grid_receives_each_chips_own(tmp_path):
    GridModel.grids.clear()
    scene = {
        "inputs": {"image": torch.zeros(2, 4, 4)},
        "target": torch.zeros(4, 4, dtype=torch.long),
        "transform": TRANSFORM,
        "crs": CRS,
    }
    task = build_module(
        {
            "model": {
                "class_path": f"{__name__}.GridModel",
                "init_args": {"num_classes": 2},
            }
        }
    )
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.validate(task, dataloaders=DataLoader(Samples([scene])), verbose=False)

    scene_grid = GeoBox((4, 4), Affine(*TRANSFORM.tolist()), CRS)
    layout = ChipLayout([scene_grid], size=2)
    expected = []
    for index in range(len(layout)):
        expected.append(layout.geobox(index))
    assert GridModel.grids == expected


def test_predict_returns_logits_for_chips_without_labels(stages, tmp_path):
    task, _, trainer = _evaluation(stages, tmp_path)
    image = torch.stack([torch.zeros(2, 2), torch.ones(2, 2)])
    sample = {"inputs": {"image": image}, "transform": TRANSFORM, "crs": CRS}
    loader = DataLoader(Samples([sample] * 3), batch_size=2)

    batches = trainer.predict(task, dataloaders=loader)

    assert [tuple(logits.shape) for logits in batches] == [(2, 2, 2, 2), (1, 2, 2, 2)]


class SparseLabels(Dataset[supervised.Sample]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> supervised.Sample:
        return {
            "inputs": {"image": torch.ones(2, 2, 2)},
            "target": torch.full((2, 2), 255 if index == 0 else 0, dtype=torch.long),
            "transform": TRANSFORM,
            "crs": CRS,
        }


def test_trainer_consumes_model_ready_samples(tmp_path, stages):
    data = supervised.DataModule(
        _preprocessor(),
        Labels(),
        Labels(),
        augmentations=FLIP,
        batch_size=2,
        num_workers=0,
    )
    module = build_module(stages)
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
    assert set(Labels()[0]) == {"inputs", "target", "transform", "crs"}


@pytest.mark.slow
def test_distributed_metrics_include_ranks_without_known_targets(tmp_path, stages):
    from datetime import timedelta

    trainer = Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="forkserver", timeout=timedelta(seconds=15)),
        max_epochs=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(
        build_module(stages),
        train_dataloaders=DataLoader(SparseLabels(), batch_size=1, shuffle=False),
        val_dataloaders=DataLoader(SparseLabels(), batch_size=1, shuffle=False),
    )
    assert torch.isfinite(trainer.callback_metrics["train_iou_macro"])
    assert torch.isfinite(trainer.callback_metrics["val_iou_macro"])
