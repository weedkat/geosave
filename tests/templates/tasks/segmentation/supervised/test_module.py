from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave.ml.torch.training.cli import GeosaveCLI
from generated_kits.segmentation.supervised import training as supervised

from tests.templates.tasks.segmentation.supervised.test_data import FLIP
from tests.templates.tasks.segmentation.supervised.test_data import (
    Chips as Labels,
)


class SegmentationModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.factor


class OffsetSegmentationModel(nn.Module):
    """Read an `offset` context input besides the image."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return image * self.factor + offset


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


@pytest.fixture
def context_stages() -> dict[str, dict[str, object]]:
    return {
        "model": {
            "class_path": f"{__name__}.OffsetSegmentationModel",
            "init_args": {"num_classes": 2},
        }
    }


def build_module(model_chain, **changes) -> supervised.Module:
    """Return a Module with every argument stated, as the YAML config states it."""
    arguments = {
        "model_chain": model_chain,
        "ignore_index": 255,
        "criterion": nn.CrossEntropyLoss(ignore_index=255),
        "optimizer": partial(torch.optim.AdamW, lr=1e-3),
        "param_groups": None,
        "lr_scheduler": None,
        "lr_scheduler_config": None,
        "metrics": ["iou_macro"],
    }
    arguments.update(changes)
    return supervised.Module(**arguments)


def test_ignored_targets_produce_zero_loss_and_no_metric_update(stages):
    task = build_module(stages)
    task.configure_model()
    batch = {
        "inputs": {"image": torch.ones(1, 2, 2, 2)},
        "target": torch.full((1, 2, 2), 255, dtype=torch.long),
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
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
        optimizer=partial(torch.optim.SGD, lr=0.1),
        param_groups={"model": {"lr": 0.01}},
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
    context_stages: dict[str, dict[str, object]],
) -> None:
    task = build_module(context_stages)
    task.configure_model()
    image = torch.tensor([[[[0.0, 1000.0], [2000.0, 3000.0]]] * 2], dtype=torch.float64)
    original = image.clone()
    offset = torch.tensor(4.0)

    actual = task(image=image, offset=offset)

    torch.testing.assert_close(actual, image + offset)
    torch.testing.assert_close(image, original)
    assert actual.dtype == torch.float64


def test_training_step_routes_context_and_ignores_unlabeled_pixels(
    context_stages: dict[str, dict[str, object]],
) -> None:
    task = build_module(context_stages)
    task.configure_model()
    image = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]])
    target = torch.tensor([[[0, 1], [1, 255]]])
    offset = torch.tensor([0.0, 0.5]).reshape(1, 2, 1, 1)
    expected = torch.nn.functional.cross_entropy(
        image + offset, target, ignore_index=255
    )

    changed = image.clone()
    changed[0, :, 1, 1] = torch.tensor([-50.0, 50.0])
    for scores in (image, changed):
        task.train_metrics.reset()
        batch = {"inputs": {"image": scores, "offset": offset}, "target": target}
        torch.testing.assert_close(task.training_step(batch, 0), expected)
        assert task.train_metrics.compute()["train_iou_macro"].item() == pytest.approx(
            1.0
        )


def test_training_and_checkpoint_reload_preserve_construction(
    context_stages: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    task = build_module(
        context_stages,
        ignore_index=-1,
        optimizer=partial(torch.optim.SGD, lr=0.1),
        criterion=nn.CrossEntropyLoss(ignore_index=-1),
        lr_scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=2),
        lr_scheduler_config={"interval": "step"},
    )
    image = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    loader = DataLoader(
        Samples(
            [
                {
                    "inputs": {"image": image, "offset": torch.tensor(0.0)},
                    "target": torch.tensor([[0, 1], [1, 0]]),
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
    assert isinstance(model, OffsetSegmentationModel)
    assert model.factor.item() > 1.0
    torch.testing.assert_close(
        restored(image=image.unsqueeze(0), offset=torch.tensor(0.0)),
        task(image=image.unsqueeze(0), offset=torch.tensor(0.0)),
    )
    assert restored.hparams["model_chain"] == context_stages
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
                "metrics": ["iou_macro"],
                "criterion": {
                    "class_path": "torch.nn.CrossEntropyLoss",
                    "init_args": {"ignore_index": -1},
                },
                "optimizer": {
                    "class_path": "torch.optim.SGD",
                    "init_args": {"lr": 0.1},
                },
                "param_groups": {"model": {"lr": 0.01}},
                "lr_scheduler": {
                    "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                },
                "lr_scheduler_config": {"monitor": "val_loss", "interval": "epoch"},
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
    assert config["model"]["optimizer"]["init_args"]["lr"] == 0.1


@pytest.mark.parametrize("step", ["validation_step", "test_step"])
def test_evaluation_requires_the_project_input_contract(stages, step):
    task = build_module(stages)
    with pytest.raises(NotImplementedError, match="input contract"):
        getattr(task, step)({}, 0)


def test_predict_returns_logits_for_chips_without_labels(stages, tmp_path):
    task = build_module(stages)
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    image = torch.stack([torch.zeros(2, 2), torch.ones(2, 2)])
    sample = {"inputs": {"image": image}}
    loader = DataLoader([sample] * 3, batch_size=2)

    batches = trainer.predict(task, dataloaders=loader)

    assert [tuple(logits.shape) for logits in batches] == [(2, 2, 2, 2), (1, 2, 2, 2)]


class SparseLabels(Dataset[supervised.Sample]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> supervised.Sample:
        return {
            "inputs": {"image": torch.ones(2, 2, 2)},
            "target": torch.full((2, 2), 255 if index == 0 else 0, dtype=torch.long),
        }


def test_trainer_consumes_model_ready_samples(tmp_path, stages):
    data = supervised.DataModule(
        Labels(),
        Labels(),
        augmentations=FLIP,
        batch_size=2,
        num_workers=0,
    )
    module = build_module(stages)
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
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
    )
    assert torch.isfinite(trainer.callback_metrics["train_iou_macro"])
