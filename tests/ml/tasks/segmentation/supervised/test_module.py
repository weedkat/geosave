from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave.ml.cli import GeosaveCLI
from geosave.ml.tasks.segmentation import supervised
from geosave.model.chain import chain_step


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


def test_task_defaults(stages):
    task = supervised.Module(model_chain=stages, ignore_index=7)
    task.configure_model()
    assert task.num_classes == 2
    assert isinstance(task.criterion, nn.CrossEntropyLoss)
    assert task.criterion.ignore_index == 7
    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3


def test_ignored_targets_produce_zero_loss_and_no_metric_update(stages):
    task = supervised.Module(model_chain=stages)
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


def test_criterion_cannot_override_the_task_ignore_value(stages):
    with pytest.raises(ValueError, match="ignore_index"):
        supervised.Module(
            model_chain=stages,
            ignore_index=255,
            criterion={"name": "cross_entropy", "init_args": {"ignore_index": -1}},
        )


def test_the_module_takes_no_data_shaped_arguments(stages):
    for argument in ("in_channels", "input_size"):
        with pytest.raises(TypeError, match=argument):
            supervised.Module(model_chain=stages, **{argument: 2})


class UncountedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def logits(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.factor


def test_the_last_stage_states_the_class_count() -> None:
    task = supervised.Module(
        model_chain={
            "model": {
                "class_path": f"{__name__}.SegmentationModel",
                "init_args": {"num_classes": 3},
            }
        },
        class_thresholds=[0.2, 0.4, 0.6],
    )

    task.configure_model()

    assert task.num_classes == 3
    assert task.class_thresholds.tolist() == pytest.approx([0.2, 0.4, 0.6])
    assert len(task.val_metrics) > 0


def test_a_last_stage_stating_no_class_count_is_refused() -> None:
    task = supervised.Module(
        model_chain={"model": {"class_path": f"{__name__}.UncountedModel"}},
    )

    with pytest.raises(TypeError, match="UncountedModel.*num_classes"):
        task.configure_model()


def test_thresholds_must_number_the_classes(stages) -> None:
    task = supervised.Module(model_chain=stages, class_thresholds=[0.5])

    with pytest.raises(ValueError, match="must have 2 entries, got 1"):
        task.configure_model()


def test_task_uses_registered_training_builders(stages) -> None:
    task = supervised.Module(
        model_chain=stages,
        ignore_index=-1,
        criterion={"name": "cross_entropy", "init_args": {"ignore_index": -1}},
        optimizer={"name": "sgd", "init_args": {"lr": 0.1}},
        lr_scheduler={
            "name": "reduce_on_plateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
        },
    )
    task.configure_model()

    configured = task.configure_optimizers()

    assert isinstance(task.criterion, nn.CrossEntropyLoss)
    assert task.criterion.ignore_index == -1
    assert isinstance(configured["optimizer"], torch.optim.SGD)
    assert configured["lr_scheduler"]["monitor"] == "val_loss"


@pytest.mark.parametrize(
    "criterion, expected",
    [
        ({"class_path": "torch.nn.CrossEntropyLoss"}, 7),
        (
            {
                "class_path": "torch.nn.CrossEntropyLoss",
                "init_args": {"ignore_index": -1},
            },
            -1,
        ),
    ],
)
def test_criterion_ignore_index(stages, criterion, expected):
    task = supervised.Module(
        model_chain=stages,
        ignore_index=expected,
        criterion=criterion,
    )
    assert task.criterion.ignore_index == expected


class Encoder(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.frozen = nn.Parameter(torch.zeros(in_channels), requires_grad=False)

    @chain_step(outputs=("features",))
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.weight.reshape(1, -1, 1, 1)


class Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.ones(num_classes))

    @chain_step(head=True)
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.weight.reshape(1, -1, 1, 1)


def test_optimizer_exact_groups_and_remaining_trainable_parameters():
    task = supervised.Module(
        model_chain={
            "encoder": {"class_path": f"{__name__}.Encoder"},
            "head": {
                "class_path": f"{__name__}.Head",
                "init_args": {"num_classes": 2},
            },
        },
        optimizer={
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 1e-3, "weight_decay": 1e-2},
            "groups": {"head": {"lr": 1e-4, "weight_decay": 0.0}},
        },
    )
    task.configure_model()
    optimizer = task.configure_optimizers()
    assert len(optimizer.param_groups) == 2
    head, remaining = optimizer.param_groups
    assert head["params"] == [task.model.get_submodule("head").weight]
    assert remaining["params"] == [task.model.get_submodule("encoder").weight]
    assert (head["lr"], head["weight_decay"]) == (1e-4, 0.0)
    assert (remaining["lr"], remaining["weight_decay"]) == (1e-3, 1e-2)


def test_optimizer_keeps_direct_chain_parameters_in_default_group(stages):
    task = supervised.Module(
        model_chain=stages,
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


@pytest.mark.parametrize("group", ["mod", "MODEL", "model.factor"])
def test_unknown_optimizer_groups_fail_before_optimizer_construction(stages, group):
    task = supervised.Module(
        model_chain=stages,
        optimizer={
            "class_path": "torch.optim.SGD",
            "groups": {group: {"lr": 0.1}},
            "init_args": {"invalid_argument": True},
        },
    )
    task.configure_model()
    with pytest.raises(KeyError, match="Unknown model groups"):
        task.configure_optimizers()


def test_scheduler_metadata_and_plateau_monitoring(stages):
    task = supervised.Module(
        model_chain=stages,
        lr_scheduler={
            "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 3,
            "strict": False,
            "scheduler_name": "rate",
        },
    )
    task.configure_model()
    configured = task.configure_optimizers()
    scheduler = configured["lr_scheduler"].pop("scheduler")
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.optimizer is configured["optimizer"]
    assert scheduler.patience == 2
    assert configured["lr_scheduler"] == {
        "monitor": "val_loss",
        "interval": "epoch",
        "frequency": 3,
        "strict": False,
        "name": "rate",
    }


@pytest.mark.parametrize(
    "field, path",
    [
        ("criterion", "torch.optim.AdamW"),
        ("optimizer", "torch.nn.CrossEntropyLoss"),
        ("lr_scheduler", "torch.nn.CrossEntropyLoss"),
    ],
)
def test_training_specs_require_expected_torch_subclass(stages, field, path):
    with pytest.raises(TypeError, match="subclass"):
        task = supervised.Module(
            model_chain=stages,
            **{field: {"class_path": path}},
        )
        task.configure_model()
        task.configure_optimizers()


@pytest.mark.parametrize(
    "field, path",
    [
        ("criterion", "torch.nn.CrossEntropyLoss"),
        ("optimizer", "torch.optim.AdamW"),
        ("lr_scheduler", "torch.optim.lr_scheduler.StepLR"),
    ],
)
def test_training_constructor_errors_remain_native(stages, field, path):
    with pytest.raises(TypeError, match="unexpected keyword argument 'wrong'") as error:
        task = supervised.Module(
            model_chain=stages,
            **{field: {"class_path": path, "init_args": {"wrong": True}}},
        )
        task.configure_model()
        task.configure_optimizers()
    assert error.value.__cause__ is None


def test_forward_preserves_prepared_inputs_and_routes_context(
    stages: dict[str, dict[str, object]],
) -> None:
    task = supervised.Module(
        model_chain=stages,
    )
    task.configure_model()
    image = torch.tensor([[[[0.0, 1000.0], [2000.0, 3000.0]]] * 2], dtype=torch.float64)
    original = image.clone()
    offset = torch.tensor(4.0)

    actual = task(image=image, offset=offset)

    torch.testing.assert_close(actual, image + offset)
    torch.testing.assert_close(image, original)
    assert actual.dtype == torch.float64
    assert not hasattr(task, "preprocessor")
    assert not hasattr(task, "preprocess")


def test_forward_rejects_mapped_model_results() -> None:
    task = supervised.Module(
        model_chain={
            "model": {
                "class_path": f"{__name__}.MappedSegmentationModel",
                "init_args": {"num_classes": 2},
            }
        },
    )
    task.configure_model()
    image = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]])

    native = task.model(image=image)
    assert isinstance(native, dict)
    with pytest.raises(TypeError, match="must return logits as a tensor"):
        task(image=image)


def test_steps_accept_sample_batches(
    stages: dict[str, dict[str, object]],
) -> None:
    task = supervised.Module(
        model_chain=stages,
    )
    task.configure_model()
    image = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]])
    target = torch.tensor([[[0, 1], [1, 0]]])

    loss = task.training_step(
        {
            "inputs": {"image": image},
            "target": target,
        },
        0,
    )

    expected = torch.nn.functional.cross_entropy(image, target)
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    "batch",
    [
        (torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)),
        {"image": torch.zeros(1, 2, 2, 2), "label": torch.zeros(1, 2, 2)},
    ],
)
def test_incompatible_batches_raise_native_python_errors(
    stages: dict[str, dict[str, object]], batch: object
) -> None:
    task = supervised.Module(
        model_chain=stages,
    )
    task.configure_model()

    with pytest.raises((KeyError, TypeError)):
        task.training_step(batch, 0)  # type: ignore[arg-type]


def test_training_and_checkpoint_reload_preserve_construction(
    stages: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    task = supervised.Module(
        model_chain=stages,
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
    task = supervised.Module(model_chain=stages)
    image = torch.stack([torch.zeros(2, 2), torch.ones(2, 2)])
    loader = DataLoader(
        Samples(
            [
                {
                    "inputs": {"image": image},
                    "target": torch.tensor([[label, label], [label, 255]]),
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


def test_test_scores_chips_over_their_known_labels(stages, tmp_path):
    task, loader, trainer = _evaluation(stages, tmp_path)

    result = trainer.test(task, dataloaders=loader, verbose=False)

    assert result[0]["test_iou_macro"] == pytest.approx(0.25)


def test_evaluation_preserves_supplied_targets_without_input_validity(stages) -> None:
    task = supervised.Module(model_chain=stages, ignore_index=9)
    task.configure_model()
    batch: supervised.Sample = {
        "inputs": {"image": torch.zeros(1, 2, 2, 2)},
        "target": torch.tensor([[[0, 9], [0, 0]]]),
    }

    _, target = task._evaluate(batch)

    assert target is batch["target"]


def test_predict_returns_logits_for_chips_without_labels(stages, tmp_path):
    task, _, trainer = _evaluation(stages, tmp_path)
    image = torch.stack([torch.zeros(2, 2), torch.ones(2, 2)])
    loader = DataLoader(Samples([{"inputs": {"image": image}}] * 3), batch_size=2)

    batches = trainer.predict(task, dataloaders=loader)

    assert [tuple(logits.shape) for logits in batches] == [(2, 2, 2, 2), (1, 2, 2, 2)]
