"""Generated segmentation workspaces compose library Lightning classes."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from geosave.cli.core.templates import get_tasks
from geosave.cli.core.workspace import add_task, create_workspace
from geosave.ml.cli import GeosaveCLI
from geosave.ml.tasks.segmentation import supervised
from geosave.model.spec import ModelSpec, Ref


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    create_workspace(tmp_path)
    add_task(tmp_path, "segmentation-supervised")
    assert (tmp_path / "configs/segmentation-supervised.yaml").is_file()
    assert not (tmp_path / "configs/model.yaml").exists()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_segmentation_configs_agree_on_model_inputs(workspace):
    spec = ModelSpec.load(workspace / "configs/model_spec.yaml")
    config = yaml.safe_load(
        (workspace / "configs/segmentation-supervised.yaml").read_text()
    )
    requirement = spec.rasters["sentinel_2_l2a"]
    assert requirement.stac is not None
    assert requirement.stac.collection == "sentinel-2-l2a"
    assert requirement.variables == ("B02", "B03", "B04", "B08")
    assert tuple(map(str, requirement.stac.endpoints)) == (
        "planetary_computer",
        "cdse",
    )
    assert requirement.require_crs
    supervised.Module(**config["model"]["init_args"])
    encoder = config["model"]["init_args"]["model_chain"]["encoder"]["init_args"]
    assert encoder["in_channels"] == len(requirement.variables) == 4
    head = config["model"]["init_args"]["model_chain"]["head"]
    assert head == {
        "name": "segmentation",
        "init_args": {"num_classes": 2},
    }
    assert config["data"]["class_path"] == (
        "geosave.ml.tasks.segmentation.supervised.DataModule"
    )
    data = supervised.DataModule(**config["data"]["init_args"])
    assert data.spec == spec
    assert spec.chips is not None
    assert spec.chips.shape == (encoder["input_size"],) * 2
    assert spec.pixel_inputs == ("image",)
    assert len(spec.transforms["image"][0].init_args["mean"]) == 4
    assert [step["name"] for step in config["data"]["init_args"]["augmentations"]] == [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
    ]
    assert not (workspace / "modules" / "data.py").exists()
    assert not (workspace / "configs" / "augmentation.yaml").exists()


def test_model_config_uses_current_task_constructor(workspace):
    config = yaml.safe_load(
        (workspace / "configs/segmentation-supervised.yaml").read_text()
    )
    model = supervised.Module(**config["model"]["init_args"])
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)
    assert model.optimizer_spec["name"] == "adamw"
    assert model.scheduler_spec["name"] == "cosine_annealing"


def test_generated_entrypoint_keeps_task_optimizer_ownership(workspace, monkeypatch):
    options = {}

    def observe_cli(self, **kwargs):
        options.update(kwargs)

    # Stop at the CLI boundary: entry-point execution must not start training.
    monkeypatch.setattr(GeosaveCLI, "__init__", observe_cli)
    runpy.run_path(str(workspace / "main.py"), run_name="__main__")
    assert options["auto_configure_optimizers"] is False
    assert options["subclass_mode_model"] is True
    assert options["subclass_mode_data"] is True


def test_generated_entrypoint_parses_library_lightning_pair(
    workspace, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "fit",
            "--config",
            str(workspace / "configs/segmentation-supervised.yaml"),
            "--print_config",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(workspace / "main.py"), run_name="__main__")
    assert stopped.value.code == 0
    config = yaml.safe_load(capsys.readouterr().out)
    chain = config["model"]["init_args"]["model_chain"]
    assert chain["encoder"]["init_args"] == {"in_channels": 4, "input_size": 224}
    assert config["data"]["init_args"]["spec"] == "configs/model_spec.yaml"
    assert config["data"]["class_path"] == (
        "geosave.ml.tasks.segmentation.supervised.DataModule"
    )


@pytest.mark.slow
def test_lightning_boilerplate_runs_one_train_and_validation_batch(tmp_path):
    assert "lightning-boilerplate" in get_tasks()
    create_workspace(tmp_path)
    add_task(tmp_path, "lightning-boilerplate")
    assert (tmp_path / "configs/lightning-boilerplate.yaml").is_file()
    assert not (tmp_path / "configs/model.yaml").exists()
    script = """
from modules.data import CustomDataModule
from modules.task import CustomTask
from geosave.ml.cli import GeosaveCLI

cli = GeosaveCLI(
    run=False,
    args=["--config", "configs/lightning-boilerplate.yaml"],
    save_config_callback=None,
    auto_configure_optimizers=False,
    subclass_mode_model=True,
    subclass_mode_data=True,
)
assert isinstance(cli.model, CustomTask)
assert isinstance(cli.datamodule, CustomDataModule)
cli.datamodule.setup("fit")
train_batch = next(iter(cli.datamodule.train_dataloader()))
validation_batch = next(iter(cli.datamodule.val_dataloader()))
assert cli.model.training_step(train_batch, 0).ndim == 0
assert cli.model.validation_step(validation_batch, 0).ndim == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def regression_workspace(tmp_path, monkeypatch):
    create_workspace(tmp_path)
    add_task(tmp_path, "regression-supervised")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_regression_swaps_only_the_head_loss_metrics_and_target(
    regression_workspace,
):
    from geosave.ml.tasks.regression import supervised as regression

    config = yaml.safe_load(
        (regression_workspace / "configs/regression-supervised.yaml").read_text()
    )
    chain = config["model"]["init_args"]["model_chain"]
    assert chain["head"] == {
        "name": "regression",
        "init_args": {"variables": ["target"]},
    }
    assert chain["encoder"]["name"] == "dinov3"
    module = regression.Module(**config["model"]["init_args"])
    assert isinstance(module.criterion, torch.nn.MSELoss)
    data = config["data"]
    assert data["class_path"] == "geosave.ml.tasks.regression.supervised.DataModule"
    assert data["init_args"]["spec"] == "configs/model_spec.yaml"
    assert set(data["init_args"]["train"]) == {"class_path", "init_args"}


def test_shipped_model_spec_round_trips_as_inert_declarations(tmp_path):
    path = (
        Path(__file__).parents[3]
        / "src/geosave/templates/tasks/segmentation-supervised/configs/model_spec.yaml"
    )

    spec = ModelSpec.load(path)
    restored = ModelSpec.load(spec.save(tmp_path / "model_spec.yaml"))

    assert restored == spec
    assert spec.rasters["sentinel_2_l2a"].variables == (
        "B02",
        "B03",
        "B04",
        "B08",
    )
    assert spec.preprocessing["image"].call == Ref("sentinel_2_l2a.gs.mask_and_scale")
    assert spec.preprocessing["image"].kwargs == {}


def test_walkthrough_runs_without_model_or_external_service(tmp_path):
    create_workspace(tmp_path)
    add_task(tmp_path, "segmentation-supervised")

    result = subprocess.run(
        [sys.executable, "scripts/prepare_example.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "B08" in result.stdout
    assert "0.6" in result.stdout
